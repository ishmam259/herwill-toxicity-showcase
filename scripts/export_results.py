"""Export verified historical artifacts into text-free public aggregates.

Run with --workspace pointing to the surrounding datathon workspace. Only the
explicit --private flag produces row-level text, outside all public paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {
    "V0.0": ("TF-IDF + Logistic Regression", "tfidf", False),
    "V1.2avg": ("BanglaBERT · seed average", "encoder", True),
    "V1.7": ("BanglaBERT Large", "encoder", True),
    "V1.9": ("IndicBERT v2", "encoder", False),
    "V2.2avg": ("XLM-R Large · seed average", "encoder", False),
    "V5.0": ("Qwen 2.5 7B · LoRA", "llm", False),
}
# Extra members read directly from the competition repo (not in the hashed bundle).
# Row order matches the bundle: Kernels/V5.0 preds are identical to members/V5.0.
EXTRA_MEMBERS = {
    "V1.3avg": ("MuRIL · seed average", "encoder", "V1.3avg_muril-2seed"),
    "V5.11": ("Qwen 2.5 7B · region tag · LoRA", "llm", "V5.11_qwen25-7b-region"),
    "V5.2avg": ("Qwen 3 14B · seed average", "llm", "V5.2avg_qwen3-14b-2seed"),
    "V5.3avg": ("Llama 3.1 8B · seed average", "llm", "V5.3avg_llama31-8b-2seed"),
}
LLM_IDS = {"V5.0", *(n for n, (_, f, _) in EXTRA_MEMBERS.items() if f == "llm")}
# Text ensemble shown on the pages: V3.11i, our final text-only submission (private 0.633).
ENSEMBLE_ID, ENSEMBLE_KERNEL = "V3.11i", "V3.11i_ensemble"
# "llm-only" hard cases: the best single LLM is right and every non-LLM member is wrong.
HARD_CASE_LLM = "V5.11"
OFFSETS = np.round(np.linspace(-0.5, 0.5, 21), 2)
WARNING = "Historical, adaptively inspected out-of-fold predictions. These diagnostics are not a fresh held-out evaluation. Specialist models cover Bangla-containing posts only; overall scores with different coverage are not directly comparable."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred):
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3)
    den = cm.sum(0) + cm.sum(1)
    f = np.divide(2 * cm.diagonal(), den, out=np.zeros(3), where=den > 0)
    return {
        "macro_f1": float(f.mean()),
        "per_class_f1": f.tolist(),
        "confusion": cm.tolist(),
        "support": cm.sum(1).tolist(),
        "rows": len(y),
    }


def reliability(y, p):
    conf, pred = p.max(1), p.argmax(1)
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (conf >= i / 10) & (conf < (i + 1) / 10 if i < 9 else conf <= 1.000001)
        count = int(mask.sum())
        c = float(conf[mask].mean()) if count else None
        a = float((pred[mask] == y[mask]).mean()) if count else None
        if count:
            ece += count / len(y) * abs(c - a)
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": count,
                "confidence": c,
                "accuracy": a,
            }
        )
    return {
        "bins": bins,
        "ece": ece,
        "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
    }


def export(workspace, output, private=False):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    # Never let row-level exports share the public output directory.
    if private and (output == ROOT / "private" or ROOT / "private" in output.parents):
        raise ValueError("Public aggregates must be outside private/")
    bundle = workspace / "kaggle/v6-replay-foundation"
    manifest = json.loads((bundle / "manifest.json").read_text())
    train_path = workspace / "dataset/train.csv"
    if sha(train_path) != manifest["official_data"]["train.csv"]["sha256"]:
        raise ValueError(
            "Training data hash differs from the verified artifact contract"
        )
    tr = pd.read_csv(train_path).sort_values("id").reset_index(drop=True)
    ids_path = bundle / "train_ids.npy"
    if sha(ids_path) != manifest["files"]["train_ids.npy"]["sha256"]:
        raise ValueError("Alignment array hash mismatch")
    if not np.array_equal(tr.id.to_numpy(), np.load(ids_path, allow_pickle=False)):
        raise ValueError("OOF row IDs do not match sorted training IDs")
    y = tr.y.to_numpy(dtype=int)
    if set(y) != {0, 1, 2} or not tr.id.is_unique:
        raise ValueError("Invalid labels or duplicate IDs")
    groups = np.where(tr.text.fillna("").str.contains("[\u0980-\u09ff]"), "B", "L")
    arrays, sources = {}, {}
    for name, (_, _, specialist) in MEMBERS.items():
        rel = f"members/{name}/oof_probs.npy"
        path = bundle / rel
        if sha(path) != manifest["files"][rel]["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")
        p = np.load(path, allow_pickle=False)
        if p.shape != (len(y), 3):
            raise ValueError(f"Shape mismatch: {name}")
        covered = np.isfinite(p).all(1)
        expected = groups == "B" if specialist else np.ones(len(y), dtype=bool)
        if not np.array_equal(covered, expected):
            raise ValueError(f"Coverage mismatch: {name}")
        if (p[covered] < 0).any() or not np.allclose(p[covered].sum(1), 1, atol=1e-5):
            raise ValueError(f"Invalid probability simplex: {name}")
        arrays[name], sources[name] = p.astype(float), sha(path)
    kernels = workspace / "repository/UAP-Datathon/Kernels"
    for name, (_, _, folder) in EXTRA_MEMBERS.items():
        path = kernels / folder / "preds/oof_probs.npy"
        p = np.load(path, allow_pickle=False)
        if p.shape != (len(y), 3) or not np.isfinite(p).all():
            raise ValueError(f"Shape or coverage mismatch: {name}")
        if (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError(f"Invalid probability simplex: {name}")
        arrays[name], sources[name] = p.astype(float), sha(path)
    ensemble = np.zeros((len(y), 3))
    weights = {
        "B": {
            "V0.0": 0.1,
            "V1.2avg": 0.4,
            "V1.7": 0.2,
            "V1.9": 0.1,
            "V2.2avg": 0.1,
            "V5.0": 0.1,
        },
        "L": {"V0.0": 0.2, "V2.2avg": 0.2, "V5.0": 0.6},
    }
    for group, recipe in weights.items():
        mask = groups == group
        ensemble[mask] = sum(w * arrays[name][mask] for name, w in recipe.items())
    # Alignment check: rebuild V3.7 from bundle members and compare to its saved probabilities.
    reference = (
        workspace / "repository/UAP-Datathon/Kernels/V3.7_ensemble/preds/oof_probs.npy"
    )
    if not np.allclose(ensemble, np.load(reference, allow_pickle=False), atol=1e-6):
        raise ValueError("Text ensemble differs from saved V3.7")
    # The pages show V3.11i (cross-fitted OOF probabilities saved by its kernel).
    path = kernels / ENSEMBLE_KERNEL / "preds/oof_probs.npy"
    p = np.load(path, allow_pickle=False)
    if p.shape != (len(y), 3) or not np.isfinite(p).all():
        raise ValueError(f"Shape or coverage mismatch: {ENSEMBLE_ID}")
    arrays[ENSEMBLE_ID], sources[ENSEMBLE_ID] = p.astype(float), sha(path)
    catalog = {
        **MEMBERS,
        **{n: (title, family, False) for n, (title, family, _) in EXTRA_MEMBERS.items()},
        ENSEMBLE_ID: ("Text ensemble · V3.11i", "ensemble", False)}
    models, calibration = [], {}
    for name, (title, family, specialist) in catalog.items():
        p = arrays[name]
        mask = np.isfinite(p).all(1)
        yy, pp = y[mask], p[mask]
        # Raw argmax for every model, including the ensemble, for a consistent comparison.
        summary = metrics(yy, pp.argmax(1))
        slices = {
            g: metrics(y[mask & (groups == g)], p[mask & (groups == g)].argmax(1))
            for g in ("B", "L")
            if (mask & (groups == g)).any()
        }
        models.append(
            {
                "id": name,
                "name": title,
                "family": family,
                "specialist": specialist,
                **summary,
                "by_script": slices,
                "source_sha256": sources.get(name),
                "inference": "historical-only",
                "decision_rule": "raw probability argmax",
            }
        )
        grid = []
        logp = np.log(np.maximum(pp, 1e-8))
        for a in OFFSETS:
            row = []
            for b in OFFSETS:
                m = metrics(yy, (logp + [0, a, b]).argmax(1))
                row.append(round(m["macro_f1"], 9))
            grid.append(row)
        calibration[name] = {
            **reliability(yy, pp),
            "macro_f1_grid": grid,
            "rows": len(yy),
        }
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "class_order": [0, 1, 2],
        "warning": WARNING,
        "rows": len(y),
        "source": "Verified local historical OOF artifacts",
        "data_sha256": sha(train_path),
        "script_definition": {
            "B": "Contains at least one Bangla character (including mixed script)",
            "L": "No Bangla characters; includes English, Banglish and other scripts",
        },
        "raw_text_included": False,
    }
    write(output / "models.json", {**provenance, "models": models})
    write(
        output / "calibration.json",
        {**provenance, "offsets": OFFSETS.tolist(), "models": calibration},
    )
    if private:
        rows = []
        votes = {
            n: np.where(np.isfinite(p).all(1), np.nan_to_num(p, nan=-1).argmax(1), -1)
            for n, p in arrays.items()
        }
        all_wrong = np.ones(len(y), bool)
        non_llm_wrong = np.ones(len(y), bool)
        for n, v in votes.items():
            all_wrong &= v != y
            if n not in LLM_IDS and n != ENSEMBLE_ID:
                non_llm_wrong &= v != y
        llm_only = (votes[HARD_CASE_LLM] == y) & non_llm_wrong
        for category, keep in [("all-wrong", all_wrong), ("llm-only", llm_only)]:
            for i in np.flatnonzero(keep)[:40]:
                rows.append(
                    {
                        "id": int(tr.id.iloc[i]),
                        "category": category,
                        "text": str(tr.text.iloc[i]),
                        "label": int(y[i]),
                        "script": groups[i],
                        "votes": [
                            {"id": n, "label": int(v[i]) if v[i] >= 0 else None}
                            for n, v in votes.items()
                        ],
                    }
                )
        write(
            ROOT / "private/hard_cases.json",
            {
                "warning": "Competition text: local use only. Never publish.",
                "cases": rows,
            },
        )
    print(
        f"Exported {len(models)} models, {len(y):,} aligned rows; public files contain aggregates only."
    )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    export(args.workspace, args.output, args.private)
