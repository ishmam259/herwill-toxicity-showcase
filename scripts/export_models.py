"""Export model results from the competition repo into App data files (Phase 0 contract, TASKS.md).

Writes, into private/legacy-exports/ by default:
  models.json       one entry per model: metrics, per-class / per-script F1, confusion matrices.
                    Merged by id, so each owner re-runs their own exporter without touching the others'.
  oof_rows.json     per train row (sorted by id): script, true label, fold.            [gitignored]
  oof_<id>.json     per train row: probs[3] or null where the model did not predict.  [gitignored]

The oof_* files are derived from the competition labels, so they stay local (see the data rule in README).
models.json uses the legacy list schema, distinct from the active showcase schema.
The preserved aggregate snapshot is docs/legacy-models.json. Use export_results.py
to regenerate the active showcase data; do not point --out at data/.

    python scripts/export_models.py                       # Ishmam's models (the default registry)
    python scripts/export_models.py --competition-repo ../UAP-Datathon --only V0.0 V3.11i
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.text import script_bucket  # noqa: E402

LABELS = [0, 1, 2]
CLASS_NAMES = ["explicit", "subtle", "neutral"]
SCRIPTS = ["bangla", "latin", "mixed"]

# Ishmam's models. Farhan adds the LLM members (family "llm") in his own registry or with --registry.
REGISTRY = [
    dict(id="V0.0", kernel="V0.0_tfidf-logreg", name="TF-IDF + logistic regression", family="tfidf", owner="Ishmam"),
    dict(id="V1.2avg", kernel="V1.2avg_banglabert-3seed", name="BanglaBERT (3-seed avg)", family="encoder", owner="Ishmam"),
    dict(id="V1.3avg", kernel="V1.3avg_muril-2seed", name="MuRIL (2-seed avg)", family="encoder", owner="Ishmam"),
    dict(id="V1.7", kernel="V1.7_banglabert-large", name="BanglaBERT-large", family="encoder", owner="Ishmam"),
    dict(id="V1.9", kernel="V1.9_indicbertv2", name="IndicBERT v2", family="encoder", owner="Ishmam"),
    dict(id="V2.2avg", kernel="V2.2avg_xlmr-large-2seed", name="XLM-R-large (2-seed avg)", family="encoder", owner="Ishmam"),
    dict(id="V3.11i", kernel="V3.11i_ensemble", name="Text-only ensemble (final)", family="ensemble", owner="Ishmam",
         public_lb=0.61727, private_lb=0.63275),
]


def load_train(repo: Path) -> pd.DataFrame:
    path = repo / "Competition Info & Starter Files" / "HerWILL-Safe-Social-Media-Datathon-2026" / "train.csv"
    sys.path.insert(0, str(repo / "Scripts"))
    import common  # the competition's own fold code, so rows and folds match every .npy

    tr = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    tr["fold"] = common.make_folds(tr)
    assert common.folds_md5(tr.fold.values) == common.FOLDS_MD5, "folds differ from folds_v1"
    tr["script"] = tr["text"].astype(str).map(script_bucket)
    return tr


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    f1 = f1_score(y, pred, labels=LABELS, average=None, zero_division=0)
    return {"n": int(len(y)),
            "macro_f1": round(float(f1.mean()), 4),
            "f1": {c: round(float(v), 4) for c, v in zip(CLASS_NAMES, f1)},
            "confusion": confusion_matrix(y, pred, labels=LABELS).tolist()}


def entry(spec: dict, probs: np.ndarray, tr: pd.DataFrame, kernel_dir: Path) -> dict:
    ok = ~np.isnan(probs).any(1)  # Bangla-only models leave Latin rows empty
    y, pred, script = tr.y.values[ok], probs[ok].argmax(1), tr.script.values[ok]
    out = {k: v for k, v in spec.items() if k != "kernel"}
    out["coverage"] = round(float(ok.mean()), 4)
    out["oof"] = scores(y, pred)
    out["by_script"] = {s: scores(y[script == s], pred[script == s]) for s in SCRIPTS if (script == s).sum() >= 50}
    metrics = kernel_dir / "preds" / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        if "members" in m and isinstance(m["members"], list):
            out["members"] = m["members"]
        if "cross_fitted" in m:
            out["cross_fitted_macro_f1"] = m["cross_fitted"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition-repo", type=Path, default=ROOT.parent / "UAP-Datathon")
    ap.add_argument("--registry", type=Path, help="JSON list of specs to use instead of the built-in one")
    ap.add_argument("--only", nargs="*", help="export just these ids")
    ap.add_argument("--out", type=Path, default=ROOT / "private" / "legacy-exports")
    args = ap.parse_args()

    registry = json.loads(args.registry.read_text()) if args.registry else REGISTRY
    if args.only:
        registry = [s for s in registry if s["id"] in args.only]
    if args.out.resolve() == (ROOT / "data").resolve():
        ap.error("--out data/ would overwrite the showcase schema; use a separate directory")
    args.out.mkdir(parents=True, exist_ok=True)
    tr = load_train(args.competition_repo)

    rows = {"columns": ["script", "label", "fold"],
            "rows": [[s, int(y), int(f)] for s, y, f in zip(tr.script, tr.y, tr.fold)]}
    (args.out / "oof_rows.json").write_text(json.dumps(rows, separators=(",", ":")))

    models_path = args.out / "models.json"
    models = {m["id"]: m for m in json.loads(models_path.read_text())} if models_path.exists() else {}
    for spec in registry:
        kdir = args.competition_repo / "Kernels" / spec["kernel"]
        probs = np.load(kdir / "preds" / "oof_probs.npy").astype(np.float64)
        assert probs.shape == (len(tr), 3), f"{spec['id']}: {probs.shape}"
        models[spec["id"]] = e = entry(spec, probs, tr, kdir)
        rounded = [None if np.isnan(p).any() else [round(float(x), 4) for x in p] for p in probs]
        (args.out / f"oof_{spec['id']}.json").write_text(
            json.dumps({"id": spec["id"], "probs": rounded}, separators=(",", ":")))
        print(f"{spec['id']:8s} coverage {e['coverage']:.2f}  OOF macro F1 {e['oof']['macro_f1']:.4f}  "
              + "  ".join(f"{s} {v['macro_f1']:.3f}" for s, v in e["by_script"].items()))
    models_path.write_text(json.dumps(list(models.values()), indent=1, ensure_ascii=False))
    print(f"wrote {models_path} ({len(models)} models)")


if __name__ == "__main__":
    main()
