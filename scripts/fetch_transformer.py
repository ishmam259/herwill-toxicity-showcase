"""Fetch the trained int8 transformer from the Kaggle kernel and install it for the Live Demo.

Run from the repo root once the kernel (kaggle/deployable-transformer/) has finished:

    set -a; source <path>/nasin.env; set +a      # token of the account that ran V6, never committed
    python scripts/fetch_transformer.py

Steps: download the kernel output to a temporary folder, check the quality gates, copy ONLY the
int8 folder to model/transformer-int8 (gitignored), delete everything else (the fp32 checkpoint and
any competition text the kernel left behind), then load it through the real predictor and print the
command that serves it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KERNEL = "nasreenakhter229/herwill-showcase-xlmr-large-export"  # V6 stage 2 output
REQUIRED = ["config.json", "int8_state.pt", "showcase.json", "tokenizer.json", "tokenizer_config.json"]


def check(meta, min_f1):
    """Refuse a collapsed or degraded model before it reaches the demo."""
    evaluation = meta.get("evaluation") or {}
    f1 = evaluation.get("eval_macro_f1")
    if f1 is None:
        raise SystemExit("showcase.json has no held-out macro F1; refusing an unvalidated model")
    # Always predicting one class scores ~0.22 here; a collapsed run lands near that.
    if f1 < min_f1:
        raise SystemExit(f"Held-out macro F1 {f1:.4f} < {min_f1}: the run likely collapsed; not installing")
    q = meta.get("quantization_check") or {}
    if q.get("drop") is None or q["drop"] >= 0.005:
        raise SystemExit(f"Int8 quality gate missing or failed: {q}")
    return f1, q


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", default=KERNEL)
    p.add_argument("--source", type=Path, help="use an already-downloaded output folder instead of Kaggle")
    p.add_argument("--dest", type=Path, default=ROOT / "model/transformer-int8")
    p.add_argument("--min-f1", type=float, default=0.45)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.source:
            out = args.source
        else:
            out = Path(tmp)
            print(f"Downloading {args.kernel} output ...", flush=True)
            subprocess.run(["kaggle", "kernels", "output", args.kernel, "-p", str(out)], check=True)
        src = out / "transformer-int8"
        missing = [f for f in REQUIRED if not (src / f).is_file()]
        if missing:
            raise SystemExit(f"Kernel output has no complete transformer-int8 folder (missing {missing}); check the kernel log")
        meta = json.loads((src / "showcase.json").read_text(encoding="utf-8"))
        f1, q = check(meta, args.min_f1)

        if args.dest.exists():
            shutil.rmtree(args.dest)
        args.dest.mkdir(parents=True)
        for f in REQUIRED:
            shutil.copy2(src / f, args.dest / f)
        # Leaving the with-block deletes the download, including the fp32 weights and any heldout.csv.

    from backend.app.predictors import TransformerPredictor

    model = TransformerPredictor(args.dest)
    samples = ["Thanks for the thoughtful explanation.", "আজকের আলোচনা ভালো লেগেছে।", "Ajker discussion bhalo legeche."]
    probs = model.predict_proba(samples)
    size_mb = sum(f.stat().st_size for f in args.dest.iterdir()) / 2**20
    print(f"Installed {args.dest} ({size_mb:.0f} MB): held-out macro F1 {f1:.4f}, "
          f"int8 drop {q['drop']:.4f}, CPU latency p50 {q.get('latency_p50_ms', 0):.0f} ms")
    for text, row in zip(samples, probs):
        print(f"  {row.round(3).tolist()}  {text}")
    print("\nServe it:\n  TRANSFORMER_MODEL_PATH=model/transformer-int8 "
          "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860 --no-proxy-headers")


if __name__ == "__main__":
    main()
