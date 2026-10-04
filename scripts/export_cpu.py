"""Export dynamic int8 weights after held-out quality and reload verification.

Fails if the macro-F1 drop is >=0.005, or any validation text was used in
training. Writes measured single-post latency (including tokenization).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_transformer import text_key


def export(args):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    meta = json.loads((args.model / "showcase.json").read_text())
    if meta.get("class_order") != [0, 1, 2]:
        raise ValueError("Class order must be [0,1,2]")
    known = set((args.model / "training_keys.sha256").read_text().splitlines())
    val = pd.read_csv(args.validation)
    if val.text.isna().any() or set(val.y.unique()) != {0, 1, 2}:
        raise ValueError("Validation must cover all classes")
    if any(text_key(t) in known for t in val.text):
        raise ValueError(
            "Validation text overlaps training; parity score would be invalid"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    original = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model, local_files_only=True
        )
        .cpu()
        .eval()
    )
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(original), {torch.nn.Linear}, dtype=torch.qint8
    )

    texts = val.text.tolist()

    def run(model, batch_size=32):
        # Same batching, padding and order as TransformerPredictor.predict_proba, so the reload
        # check below compares like with like (dynamic int8 depends on the padded batch).
        predictions = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=256,
                                padding=True, return_tensors="pt")
                predictions.extend(model(**enc).logits.argmax(-1).tolist())
        return predictions

    def latency_ms(model, n=200):
        # Sequential single-post latency, as the Live Demo serves it.
        out = []
        with torch.inference_mode():
            for text in texts[:n]:
                start = time.perf_counter()
                model(**tokenizer(text, return_tensors="pt", truncation=True, max_length=256))
                out.append((time.perf_counter() - start) * 1000)
        return out

    a = run(original)
    b = run(quantized)
    latency = latency_ms(quantized)
    macro = lambda pred: float(
        f1_score(val.y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
    )
    fa, fb = macro(a), macro(b)
    if fa - fb >= 0.005:
        raise ValueError(f"Int8 macro F1 drop {fa - fb:.6f} exceeds the quality budget")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    original.config.save_pretrained(args.output)
    torch.save(quantized.state_dict(), args.output / "int8_state.pt")
    metadata = {
        **meta,
        "inference_backend": "dynamic-int8",
        "torch_version": torch.__version__,
        "quantization_check": {
            "validation_rows": len(val),
            "fp32_macro_f1": fa,
            "int8_macro_f1": fb,
            "drop": fa - fb,
            "latency_p50_ms": float(np.percentile(latency, 50)),
            "latency_p95_ms": float(np.percentile(latency, 95)),
            "latency_note": "CPU, sequential single-post inference including tokenization, first 200 validation posts; first-run effects included",
        },
    }
    (args.output / "showcase.json").write_text(json.dumps(metadata, indent=2) + "\n")
    from backend.app.predictors import TransformerPredictor

    loaded = TransformerPredictor(args.output)
    # Compare every validation decision after reload, not just one sample.
    reload_pred = loaded.predict_proba(texts).argmax(1).tolist()
    if reload_pred != b:
        raise ValueError("Reloaded int8 decisions differ")
    print(json.dumps(metadata["quantization_check"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "model/transformer-int8")
    export(p.parse_args())
