"""Optional saved-checkpoint training; requires a suitable GPU and local data.

Defaults to a duplicate-grouped held-out split for an honest diagnostic.
--full-fit instead trains all rows; no CV score is claimed for that artifact.
This is a deployable training recipe, not a recreation of historical V1.x FGM,
layerwise LR decay, or seed averaging. Those historical scores must not be
assigned to this new model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.inference import normalize


def text_key(text):
    return hashlib.sha256(normalize(text).lower().encode()).hexdigest()


def train(args):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No GPU found. Use a GPU machine or explicitly pass --allow-cpu for a slow small-model test."
        )
    set_seed(args.seed)
    tr = pd.read_csv(args.train)
    if args.limit:
        tr = tr.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
    if (
        not {"text", "y"}.issubset(tr.columns)
        or set(tr.y.unique()) != {0, 1, 2}
        or tr.text.isna().any()
    ):
        raise ValueError("Expected nonmissing text and labels 0,1,2")
    texts = tr.text.map(normalize).tolist()
    keys = np.array([text_key(t) for t in texts])
    if args.full_fit:
        fit = np.arange(len(tr))
        val = np.array([], dtype=int)
    else:
        fit, val = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(
                tr, groups=keys
            )
        )
        if set(tr.y.iloc[fit]) != {0, 1, 2} or set(tr.y.iloc[val]) != {0, 1, 2}:
            raise ValueError("Held-out split must cover every class")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base, local_files_only=not args.download_base
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=3,
        id2label={0: "Explicit", 1: "Subtle", 2: "Neutral"},
        label2id={"Explicit": 0, "Subtle": 1, "Neutral": 2},
        local_files_only=not args.download_base,
    )

    class Posts(torch.utils.data.Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            i = self.indices[index]
            out = tokenizer(texts[i], truncation=True, max_length=256)
            out["labels"] = int(tr.y.iloc[i])
            return out

    args.output.mkdir(parents=True, exist_ok=True)
    steps_total = int(np.ceil(len(fit) / (args.batch_size * args.accumulation)) * args.epochs)
    training = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        # warmup_steps (not warmup_ratio) works on transformers 4.x and 5.x.
        warmup_steps=int(args.warmup * steps_total),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if len(val) else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=bool(len(val)),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )
    class CollapseGuard(TrainerCallback):
        # A collapsed run predicts the class prior (macro F1 ~0.22) and never recovers; stop early.
        def on_evaluate(self, targs, state, control, metrics=None, **kwargs):
            f1 = (metrics or {}).get("eval_macro_f1")
            if args.abort_below and f1 is not None and f1 < args.abort_below:
                # Marker tells the kernel not to retry a collapse (only a crash is worth retrying).
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "COLLAPSED").write_text(f"{f1}\n")
                raise RuntimeError(f"Held-out macro F1 {f1:.4f} < {args.abort_below} at epoch {state.epoch:.2f}: collapsed")

    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=Posts(fit),
        eval_dataset=Posts(val) if len(val) else None,
        processing_class=tokenizer,
        callbacks=[CollapseGuard()],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda ev: {
            "macro_f1": float(
                f1_score(
                    ev.label_ids,
                    ev.predictions.argmax(1),
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            )
        },
    )
    trainer.train()
    trainer.save_model(str(args.output))  # rank-0 only under torchrun
    metrics = trainer.evaluate() if len(val) else None  # collective: every rank must call it
    if not trainer.is_world_process_zero():
        return
    tokenizer.save_pretrained(args.output)
    metadata = {
        "id": "deployable-transformer",
        "name": f"{Path(args.base).name} · local fit",
        "class_order": [0, 1, 2],
        "mode": "local-trained",
        "base": args.base,
        "full_fit": args.full_fit,
        "training_rows": len(fit),
        "validation_rows": len(val),
        "evaluation": metrics,
        "recipe": f"three-class cross-entropy, 256 tokens, LR {args.lr}, warmup {args.warmup}, duplicate-grouped holdout or full fit",
        "torch_version": torch.__version__,
    }
    (args.output / "showcase.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "training_keys.sha256").write_text(
        "\n".join(sorted(set(keys[fit]))) + "\n"
    )
    if len(val):
        tr.iloc[val][["text", "y"]].to_csv(args.output / "heldout.csv", index=False)
    print(
        "Saved local checkpoint. "
        + (
            "No held-out score exists for this full fit."
            if args.full_fit
            else f"Held-out diagnostics: {metrics}"
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument(
        "--base",
        required=True,
        help="Trusted local model directory, or model ID with --download-base",
    )
    p.add_argument("--output", type=Path, default=ROOT / "model/transformer")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1, help="warmup as a fraction of total steps")
    p.add_argument("--abort-below", type=float, default=0.0, help="stop if any held-out eval macro F1 is below this")
    p.add_argument("--limit", type=int, default=0, help="use only the first N rows (pipeline smoke tests)")
    p.add_argument("--full-fit", action="store_true")
    p.add_argument("--download-base", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    train(p.parse_args())
