"""CPU inference with an authored-example fallback and optional local weights."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Explicit", "Subtle", "Neutral"]


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def script_info(text):
    bangla = bool(re.search("[\u0980-\u09ff]", text))
    latin = bool(re.search("[a-zA-Z]", text))
    return {
        "id": "mixed"
        if bangla and latin
        else "bangla"
        if bangla
        else "latin"
        if latin
        else "other",
        "name": "Mixed script"
        if bangla and latin
        else "Bangla script"
        if bangla
        else "Latin script"
        if latin
        else "Other script",
        "note": "Script detection does not distinguish English from Banglish.",
    }


def fit_sparse(texts, labels, full=False):
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3 if full else 1,
        max_features=300_000 if full else 20_000,
        sublinear_tf=True,
    )
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2 if full else 1,
        max_features=200_000 if full else 10_000,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
    )
    x = hstack([char.fit_transform(texts), word.fit_transform(texts)]).tocsr()
    model = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
    model.fit(x, labels)
    return {"char": char, "word": word, "model": model}


class SparsePredictor:
    def __init__(self, path=None):
        if path:
            import joblib

            # Explicit operator-selected, trusted artifact only; never user upload.
            bundle = joblib.load(path)
            self.metadata = bundle["metadata"]
            if self.metadata.get("class_order") != [0, 1, 2]:
                raise ValueError("Sparse artifact class order must be [0,1,2]")
            self.bundle = bundle
        else:
            examples = json.loads((ROOT / "data/examples.json").read_text())["examples"]
            self.bundle = fit_sparse(
                [normalize(e["text"]) for e in examples], [e["label"] for e in examples]
            )
            self.metadata = {
                "id": "authored-tfidf",
                "name": "TF-IDF · authored examples",
                "source": "18 authored demonstration examples",
                "class_order": [0, 1, 2],
                "mode": "illustrative",
                "warning": "This small demo model illustrates the interface. It is not a competition checkpoint and has no validated generalization score.",
            }
        if list(self.bundle["model"].classes_) != [0, 1, 2]:
            raise ValueError("Classifier classes differ from the API contract")

    def features(self, texts):
        return hstack(
            [self.bundle["char"].transform(texts), self.bundle["word"].transform(texts)]
        ).tocsr()

    def probabilities(self, texts):
        return self.bundle["model"].predict_proba(self.features(texts))

    def predict(self, text):
        p = self.probabilities([normalize(text)])[0]
        return {
            "id": self.metadata["id"],
            "name": self.metadata["name"],
            "label": int(p.argmax()),
            "probs": p.tolist(),
            "mode": self.metadata["mode"],
            "has_features": bool(self.features([normalize(text)]).nnz),
        }

    def explain(self, text, label):
        # Deletion influence: change in the selected-class probability when one
        # token is removed. Includes both word and character TF-IDF features.
        spans = list(re.finditer(r"\s+|[^\s]+", text))
        words = [s for s in spans if not s.group().isspace()][:80]
        base = self.probabilities([normalize(text)])[0, label]
        variants = [normalize(text[: s.start()] + text[s.end() :]) for s in words]
        reduced = self.probabilities(variants)[:, label] if variants else []
        weights = {s.start(): float(base - p) for s, p in zip(words, reduced)}
        return [
            {"text": s.group(), "weight": weights.get(s.start(), 0.0)} for s in spans
        ]


class TransformerPredictor:
    def __init__(self, path):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        path = Path(path)
        self.metadata = json.loads((path / "showcase.json").read_text())
        if self.metadata.get("class_order") != [0, 1, 2]:
            raise ValueError(
                "Transformer artifact requires explicit [0,1,2] class order"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if self.metadata.get("inference_backend") == "dynamic-int8":
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_config(config).cpu().eval()
            self.model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.model.load_state_dict(
                torch.load(
                    path / "int8_state.pt", map_location="cpu", weights_only=True
                )
            )
        else:
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    path, local_files_only=True
                )
                .cpu()
                .eval()
            )
        if self.model.config.num_labels != 3:
            raise ValueError("Expected three classifier outputs")
        self.torch = torch

    def predict(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        with self.torch.inference_mode():
            p = self.model(**inputs).logits.softmax(-1)[0].tolist()
        return {
            "id": self.metadata.get("id", "deployable-transformer"),
            "name": self.metadata.get("name", "Local transformer"),
            "label": int(np.argmax(p)),
            "probs": p,
            "mode": "local-trained",
            "has_features": True,
        }


def predict_response(text, sparse, transformer=None):
    started = time.perf_counter()
    vote = sparse.predict(text)
    models = [vote]
    if transformer:
        models.append(transformer.predict(normalize(text)))
    return {
        "script": script_info(text),
        "models": models,
        "tokens": sparse.explain(text, vote["label"]),
        "explanation": "TF-IDF token deletion influence on its selected-class probability; positive values support the prediction. This is model behavior, not a causal explanation.",
        "explanation_model": vote["id"],
        "explanation_truncated": len(text.split()) > 80,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "warning": sparse.metadata.get("warning"),
    }
