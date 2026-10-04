"""The models the demo serves. Each predictor maps a batch of normalized texts to (n, 3) probabilities.

- TfidfPredictor: the V0.0 recipe (char_wb 2-5 + word 1-2 TF-IDF, balanced logistic regression),
  refit on the full train set by `train_tfidf.py`. Always available.
- TransformerPredictor: Obidit's fine-tuned encoder (Hugging Face folder). Loaded only when
  MODEL_DIR points at it and torch + transformers are installed.
"""
import os
from pathlib import Path

import joblib
import numpy as np


class TfidfPredictor:
    id = "tfidf"
    name = "TF-IDF + logistic regression"

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.vectorizers = bundle["vectorizers"]
        self.clf = bundle["clf"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        from scipy.sparse import hstack

        X = hstack([v.transform(texts) for v in self.vectorizers]).tocsr()
        return self.clf.predict_proba(X)


class TransformerPredictor:
    id = "transformer"

    def __init__(self, model_dir: Path, max_len: int = 192, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.name = os.environ.get("MODEL_NAME", Path(model_dir).name)
        self.max_len, self.batch_size = max_len, batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.model(**enc).logits, -1).numpy())
        return np.concatenate(out)


def load_predictors(model_root: Path) -> list:
    """TF-IDF first; the transformer is appended (and becomes the primary model) when available."""
    preds = [TfidfPredictor(model_root / "tfidf.joblib")]
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir and Path(model_dir).exists():
        preds.append(TransformerPredictor(Path(model_dir)))
    return preds
