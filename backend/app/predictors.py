"""The models the demo serves. Each predictor maps a batch of normalized texts to (n, 3) probabilities.

- TfidfPredictor: the V0.0 recipe (char_wb 2-5 + word 1-2 TF-IDF, balanced logistic regression).
  Loads, in order: SPARSE_MODEL_PATH; else backend/model/tfidf.joblib (built by train_tfidf.py) unless
  PUBLIC_DEPLOYMENT=1; else fits a tiny illustrative model on the authored examples in data/examples.json,
  so a fresh checkout or a public image runs without competition data. SPARSE_MODEL_PATH="" forces that
  fallback. Reads both bundle formats: ours ({vectorizers, clf}) and scripts/train_sparse.py's
  ({char, word, model, metadata}).
- TransformerPredictor: the fine-tuned encoder (Hugging Face folder, optionally the int8 export from
  scripts/export_cpu.py). Loaded only when TRANSFORMER_MODEL_PATH (or MODEL_DIR) is set.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text import normalize

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
DEFAULT_TFIDF = BACKEND / "model" / "tfidf.joblib"

ILLUSTRATIVE_WARNING = ("This small demo model is fitted on 18 hand-written examples to show the interface. "
                        "It is not a competition model and has no validated score.")


class TfidfPredictor:
    def __init__(self, vectorizers: list, clf, metadata: dict):
        if list(clf.classes_) != [0, 1, 2]:
            raise ValueError("Classifier classes differ from the API contract [0, 1, 2]")
        self.vectorizers, self.clf, self.metadata = vectorizers, clf, metadata
        self.id, self.name, self.mode = metadata["id"], metadata["name"], metadata["mode"]

    @classmethod
    def from_file(cls, path: Path) -> "TfidfPredictor":
        bundle = joblib.load(path)  # trusted, operator-selected artifact only; never a user upload
        if "vectorizers" in bundle:  # train_tfidf.py
            meta = {"id": "tfidf", "name": "TF-IDF + logistic regression", "mode": "local-trained",
                    "warning": None}
            return cls(bundle["vectorizers"], bundle["clf"], meta)
        meta = dict(bundle["metadata"])  # scripts/train_sparse.py
        meta.setdefault("mode", "local-trained")
        return cls([bundle["char"], bundle["word"]], bundle["model"], meta)

    @classmethod
    def illustrative(cls) -> "TfidfPredictor":
        examples = json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))["examples"]
        texts, labels = [normalize(e["text"]) for e in examples], [e["label"] for e in examples]
        vectorizers = [
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"),
        ]
        X = hstack([v.fit_transform(texts) for v in vectorizers]).tocsr()
        clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000).fit(X, labels)
        meta = {"id": "authored-tfidf", "name": "TF-IDF (authored examples)", "mode": "illustrative",
                "warning": ILLUSTRATIVE_WARNING}
        return cls(vectorizers, clf, meta)

    def features(self, texts: list[str]):
        return hstack([v.transform(texts) for v in self.vectorizers]).tocsr()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.features(texts))

    def has_features(self, text: str) -> bool:
        return bool(self.features([text]).nnz)


class TransformerPredictor:
    mode = "local-trained"

    def __init__(self, model_dir: Path, max_len: int = 256, batch_size: int = 32):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        meta_path = model_dir / "showcase.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if self.metadata.get("class_order", [0, 1, 2]) != [0, 1, 2]:
            raise ValueError("Transformer artifact must use class order [0, 1, 2]")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.metadata.get("inference_backend") == "dynamic-int8":
            config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_config(config).cpu().eval()
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.load_state_dict(torch.load(model_dir / "int8_state.pt", map_location="cpu", weights_only=True))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).cpu()
        self.model = model.eval()
        if self.model.config.num_labels != 3:
            raise ValueError("Expected three classifier outputs")
        self.id = self.metadata.get("id", "transformer")
        self.name = self.metadata.get("name") or os.environ.get("MODEL_NAME", model_dir.name)
        self.max_len, self.batch_size = max_len, batch_size

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.model(**enc).logits, -1).numpy())
        return np.concatenate(out)

    def has_features(self, text: str) -> bool:
        return True


def load_sparse() -> TfidfPredictor:
    path = os.environ.get("SPARSE_MODEL_PATH")
    if path is None and DEFAULT_TFIDF.is_file() and os.environ.get("PUBLIC_DEPLOYMENT") != "1":
        path = str(DEFAULT_TFIDF)
    return TfidfPredictor.from_file(Path(path)) if path else TfidfPredictor.illustrative()


def load_predictors() -> list:
    """TF-IDF first; the transformer is appended (and becomes the primary model) when configured."""
    preds = [load_sparse()]
    model_dir = os.environ.get("TRANSFORMER_MODEL_PATH") or os.environ.get("MODEL_DIR")
    if model_dir:
        preds.append(TransformerPredictor(Path(model_dir)))
    return preds
