"""Fit a trusted local CPU model. Artifacts remain in ignored model/.

No holdout score is invented: the full-fit artifact is distinct from historical
five-fold V0.0 predictions. Never package these weights for public use until the
organizers authorize trained-model reuse.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.inference import fit_sparse, normalize


def train(data, output):
    tr = pd.read_csv(data)
    if (
        not {"text", "y"}.issubset(tr.columns)
        or set(tr.y.unique()) != {0, 1, 2}
        or tr.text.isna().any()
    ):
        raise ValueError("Expected nonmissing text and labels 0,1,2")
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        bundle = fit_sparse(tr.text.map(normalize).tolist(), tr.y.tolist(), full=True)
    samples = [
        "Hello, thanks for the explanation.",
        "তোমার ব্যাখ্যার জন্য ধন্যবাদ।",
        "Tui ekta boka.",
    ]
    bundle["metadata"] = {
        "id": "local-tfidf",
        "name": "TF-IDF · full local fit",
        "class_order": [0, 1, 2],
        "mode": "local-trained",
        "source": "Local competition training data",
        "rows": len(tr),
        "sklearn_version": sklearn.__version__,
        "fit_seconds": round(time.perf_counter() - start, 2),
        "warning": "Local full-data TF-IDF fit; its live predictions are distinct from the historical five-fold model. Public trained-model reuse is pending organizer permission.",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    from backend.inference import SparsePredictor

    reloaded = SparsePredictor(output)
    for text in samples:
        vote = reloaded.predict(text)
        if abs(sum(vote["probs"]) - 1) > 1e-6:
            raise ValueError("Reload verification failed")
    (output.parent / "sparse_metadata.json").write_text(
        json.dumps(bundle["metadata"], indent=2) + "\n"
    )
    print(
        f"Fitted and reload-verified {len(tr):,} rows in {time.perf_counter() - start:.1f}s. Weights: {output}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "model/sparse.joblib")
    a = p.parse_args()
    train(a.train, a.output)
