"""Fit the demo's TF-IDF model on the full train set and save it to model/tfidf.joblib.

Same recipe as UAP-Datathon/Kernels/V0.0_tfidf-logreg (5-fold OOF macro F1 0.5225), refit on all rows.
The output is derived from the competition data, so it is gitignored: every teammate builds it locally.

    python train_tfidf.py --train "../../UAP-Datathon/Competition Info & Starter Files/HerWILL-Safe-Social-Media-Datathon-2026/train.csv"
"""
import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.text import normalize

DEFAULT_TRAIN = (Path(__file__).resolve().parents[2] / "UAP-Datathon" / "Competition Info & Starter Files"
                 / "HerWILL-Safe-Social-Media-Datathon-2026" / "train.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "model" / "tfidf.joblib")
    args = ap.parse_args()

    t0 = time.time()
    tr = pd.read_csv(args.train)
    text, y = tr["text"].map(normalize), tr["y"].values
    vectorizers = [
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                        max_features=300_000, lowercase=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        max_features=200_000, lowercase=True, token_pattern=r"(?u)\b\w+\b"),
    ]
    X = hstack([v.fit_transform(text) for v in vectorizers]).tocsr()
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000).fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizers": vectorizers, "clf": clf, "recipe": "V0.0", "n_train": len(tr)},
                args.out, compress=3)
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
