"""Training helpers shared by scripts/ (train_sparse.py, train_transformer.py).

Serving lives in backend/app/ (main.py, predictors.py); this module only fits models.
"""

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.app.text import normalize  # noqa: F401  (re-exported for the scripts)


def fit_sparse(texts, labels, full=False):
    """The V0.0 TF-IDF recipe; returns the bundle format TfidfPredictor.from_file reads."""
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
