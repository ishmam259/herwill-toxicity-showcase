"""Word importance by occlusion: drop one word, re-score, and see how much the predicted class loses.

Model-agnostic, so the same code explains TF-IDF and the transformer. One extra forward pass per word,
batched; MAX_WORDS caps the cost on long posts.
"""
import numpy as np

from .text import normalize, word_spans

MAX_WORDS = 80
EXPLANATION = ("Word deletion influence: how much the predicted class's probability drops when the word is "
               "removed. Positive values support the prediction. This describes model behaviour, not a cause.")


def occlusion(predictor, text: str, label: int) -> tuple[list[dict], bool]:
    """Returns (tokens, truncated). Token offsets refer to `text`, which must already be normalized."""
    all_spans = word_spans(text)
    spans = all_spans[:MAX_WORDS]
    truncated = len(all_spans) > MAX_WORDS
    if len(spans) < 2:  # one word: removing it leaves nothing to score
        return [{"text": text[s:e], "start": s, "end": e, "weight": 0.0} for s, e in spans], truncated
    variants = [text] + [normalize(text[:s] + text[e:]) for s, e in spans]
    probs = predictor.predict_proba(variants)
    # positive weight = this word pushed the post toward the predicted class
    weights = probs[0, label] - probs[1:, label]
    return [{"text": text[s:e], "start": s, "end": e, "weight": round(float(w), 4)}
            for (s, e), w in zip(spans, np.asarray(weights))], truncated
