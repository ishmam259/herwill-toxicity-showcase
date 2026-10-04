"""Word importance by occlusion: drop one word, re-score, and see how much the predicted class loses.

Model-agnostic, so the same code explains TF-IDF and the transformer. One extra forward pass per word,
batched; MAX_WORDS caps the cost on long posts.
"""
import numpy as np

from .text import word_spans

MAX_WORDS = 64


def occlusion(predictor, text: str, label: int) -> list[dict]:
    spans = word_spans(text)[:MAX_WORDS]
    if len(spans) < 2:  # one word: removing it leaves nothing to score
        return [{"text": text[s:e], "start": s, "end": e, "weight": 0.0} for s, e in spans]
    variants = [text]
    variants += [(text[:s] + text[e:]).strip() for s, e in spans]
    probs = predictor.predict_proba(variants)
    base = probs[0, label]
    # positive weight = this word pushed the post toward the predicted class
    weights = base - probs[1:, label]
    return [{"text": text[s:e], "start": s, "end": e, "weight": round(float(w), 4)}
            for (s, e), w in zip(spans, np.asarray(weights))]
