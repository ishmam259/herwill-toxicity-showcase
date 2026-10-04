# Back end (FastAPI)

One server, `backend/app/` (owner: Ishmam). It serves the classifier, the public result files and the
compiled front end. Run it from the repository root:

```bash
pip install -r backend/requirements-dev.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
python -m pytest tests backend/tests -q        # no competition data needed
```

| File | Role |
|---|---|
| `app/main.py` | Routes, validation, privacy headers, the busy lock, static front end |
| `app/predictors.py` | TF-IDF and transformer loaders |
| `app/explain.py` | Word influence by deletion (occlusion), for whichever model is primary |
| `app/text.py` | Normalization and the script badge (same rules as the competition code) |
| `inference.py` | Training helpers only (`fit_sparse`), used by `scripts/` |
| `train_tfidf.py` | Fits the V0.0 TF-IDF on the full train set into `model/tfidf.joblib` |

## Which models load

1. **TF-IDF**, always the first vote, from the first of:
   - `SPARSE_MODEL_PATH` (e.g. `model/sparse.joblib` from `scripts/train_sparse.py`);
   - `backend/model/tfidf.joblib` from `train_tfidf.py`, except when `PUBLIC_DEPLOYMENT=1`;
   - a small illustrative model fitted at start-up on the 18 hand-written posts in `data/examples.json`.
     A fresh checkout and the public Docker image use this. `SPARSE_MODEL_PATH=""` forces it.
2. **Transformer**, when `TRANSFORMER_MODEL_PATH` (or `MODEL_DIR`) points at a Hugging Face folder or the int8
   export from `scripts/export_cpu.py`. It becomes the primary model (headline verdict and word weights) and
   TF-IDF stays as the second vote. Needs `requirements-transformer.txt`.

Model files are derived from the competition data, so they are gitignored and kept out of the Docker image.

## Routes

| Route | |
|---|---|
| `POST /api/predict` | Classify one post (below) |
| `GET /api/health` | Loaded models, `transformer_available`, `private_cases`, `stores_user_text: false` |
| `GET /api/data/{models,calibration,examples}` | Public aggregate files from `data/` |
| `GET /api/hard-cases` | `private/hard_cases.json`, only with `ENABLE_PRIVATE_CASES=1`, not `PUBLIC_DEPLOYMENT=1`, from a loopback client |
| `GET /`, `/compare`, `/confusion`, `/calibration`, `/hard-cases` | The compiled UI from `frontend/dist` |

## `POST /api/predict`

Request: `{"text": "..."}`, 1–2000 characters, not blank, no other fields. A second request while one is
running gets 429.

```json
{"text": "Not bad for a girl, I guess.", "script": "latin", "script_name": "Latin script",
 "primary": "tfidf", "label": 1, "label_name": "subtle",
 "models": [{"id": "tfidf", "name": "TF-IDF + logistic regression", "label": 1, "probs": [0.27, 0.38, 0.35],
             "mode": "local-trained", "has_features": true}],
 "tokens": [{"text": "Not", "start": 0, "end": 3, "weight": 0.05}, "..."],
 "explanation": "Word deletion influence: ...", "explanation_truncated": false,
 "latency_ms": 9.6, "warning": null}
```

- `text` is the normalized post; token offsets refer to it. Labels: 0 explicit, 1 subtle, 2 neutral.
- `primary` names the model behind `label` and `tokens`; `models` holds every vote, TF-IDF first.
- Token `weight` = drop in the primary model's probability for `label` when that word is removed.
  Only the first 80 words are tested (`explanation_truncated`).
- `warning` is set when the primary model is the illustrative one.

## Other settings

- `CORS_ORIGINS` (comma-separated) defaults to `http://localhost:5173`.
- `scripts/export_models.py` writes the list-schema metrics and per-row files to ignored `private/legacy-exports/`;
  the public aggregates in `data/` come from `scripts/export_results.py`.
