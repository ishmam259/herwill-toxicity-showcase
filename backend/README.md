# Back end (FastAPI)

Serves `POST /api/predict` for the Live Demo. Owner: Ishmam.

```bash
cd backend
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt     # Windows; use .venv/bin/pip on Linux/macOS
.venv/Scripts/python train_tfidf.py                    # ~1 min; needs train.csv from the competition repo (path: --train)
.venv/Scripts/python -m uvicorn app.main:app --port 8000
.venv/Scripts/python -m pytest                         # uses a toy model, no competition data needed
```

- `model/tfidf.joblib` is derived from the competition data, so it is gitignored; everyone builds it locally.
- **Transformer (Obidit):** install `requirements-transformer.txt` and set `MODEL_DIR` to the Hugging Face folder
  (optional `MODEL_NAME` for the display name). It then becomes the primary model and TF-IDF stays as the second vote.
- Word weights come from occlusion (drop a word, re-score): positive = the word pushed the post toward the predicted class.
- `CORS_ORIGINS` (comma-separated) defaults to `http://localhost:5173`.

## `POST /api/predict`

```json
{"text": "Not bad for a girl, I guess."}
```
returns
```json
{"text": "Not bad for a girl, I guess.", "script": "latin", "primary": "tfidf", "label": 1, "label_name": "subtle",
 "models": [{"id": "tfidf", "name": "TF-IDF + logistic regression", "label": 1, "probs": [0.27, 0.38, 0.35]}],
 "tokens": [{"text": "Not", "start": 0, "end": 3, "weight": 0.05}, "..."]}
```
`text` is the normalized post; token offsets refer to it. Labels: 0 explicit, 1 subtle, 2 neutral.
