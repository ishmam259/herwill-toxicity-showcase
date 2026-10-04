---
title: HerWILL Toxicity Lab
emoji: 🔬
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
---

# HerWILL Toxicity Lab

An interactive research showcase for the event_horizon team's Bangla, Banglish and English toxicity models from the HerWILL × UAP Safe Social Media Datathon 2026. Classes: **0 Explicit · 1 Subtle · 2 Neutral**.

The app includes a live classifier with script detection, probabilities, token influence and served-model votes; historical model comparisons; script-filtered confusion matrices; reliability curves and interactive class offsets; and an explicitly gated local Hard cases collection. It works on desktop and mobile, with keyboard controls, accessible chart tables and tested contrast.

**Data boundary:** competition posts, row-level OOF exports, fitted weights and private cases are never committed or copied into the public Docker image. Public results are aggregate statistics; all shipped example posts were written for this app. User-entered text is processed in memory and is not saved. Organizer permission for public use of competition-trained models or dataset posts remains unresolved.

## Quick start

Requirements: Node 22.12+ and Python 3.13+. Run from this repository's root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
npm ci
npm run build
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860 --no-proxy-headers
```

Open **http://127.0.0.1:7860**. FastAPI serves the production UI and API together. `/api/health` lists the actual served models; `/docs` exposes the API schema.

The default live model is a real TF-IDF/logistic-regression classifier fitted on **18 authored demonstration examples**, clearly marked illustrative. It has no validated generalization score. The research pages show **real historical artifacts**, rather than this fallback model's scores. LLM and encoder historical predictions are not presented as live inference.

For frontend development, run the API on port 8000 in one terminal and Vite in another:

```bash
# Terminal 1, with .venv activated
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
# Terminal 2
npm run dev
```

Vite at http://127.0.0.1:5173 proxies `/api` to port 8000. Results pages bundle public aggregates and work without the API; the live demo explains connection errors.

## Local trained TF-IDF

Use only trusted local competition data. The fitted artifact is ignored by Git and excluded from Docker.

```bash
source .venv/bin/activate
python scripts/train_sparse.py --train /absolute/path/to/train.csv
SPARSE_MODEL_PATH=model/sparse.joblib python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860 --no-proxy-headers
```

In this workspace, the fit is already saved at `model/sparse.joblib` and covers all 47,817 training rows. A fresh clone must run the training command to obtain it. This full-data fit is distinct from the historical five-fold V0.0 model; do not assign the historical score to its live predictions. Joblib loads are for trusted operator-selected artifacts only; there is no artifact-upload endpoint.

## Research exports

Committed `data/models.json` and `data/calibration.json` were computed from the verified local prediction artifacts. Seven entries cover TF-IDF, four encoders, Qwen LoRA and the reconstructed text-only V3.7 ensemble. All comparison and matrix pages use **raw probability argmax**, including the ensemble, for a consistent decision rule.

```bash
python scripts/export_results.py --workspace /absolute/path/to/datathon-workspace
python scripts/check_public_data.py
```

The workspace must contain `dataset/train.csv`, `kaggle/v6-replay-foundation/`, and `repository/UAP-Datathon/Kernels/V3.7_ensemble/preds/oof_probs.npy`. The exporter verifies official data and member SHA-256 hashes, ascending ID alignment, [0,1,2] class order, specialist coverage and ensemble parity. Exported public files contain no row-level probabilities, IDs or posts.

Scores are **historical, adaptively inspected OOF diagnostics**, not fresh held-out performance. Bangla specialists cover only posts containing Bangla characters, including mixed script. The non-Bangla group includes English, Banglish and other scripts; script proxies are not language annotations. Choose the Bangla-containing slice to compare all seven models on the same rows.

Calibration uses 10 equal-width bins of top-class confidence. ECE is support-weighted absolute confidence/accuracy error; the multiclass Brier score sums squared class-probability error. The two offset sliders use a 21×21 aggregate grid computed from `argmax(log(max(p,1e-8)) + [0, subtle, neutral])`. Every setting is an exact historical recomputation; no OOF rows are downloaded by the browser. Changing offsets does not modify live inference. Tuning on the same labels can inflate scores.

## Private Hard cases

Local setup only. Bind both API and Vite to **127.0.0.1**, keep proxy-header trust disabled, and do not place private mode behind a public reverse proxy or port forward.

```bash
python scripts/export_results.py --workspace /absolute/path/to/datathon-workspace --private
ENABLE_PRIVATE_CASES=1 PUBLIC_DEPLOYMENT=0 SPARSE_MODEL_PATH=model/sparse.joblib \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860 --no-proxy-headers
```

`private/hard_cases.json` contains up to 40 posts per collection and stays ignored. “All wrong” requires every covered member and the ensemble to be wrong. “LLM only” requires Qwen to be correct and all covered non-LLM members to be wrong; the ensemble is displayed but is not a member in that selection. Missing specialist votes appear as N/A. The endpoint checks actual socket loopback, the explicit flag and public-deployment mode. Spoofed `X-Forwarded-For` cannot grant access. Never share private screenshots or videos.

## Optional transformer checkpoint

No saved deployable transformer was present, and no suitable GPU was available during this build. The app has an offline local checkpoint adapter and reproducible training/export commands; **neural training and the int8 quality gate have not been executed**. A configured transformer adds a second real live vote. TF-IDF provides the token deletion explanation; it is not a transformer attribution.

```bash
pip install -r backend/requirements-transformer.txt
# A diagnostic fit with a duplicate-grouped 20% holdout:
python scripts/train_transformer.py --train /absolute/path/to/train.csv \
  --base /absolute/path/to/local/pretrained-model --output model/transformer
# Validate int8 quality, CPU latency and reload parity on that unseen split:
python scripts/export_cpu.py --model model/transformer \
  --validation model/transformer/heldout.csv --output model/transformer-int8
TRANSFORMER_MODEL_PATH=model/transformer-int8 SPARSE_MODEL_PATH=model/sparse.joblib \
  python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 7860 --no-proxy-headers
```

`--download-base` allows fetching a base model when explicitly selected; the default requires local files. `--full-fit` trains all supplied rows and reports no held-out score. The recipe uses three-class cross-entropy at 256 tokens, LR 2e-5 and duplicate-grouped validation; it is not an exact historical V1.x recipe replay. Artifacts declare class order in `showcase.json`. Int8 export rejects overlapping validation text and a macro-F1 drop ≥0.005, measures latency, and checks every validation decision after reload. A full-fit model requires separately reserved unseen validation data for this quality check. Public deployment of these weights awaits organizer permission.

## Tests and demo video

```bash
source .venv/bin/activate
npm run check:data
npm run test:api
npm run build
npx playwright install chromium
npm run test:e2e
npm run record:demo
```

Backend tests cover validation, multilingual predictions, explanation consistency, concurrency, public allowlists and private-access denial. Playwright checks all pages on desktop and mobile, examples, keyboard prediction, filters, thresholds, errors, stale-request cancellation, horizontal overflow and WCAG A/AA rules through axe. The walkthrough records only authored/public content; find the `.webm` in `test-results/` after `record:demo`. GitHub Actions runs the data check, backend and browser suites, build and Docker build. Optional transformer scripts require their own GPU and held-out validation before release.

## Deployment

```bash
docker compose up --build
```

Open http://127.0.0.1:7860. The image runs as a non-root user, contains aggregate JSON and the UI, serves the authored fallback, and enforces `PUBLIC_DEPLOYMENT=1`. No private data or competition-trained weights are included. `PORT` defaults to 7860. For Hugging Face Spaces, select Docker and retain the README metadata. `render.yaml` provides a Render Docker service. No external deployment or publication has been performed.

Architecture and frozen contracts: [docs/SPEC.md](docs/SPEC.md). Build status and remaining external work: [TASKS.md](TASKS.md). The implementation follows the official [Vite guide](https://vite.dev/guide/), [FastAPI container guidance](https://fastapi.tiangolo.com/deployment/docker/), [Transformers training APIs](https://huggingface.co/docs/transformers/training), and [PyTorch dynamic quantization API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html).
