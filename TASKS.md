# Build status · 4 October 2026

The software scaffold has been implemented. The original team plan is preserved below as historical context; its unchecked items are not a statement that the shipped pages are absent.

| Deliverable | Status |
|---|---|
| React/Vite shell, responsive layout, routes, class tokens | Complete |
| Frozen contracts and product specification | Complete — `docs/SPEC.md` |
| FastAPI classifier, script badges, probabilities, model votes, token highlighting | Complete — **one server, `backend/app/` (Ishmam's)**; Obidit's routes, privacy gate and authored fallback merged into it on 4 Oct; `backend/main.py` removed |
| 18 original examples across Bangla, Banglish, English × three classes | Complete |
| Verified encoder, TF-IDF, Qwen and text-ensemble aggregate exporters | Complete — 7 models; hash, alignment, coverage and ensemble checks |
| Live Demo page | Complete — **Ishmam's `features/live-demo/`** mounted on `/`; Obidit's `src/LiveDemo.tsx` removed on 4 Oct |
| Model comparison and script slices | Complete |
| Confusion matrices, normalization and Explicit/Subtle emphasis | Complete |
| Reliability curves, ECE/Brier and exact interactive offset grid | Complete |
| Private Hard cases exporter, collections and local access gate | Complete — ignored and excluded from public builds |
| Local full-data CPU TF-IDF weights | Complete — 47,817 rows; saved and reload-verified |
| Optional transformer training, saved-checkpoint serving and int8 quality-gated exporter | Implemented; GPU training and held-out quality check not executed |
| Backend, desktop/mobile smoke and accessibility tests | Implemented; see `docs/VERIFICATION.md` for executed results |
| Docker, Compose, HF Spaces metadata, Render config and CI | Complete; local Docker build checked; external CI/deploy not triggered |
| README and public-only walkthrough recording | Complete; see README for commands |
| Public deployment, teammate PR review | Pending; no deployment or review is claimed |

The task to save and shrink a deployable **transformer** remains pending suitable compute and unseen validation data. The new training recipe is explicitly identified and must not inherit a historical CV score. No competition data has been published.

---

# Showcase app: task split (Ishmam / Farhan / Obidit)

Goal: a web app that shows what our toxicity models can do on Bangla, Banglish and English posts
(0 = Explicit, 1 = Subtle, 2 = Neutral).

Stack (proposed): React + Vite front end, FastAPI back end. The results pages read pre-computed JSON,
so only the live demo needs a server.

**Features in scope:** live classifier, script badge, word highlighting, per-model votes, model
comparison, confusion matrices, hard cases, calibration & thresholds.
**Out of scope:** leaderboard story, ensemble weights, error browser, pipeline walkthrough, ablations,
rules/id page, batch CSV upload, Bangla/English UI toggle and dark mode, content-warning blur.

Rule: each person owns the models they built. Ishmam owns the encoders, TF-IDF and the text-only
ensemble (V0–V3.11i). The LLM members (V5.x) are Farhan's competition runs; Obidit exports them for the app.
Obidit owns the deployable model retrain, quality (tests, accessibility) and shipping (deploy, README, demo video).

> **⚠ Data rule check first.** The competition rules (repo guide §1a, §6) say the dataset must not be
> shared or used outside the competition, so a **public** deployment must not show any
> `train.csv` / `test.csv` text: only text the user types in, plus aggregate metrics. The
> **Hard cases** page shows dataset posts, so it stays **local / private**.

---

## Phase 0: together, before splitting

- [ ] Agree on the stack. Write a short spec of principles and features (spec-kit: github.com/github/spec-kit).
- [ ] Choose a design reference from awesome-design-md (github.com/voltagent/awesome-design-md), plus the colours for the three classes.
- [ ] Freeze the **data contract** (below). After this, all three of you can work in parallel without waiting on each other.
- [ ] Create the folders: `App/frontend/`, `App/backend/`, `App/data/` (generated JSON), `App/scripts/` (exporters) and `App/tests/`.

### Data contract (historical standalone draft)

The active five-page showcase follows `docs/SPEC.md` and uses
`scripts/export_results.py`. The standalone implementation from Ishmam is
preserved separately; its API and metrics schemas differ from the active app.

| File / endpoint | Content | Producer |
|---|---|---|
| `private/legacy-exports/models.json` | standalone list schema; aggregate snapshot in `docs/legacy-models.json`; built by `scripts/export_models.py` (merges by id, so run it with your own `--registry`). Per model: id, name, family (tfidf / encoder / llm / ensemble), macro F1, per-class F1, per-script F1, confusion matrix (overall and per script) | Ishmam |
| `private/legacy-exports/oof_rows.json` | per train row (sorted by id): script, true label, fold. **Gitignored** (derived from labels) | Ishmam (done) |
| `private/legacy-exports/oof_<model>.json` | `{id, probs}`: probs[3] per row aligned to `oof_rows.json`, `null` where the model did not predict. **Gitignored** | Ishmam |
| `data/examples.json` | hand-written demo posts per class and script (safe to publish) | Obidit |
| `POST /api/predict` | `{text}` → `{text, script, script_name, primary, label, label_name, models:[{id, name, label, probs[3], mode, has_features}], tokens:[{text, start, end, weight}], explanation, explanation_truncated, latency_ms, warning}` (see `backend/README.md`) | Ishmam (done) |

---

## Ishmam: back end and the Live Demo

**Back end**
- [x] FastAPI `/predict`: runs TF-IDF (instant) and Obidit's transformer, and returns each model's label and probabilities (per-model votes), the script (`text_utils.py`) and per-token importance (TF-IDF coefficients or integrated gradients).
- [x] Exporters in `scripts/` for the encoder, TF-IDF and V3.11i OOF predictions; their rows in the standalone `models.json` schema (snapshot: `docs/legacy-models.json`).

**Page**
- [x] **Live Demo** (`frontend/src/features/live-demo/`): text box, class + confidence bars, script badge, highlighted words, the served models' votes side by side, example posts to try.
- [ ] **Deploy** (moved to Ishmam, 4 Oct): Render or HF Spaces, then run the GitHub Actions checks.
      Progress (4 Oct): int8 serving path verified end to end with a tiny stand-in XLM-R (loads as primary, word
      weights and batching work). `scripts/prepare_space.py` builds a Space folder (`private/space/`, gitignored) with the
      app, public data and `model/transformer-int8/`, CPU torch, `TRANSFORMER_MODEL_PATH` set; it refuses CSV/OOF/joblib files.
      **Waiting on:** Farhan's `transformer-int8.zip`; the team's call on public use of a competition-trained model
      (the organizer check was removed from the plan); Ishmam's Hugging Face login for the upload. Docker build not run locally (no Docker).

## Farhan (updated 4 Oct)

Farhan is back on the app (his earlier GitHub issues are fixed). His LLM tasks return to him; reviews stay
with Ishmam and Obidit.

- [x] **Add the strongest LLMs** (back with Farhan, 4 Oct; done: 10 models exported, V5.11 0.6400 and V5.2avg 0.6123 match RESULTS/EXPERIMENTS) to `scripts/export_results.py`: today it has only V5.0
      (Qwen2.5-7B). Add V5.11 (region-tag Qwen2.5-7B, 0.640, best single model), V5.2avg (Qwen3-14B) and
      V5.3avg (Llama) from `Kernels/V5.*/preds` in the competition repo, re-run it, and check the numbers
      against `RESULTS.md` there.
- [x] **Hard cases, "only the LLMs get right"** (back with Farhan, 4 Oct; done: defined with V5.11, test updated): it is defined with V5.0 alone. Decide
      which LLM(s) it should use (probably V5.11) and update the definition and its test
      (`test_private_export_vote_definitions`). Hard cases stays local; never commit `private/`.
- [x] **Text ensemble** (moved to Farhan, 4 Oct; done: pages now show V3.11i, V3.7 kept only as the alignment check): `scripts/export_results.py` uses V3.7; switch it to **V3.11i** (our final text-only
      submission, private 0.633) or say why V3.7 is better for the pages.
- [x] Add **V1.3avg (MuRIL)** (moved to Farhan, 4 Oct; done: 0.5613) to the exporter (it is in Ishmam's model list but missing from `data/models.json`).
- [x] Update `docs/VERIFICATION.md` (moved to Farhan, 4 Oct; done: export and backend rows; browser suite not re-run) after the merge (19 API tests + 20 browser tests now; see the commit).
- [x] **Train the deployable transformer** (Farhan, done 4 Oct): XLM-R-large, 3 epochs, LR 1e-5, one T4
      (`kaggle/deployable-transformer-v6/`). Duplicate-grouped 20% held-out macro F1 **0.586** (epochs: 0.554, 0.579, 0.586).
      This is a new recipe; never label it with a historical V2.x CV score. int8 export via `scripts/export_cpu.py`
      (gate: macro F1 drop < 0.005, reload check); the exact int8 numbers are in `model/transformer-int8/showcase.json`.
- [ ] **Ishmam: build and deploy with the transformer.** Farhan sends `transformer-int8.zip` (weights are not in git).
      Unzip it so the files sit in `model/transformer-int8/` (`int8_state.pt`, `config.json`, tokenizer files, `showcase.json`),
      install `backend/requirements-transformer.txt`, and run the back end with `TRANSFORMER_MODEL_PATH=model/transformer-int8`:
      it becomes the primary Live Demo model and TF-IDF stays as the second vote. Hosting: int8 XLM-R-large needs ~600 MB RAM,
      above Render free's 512 MB; use HF Spaces CPU (upload the folder to the Space, not to GitHub).


## Obidit: shipping (updated 4 Oct)

Done (thanks): the app shell, Model comparison, Confusion matrices, Calibration, Hard cases (local),
18 examples, tests, accessibility pass, Docker / Render / HF Spaces config, CI, README.
**Changed on 4 Oct:** the team kept Ishmam's back end (`backend/app/`) and Live Demo (`features/live-demo/`).
Your server features (data routes, Hard cases gate, privacy headers, busy lock, static UI, authored fallback,
int8 transformer loading) were merged into `backend/app/`; `backend/main.py` and `src/LiveDemo.tsx` are gone.
Start the server with `python -m uvicorn backend.app.main:app` (Dockerfile, Playwright and README already updated).

- [ ] **Re-record the walkthrough video**: the old one shows the removed Live Demo.

## Shared / end

- [ ] Every PR is reviewed by the other builder: Ishmam reviews Obidit's, Obidit reviews Ishmam's (starting with the 4 Oct merge).
- [ ] Final run-through together before sharing the link.

## Milestones

1. **M1 (MVP):** contract frozen; the live demo running on TF-IDF; the shell and Model comparison page working.
2. **M2:** Obidit's transformer serves the demo; Confusion matrices and Calibration pages done; smoke tests in CI.
3. **M3:** Hard cases, accessibility pass, deploy and the demo video.

Handoff points (4 Oct): Ishmam's back end serves Obidit's transformer once it is trained; until then it serves
TF-IDF (the local full-data model on a teammate's machine, the authored-example model in public builds).
Obidit adds the LLM exports to the results pages through `scripts/export_results.py`.
