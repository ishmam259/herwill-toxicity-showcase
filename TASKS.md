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
ensemble (V0–V3.11i). Farhan owns the LLM members (V5.x). Obidit owns the deployable model retrain,
quality (tests, accessibility) and shipping (deploy, README, demo video).

> **⚠ Data rule check first.** The competition rules (repo guide §1a, §6) say the dataset must not be
> shared or used outside the competition. Until the organizers say otherwise, a **public** deployment
> must not show any `train.csv` / `test.csv` text: only text the user types in, plus aggregate
> metrics. The **Hard cases** page shows dataset posts, so it stays **local / private**. Obidit asks
> the organizers (task O1).

---

## Phase 0: together, before splitting

- [ ] Agree on the stack. Write a short spec of principles and features (spec-kit: github.com/github/spec-kit).
- [ ] Choose a design reference from awesome-design-md (github.com/voltagent/awesome-design-md), plus the colours for the three classes.
- [ ] Freeze the **data contract** (below). After this, all three of you can work in parallel without waiting on each other.
- [ ] Create the folders: `App/frontend/`, `App/backend/`, `App/data/` (generated JSON), `App/scripts/` (exporters) and `App/tests/`.

### Data contract (draft)

| File / endpoint | Content | Producer |
|---|---|---|
| `data/models.json` | per model: id, name, family (tfidf / encoder / llm / ensemble), macro F1, per-class F1, per-script F1, confusion matrix (overall and per script) | Ishmam + Farhan (own models) |
| `data/oof_<model>.json` | per train row: idx, script, true label, probs[3], fold (text joined locally only, see the data rule) | Ishmam + Farhan (own models) |
| `data/examples.json` | hand-written demo posts per class and script (safe to publish) | Obidit |
| `POST /api/predict` | `{text}` → `{script, models:[{id, label, probs[3]}], tokens:[{text, weight}]}` | Ishmam |

---

## Ishmam: back end and the Live Demo

**Back end**
- [ ] FastAPI `/predict`: runs TF-IDF (instant) and Obidit's transformer, and returns each model's label and probabilities (per-model votes), the script (`text_utils.py`) and per-token importance (TF-IDF coefficients or integrated gradients).
- [ ] Exporters in `App/scripts/` for the encoder, TF-IDF and V3.11i OOF predictions; their rows in `models.json`.

**Page**
- [ ] **Live Demo**: text box, class + confidence bars, script badge, highlighted words, the served models' votes side by side, example posts to try.

## Farhan: app shell and the results pages

**Front end and data**
- [ ] App shell: routing, layout, design tokens.
- [ ] Exporters for the LLM members (V5.x); their rows in `models.json`.

**Pages**
- [ ] **Model comparison**: macro / per-class / per-script F1 for all members.
- [ ] **Confusion matrices**: switchable by model and by script; highlight Subtle vs Explicit.
- [ ] **Hard cases** (local only): posts every model gets wrong, and posts only the LLMs get right, with every model's vote on each post.

## Obidit: the demo model, calibration and shipping

**Model** (the `obiditislam` Kaggle account still has GPU hours)
- [ ] Retrain one deployable transformer (BanglaBERT or XLM-R-large) on the full train set with the V1.x recipe, and **save the weights** (none are saved now). Hand Ishmam a `model/` folder plus the expected CV score.
- [ ] Shrink it for CPU serving (ONNX or int8 dynamic quantization); check that macro F1 drops by less than 0.005 and measure latency per post.

**Page and shipping**
- [ ] O1: ask the organizers whether a public demo may use the trained model and show dataset posts; record the answer here.
- [ ] `data/examples.json`: 15–20 hand-written example posts (Bangla, Banglish, English × the 3 classes), not copied from the dataset.
- [ ] **Calibration & thresholds** page: reliability curves; an offset slider that recomputes macro F1 live from the OOF predictions.
- [ ] Playwright smoke tests: the demo returns a label; every page renders.
- [ ] Accessibility pass (keyboard, contrast, screen-reader labels on charts).
- [ ] Dockerfile, the deploy target (HF Spaces / Render) and a CI check that runs the tests.
- [ ] README with run instructions, and the demo video (Playwright screen recording).

---

## Shared / end

- [ ] Every PR is reviewed by one of the other two.
- [ ] Final run-through together before sharing the link.

## Milestones

1. **M1 (MVP):** contract frozen; O1 sent; the live demo running on TF-IDF; the shell and Model comparison page working.
2. **M2:** Obidit's transformer serves the demo; Confusion matrices and Calibration pages done; smoke tests in CI.
3. **M3:** Hard cases, accessibility pass, deploy and the demo video.

Handoff points: Farhan's pages use Ishmam's OOF exports (Phase 0 contract). Ishmam's back end serves
Obidit's model; until it lands, it serves TF-IDF only. Ishmam's and Obidit's pages render inside
Farhan's shell, so build them as standalone components until the shell lands.
