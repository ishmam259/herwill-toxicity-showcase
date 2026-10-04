# Toxicity Lab specification

## Scope and experience

Five pages: Live demo, Model comparison, Confusion matrices, Calibration & thresholds, and local Hard cases. React + TypeScript + Vite provides the interface; FastAPI provides CPU inference and gated private reads. Research pages remain usable without an inference server because they bundle aggregate JSON only. Batch uploads, ID-based predictions, dark mode, translation, leaderboard narratives and pipeline walkthroughs are outside scope.

The design uses a quiet research workspace: warm white surfaces, plum navigation and actions, Manrope headings, DM Sans body text, and Noto Sans Bengali where available. Colors identify classes consistently: Explicit red (#ae3d45), Subtle amber (#9c6b17), Neutral green (#287767). Badges include class names so color is never the only signal. The layout collapses to a keyboard-accessible mobile menu. Google Fonts are optional; system fonts keep the interface usable offline.

## Frozen contracts (schema version 1)

- `data/examples.json`: original authored examples with `id`, `language`, intended `label`, and `text`; labels 0 Explicit, 1 Subtle, 2 Neutral.
- `data/models.json`: verified historical model aggregates. Each model has `id`, `name`, `family`, `specialist`, `macro_f1`, `per_class_f1[3]`, `confusion[3][3]`, `support[3]`, `rows`, and `by_script.B/L` with the same summary fields. Missing script coverage is absent, never filled with zero. All displayed comparisons use raw argmax, including V3.7; its submitted offsets are not silently applied.
- `data/calibration.json`: top-class reliability bins, ECE, multiclass Brier, and a 21×21 grid of macro F1 at Subtle/Neutral log-probability offsets −0.50…+0.50 in steps of 0.05. Explicit offset is 0. The slider uses exact precomputed aggregates; row-level OOF is never bundled publicly.
- `POST /api/predict`: `{text: string}` (1–2,000 characters, nonblank) returns `script`, `models:[{id,name,label,probs[3],mode,has_features}]`, `tokens:[{text,weight}]`, timing, explanation provenance and warnings. Whitespace is preserved in returned tokens. The frontend clears results on edits and aborts in-flight requests.
- `GET /api/health`: current served models, transformer availability, private mode availability, and storage policy. User text is not persisted.
- `GET /api/data/{models,examples,calibration}`: explicit public allowlist.
- `GET /api/hard-cases`: requires ENABLE_PRIVATE_CASES=1, PUBLIC_DEPLOYMENT!=1, an existing private export and an actual loopback peer. It returns at most 40 cases for each collection. Every covered member plus the ensemble must be wrong for “all wrong”; “LLM only” means the LLM is correct and all covered non-LLM members are wrong. The ensemble vote is displayed but not used in the LLM-only selection.

## Evidence and data boundaries

Historical artifacts use ascending numeric ID alignment and class order [0,1,2]. Export checks official data and prediction SHA-256 hashes, ID parity, probability sums, finite coverage and saved V3.7 reconstruction. B means any Bangla character, including mixed script. L means no Bangla characters, including English, Banglish and other scripts. These are script proxies, not annotated language groups. Specialists cover only B. Historical adaptively inspected OOF is a diagnostic, not clean validation.

Raw competition text, CSVs, OOF rows and all fitted estimators remain ignored and excluded from Docker. The public fallback is a fitted TF-IDF classifier on 18 authored examples, explicitly marked illustrative. Its probabilities are not validated calibration or competition performance. A separate full-fit local TF-IDF artifact and optional saved transformer can be served by operator-supplied environment paths. Pickle/joblib paths are trusted local artifacts, never uploads.
