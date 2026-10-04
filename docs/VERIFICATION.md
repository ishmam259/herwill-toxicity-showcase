# Verification · 4 October 2026

Executed in the datathon workspace against the implemented application.

| Check | Result |
|---|---|
| Historical export | 11 models (4 Oct, Farhan): bundle members plus V1.3avg, V5.11, V5.2avg, V5.3avg and the V3.11i text ensemble read from the competition repo's kernel predictions; 47,817 sorted-ID rows; official data and bundle hashes, probability coverage, and the V3.7 reconstruction (kept as the row-alignment check) verified. Scores match RESULTS/EXPERIMENTS (V5.11 0.6400, V5.2avg 0.6123, V1.3avg 0.5613). V3.11i shows 0.6253 under raw argmax vs its logged 0.6233 cross-fitted score with tuned thresholds |
| Public data check | Pass; aggregate-only results and 18 authored examples |
| Local CPU fit | 47,817 training rows; saved 12 MB TF-IDF artifact; reload and multilingual probability checks pass |
| Backend suite | 19 passed, 1 skipped (4 Oct re-run after the merge and export changes); includes private export vote definitions and saved local model. Those two tests intentionally skip when ignored artifacts are absent in a fresh checkout |
| Production frontend | TypeScript strict checks and Vite build pass |
| Browser suite | 20 passed across Chromium desktop and Pixel 7 viewport; 2 walkthrough cases intentionally skipped during normal test runs |
| Accessibility | axe WCAG A/AA checks pass on all five pages in desktop/mobile layouts; keyboard prediction and keyboard-scrollable tables checked |
| Responsive layout | No document-level horizontal overflow on any page; desktop/mobile screenshots manually reviewed |
| Docker | Non-root production image builds; in-image smoke checks compiled routes, prediction, public allowlist, private route denial even when the local flag is set, and absence of private exports/competition weights |
| Walkthrough | Desktop recording passed; authored examples and public aggregates only; copied to ignored `output/showcase-walkthrough.webm` |
| Source checks | Ruff checks, Python compilation and `git diff --check` pass |

Local preview: http://127.0.0.1:7860. It serves the saved local full-data TF-IDF model and keeps private cases disabled. Default fresh-checkout and Docker inference uses the authored-example model. Preview screenshots are in ignored `output/`; they contain no competition posts.

A Starlette deprecation warning currently recommends httpx2 for its test client; all HTTP assertions pass with the declared httpx development dependency. There are no failed checks being concealed.

Not executed: saved transformer GPU training, its int8 quality/latency gate, external GitHub Actions, teammate review, or remote deployment. No trained transformer or neural benchmark result is claimed. The optional training/export scripts passed syntax and static checks only. The historical metrics remain adaptively inspected diagnostic evidence, not newly certified validation.
