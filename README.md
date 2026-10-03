# HerWILL Toxicity Showcase

A web app that shows what the event_horizon team's models (HerWILL × UAP Safe Social Media Datathon 2026, 4th place)
can do on Bangla, Banglish and English posts: 0 = Explicit, 1 = Subtle, 2 = Neutral.

- Task split and data contract: [TASKS.md](TASKS.md)
- Stack (proposed): React + Vite (`frontend/`), FastAPI (`backend/`), pre-computed JSON (`data/`), exporters (`scripts/`), Playwright (`tests/`).

**Data rule:** the competition dataset must never be committed here or shown in a public deployment. Only text users type in,
hand-written examples and aggregate metrics are public. Hard cases stays local.
