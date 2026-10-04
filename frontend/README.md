# Front end (React + Vite)

The active five-page showcase uses `src/App.tsx`, `src/LiveDemo.tsx` and
`src/styles.css`, with the API in `backend/main.py`. From the repository root:

```bash
npm ci
npm run dev      # http://127.0.0.1:5173, proxies /api to :8000
npm run build
```

Dependencies are managed by the root workspace and `package-lock.json`.

Ishmam's earlier standalone demo is preserved in `src/features/live-demo/`,
with its design tokens in `src/index.css`. It expects the separate legacy API
in `backend/app/main.py`; its response format differs from the active API.
It is not mounted by the active app. `VITE_API_BASE` sets its API origin
(empty = same origin). See [the backend README](../backend/README.md).
