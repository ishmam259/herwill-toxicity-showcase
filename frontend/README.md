# Front end (React + Vite)

The active five-page showcase uses `src/App.tsx` (the shell), `src/features/live-demo/` (the Live Demo on `/`) and
`src/styles.css`, with the API in `backend/app/` (see [the backend README](../backend/README.md)). From the repository root:

```bash
npm ci
npm run dev      # http://127.0.0.1:5173, proxies /api to :8000
npm run build
```

Dependencies are managed by the root workspace and `package-lock.json`.

The Live Demo's classes are prefixed `lens-` and its tokens are scoped to `.lens`, so it does not touch the
shell's styles. `VITE_API_BASE` sets its API origin (empty = same origin).
