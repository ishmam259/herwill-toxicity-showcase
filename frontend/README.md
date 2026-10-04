# Front end (React + Vite)

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api to the back end on :8000
npm run build
```

- `src/features/live-demo/`: the Live Demo (Ishmam). It is a standalone component; `src/App.tsx` mounts it until
  Farhan's app shell lands, and then the shell mounts `<LiveDemo />` on its own route.
- Design tokens (colours, fonts) are in `src/index.css`; the shell owns them.
- `VITE_API_BASE` sets the API origin for a deployed build (empty = same origin).
