import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI back end runs on :8000 (see backend/README); the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://localhost:8000' },
  },
})
