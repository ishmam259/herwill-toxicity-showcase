FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package.json
RUN npm ci
COPY frontend frontend
COPY data data
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PUBLIC_DEPLOYMENT=1 ENABLE_PRIVATE_CASES=0 PORT=7860 OMP_NUM_THREADS=1
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN useradd --create-home --uid 1000 appuser
COPY backend backend
COPY data data
COPY --from=frontend /app/frontend/dist frontend/dist
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/api/health',timeout=4)"
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --no-proxy-headers"]
