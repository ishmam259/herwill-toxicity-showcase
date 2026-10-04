"""Assemble a Hugging Face Space (Docker SDK) folder that serves the int8 transformer.

The public Dockerfile in the repo root deliberately ships no model. A Space that should run the transformer
needs the weights inside its own repo, so this script builds a separate folder, private/space/ (gitignored),
with the app, the public data files and model/transformer-int8/. Nothing here is uploaded anywhere: push the
folder to your Space yourself (see the printed steps). Weights never go to GitHub.

    python scripts/prepare_space.py                       # uses model/transformer-int8
    python scripts/prepare_space.py --model path/to/transformer-int8 --out private/space
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["config.json", "int8_state.pt", "showcase.json", "tokenizer.json", "tokenizer_config.json"]
PUBLIC_DATA = ["models.json", "calibration.json", "examples.json"]

DOCKERFILE = """\
FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
COPY frontend/package.json frontend/package.json
RUN npm ci
COPY frontend frontend
COPY data data
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PUBLIC_DEPLOYMENT=1 ENABLE_PRIVATE_CASES=0 PORT=7860 \\
    OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    TRANSFORMER_MODEL_PATH=/app/model/transformer-int8
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \\
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \\
 && pip install --no-cache-dir "transformers>=4.48,<6"
RUN useradd --create-home --uid 1000 appuser
COPY backend backend
COPY data data
COPY model/transformer-int8 model/transformer-int8
COPY --from=frontend /app/frontend/dist frontend/dist
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/api/health',timeout=4)"
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860} --no-proxy-headers"]
"""

GITATTRIBUTES = "*.pt filter=lfs diff=lfs merge=lfs -text\n*.json filter=lfs diff=lfs merge=lfs -text\n"


def ignore(dirpath, names):
    skip = {".venv", "node_modules", "dist", "__pycache__", "tests", "model", ".pytest_cache"}
    return [n for n in names if n in skip or n.endswith((".joblib", ".pyc"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "model" / "transformer-int8")
    ap.add_argument("--out", type=Path, default=ROOT / "private" / "space")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not (args.model / f).is_file()]
    if missing:
        sys.exit(f"{args.model} is missing {missing}. Unzip Farhan's transformer-int8.zip there first.")
    meta = json.loads((args.model / "showcase.json").read_text(encoding="utf-8"))
    if meta.get("class_order") != [0, 1, 2] or meta.get("inference_backend") != "dynamic-int8":
        sys.exit("showcase.json must declare class_order [0, 1, 2] and inference_backend dynamic-int8")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for name in ("package.json", "package-lock.json"):
        shutil.copy2(ROOT / name, args.out / name)
    shutil.copytree(ROOT / "frontend", args.out / "frontend", ignore=ignore)
    shutil.copytree(ROOT / "backend", args.out / "backend", ignore=ignore)
    (args.out / "data").mkdir()
    for name in PUBLIC_DATA:
        shutil.copy2(ROOT / "data" / name, args.out / "data" / name)
    dest = args.out / "model" / "transformer-int8"
    dest.mkdir(parents=True)
    for f in args.model.iterdir():
        if f.is_file() and f.suffix != ".bin" and f.name != "model.safetensors":  # int8 only, never fp32 weights
            shutil.copy2(f, dest / f.name)
    (args.out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (args.out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    front = readme.split("---", 2)
    (args.out / "README.md").write_text(
        f"---{front[1]}---\n\n# HerWILL Toxicity Lab\n\nLive demo of the event_horizon team's toxicity models "
        "(HerWILL x UAP Safe Social Media Datathon 2026). Source: https://github.com/ishmam259/herwill-toxicity-showcase\n",
        encoding="utf-8")

    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    leaks = [str(f.relative_to(args.out)) for f in args.out.rglob("*")
             if f.suffix in {".csv", ".joblib", ".npy"} or "private" in f.relative_to(args.out).parts or "oof_" in f.name]
    if leaks:
        sys.exit(f"Refusing: competition-derived files in the Space folder: {leaks}")
    print(f"Space folder ready: {args.out} ({size:.0f} MB, model {meta.get('name', meta.get('id'))})")
    print("Next, from that folder, with your own Hugging Face login (not done by this script):")
    print("  huggingface-cli login")
    print("  huggingface-cli upload <your-user>/herwill-toxicity-lab . . --repo-type space")


if __name__ == "__main__":
    main()
