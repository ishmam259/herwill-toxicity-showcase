from __future__ import annotations

import ipaddress
import json
import os
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.inference import (
    ROOT,
    SparsePredictor,
    TransformerPredictor,
    predict_response,
)

PUBLIC_FILES = {"models", "examples", "calibration"}
PRIVATE = ROOT / "private/hard_cases.json"
DIST = ROOT / "frontend/dist"


def is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def private_available():
    return (
        os.getenv("ENABLE_PRIVATE_CASES") == "1"
        and os.getenv("PUBLIC_DEPLOYMENT") != "1"
        and PRIVATE.is_file()
    )


@asynccontextmanager
async def lifespan(app):
    sparse_path = os.getenv("SPARSE_MODEL_PATH")
    transformer_path = os.getenv("TRANSFORMER_MODEL_PATH")
    app.state.sparse = SparsePredictor(sparse_path)
    app.state.transformer = (
        TransformerPredictor(transformer_path) if transformer_path else None
    )
    app.state.inference_lock = Lock()
    yield


app = FastAPI(title="HerWILL Toxicity Showcase", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Enter some text to classify.")
        return value


@app.middleware("http")
async def privacy_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
def health(request: Request):
    sparse = request.app.state.sparse
    models = [
        {
            "id": sparse.metadata["id"],
            "name": sparse.metadata["name"],
            "mode": sparse.metadata["mode"],
        }
    ]
    if request.app.state.transformer:
        meta = request.app.state.transformer.metadata
        models.append(
            {
                "id": meta.get("id", "deployable-transformer"),
                "name": meta.get("name", "Local transformer"),
                "mode": "local-trained",
            }
        )
    return {
        "status": "ok",
        "models": models,
        "private_cases": private_available() and is_loopback(request.client.host),
        "transformer_available": bool(request.app.state.transformer),
        "stores_user_text": False,
    }


@app.post("/api/predict")
def predict(payload: PredictRequest, request: Request):
    # Bound concurrent CPU work. The server does not save or log request bodies.
    if not request.app.state.inference_lock.acquire(blocking=False):
        raise HTTPException(
            429, "Another prediction is running. Please try again shortly."
        )
    try:
        return predict_response(
            payload.text, request.app.state.sparse, request.app.state.transformer
        )
    finally:
        request.app.state.inference_lock.release()


@app.get("/api/data/{name}")
def data(name: str):
    if name not in PUBLIC_FILES:
        raise HTTPException(404, "Unknown public data file")
    path = ROOT / "data" / f"{name}.json"
    if not path.is_file():
        raise HTTPException(503, "Results have not been exported yet.")
    return FileResponse(path, media_type="application/json")


@app.get("/api/hard-cases")
def hard_cases(request: Request):
    # Checking the actual socket peer (not X-Forwarded-For) is essential.
    # Run uvicorn with --no-proxy-headers and bind 127.0.0.1 in private mode.
    if not private_available() or not is_loopback(request.client.host):
        raise HTTPException(
            403,
            "Competition posts are available only in explicitly enabled local mode.",
        )
    if os.getenv("PUBLIC_DEPLOYMENT") == "1":
        raise HTTPException(403, "Private cases are disabled in public deployments.")
    return json.loads(PRIVATE.read_text())


# Serve only the compiled UI, never the repository, data directory or model folder.
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if path.startswith("api/") or path == "api":
        raise HTTPException(404, "Unknown API route")
    if path == "favicon.svg" and (DIST / "favicon.svg").is_file():
        return FileResponse(DIST / "favicon.svg")
    if (
        path in {"", "compare", "confusion", "calibration", "hard-cases"}
        and (DIST / "index.html").is_file()
    ):
        return FileResponse(DIST / "index.html")
    raise HTTPException(
        404,
        "Build the frontend with npm run build, or use the Vite development server.",
    )
