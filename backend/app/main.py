"""FastAPI back end: the classifier, the public result files and the compiled front end.

Run from the repo root:  python -m uvicorn backend.app.main:app --port 8000 --no-proxy-headers
(or from backend/:       python -m uvicorn app.main:app --port 8000)
"""
import ipaddress
import json
import os
import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .explain import EXPLANATION, occlusion
from .predictors import ROOT, load_predictors
from .text import MAX_CHARS, normalize, script_bucket

LABELS = ["explicit", "subtle", "neutral"]  # 0, 1, 2 as in the competition
SCRIPT_NAMES = {"bangla": "Bangla script", "latin": "Latin script", "mixed": "Mixed script", "none": "No letters"}
PUBLIC_FILES = {"models", "examples", "calibration"}
PRIVATE = ROOT / "private" / "hard_cases.json"
DIST = ROOT / "frontend" / "dist"
SPA_ROUTES = {"", "compare", "confusion", "calibration", "hard-cases"}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictors"] = load_predictors()
    state["lock"] = Lock()
    yield
    state.clear()


app = FastAPI(title="HerWILL Toxicity Showcase API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def privacy_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class PredictIn(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter some text to classify.")
        return value


class ModelVote(BaseModel):
    id: str
    name: str
    label: int
    probs: list[float]
    mode: str  # "local-trained" or "illustrative"
    has_features: bool


class Token(BaseModel):
    text: str
    start: int
    end: int
    weight: float


class PredictOut(BaseModel):
    text: str  # normalized text; token offsets refer to this string
    script: str
    script_name: str
    primary: str  # id of the model whose label is the headline answer and whose words are explained
    label: int
    label_name: str
    models: list[ModelVote]
    tokens: list[Token]
    explanation: str
    explanation_truncated: bool
    latency_ms: float
    warning: str | None


def is_loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def private_available() -> bool:
    return (os.environ.get("ENABLE_PRIVATE_CASES") == "1" and os.environ.get("PUBLIC_DEPLOYMENT") != "1"
            and PRIVATE.is_file())


@app.get("/api/health")
def health(request: Request):
    predictors = state.get("predictors", [])
    return {
        "status": "ok",
        "models": [{"id": p.id, "name": p.name, "mode": p.mode} for p in predictors],
        "private_cases": private_available() and is_loopback(request.client.host),
        "transformer_available": len(predictors) > 1,
        "stores_user_text": False,
    }


@app.post("/api/predict", response_model=PredictOut)
def predict(body: PredictIn):
    # Bound concurrent CPU work. Request bodies are never saved or logged.
    if not state["lock"].acquire(blocking=False):
        raise HTTPException(429, "Another prediction is running. Please try again shortly.")
    try:
        started = time.perf_counter()
        text = normalize(body.text)
        predictors = state["predictors"]
        votes = []
        for p in predictors:
            probs = p.predict_proba([text])[0]
            votes.append(ModelVote(id=p.id, name=p.name, label=int(probs.argmax()),
                                   probs=[round(float(x), 4) for x in probs], mode=p.mode,
                                   has_features=p.has_features(text)))
        primary = predictors[-1]
        label = votes[-1].label
        tokens, truncated = occlusion(primary, text, label)
        script = script_bucket(text)
        return PredictOut(text=text, script=script, script_name=SCRIPT_NAMES[script], primary=primary.id,
                          label=label, label_name=LABELS[label], models=votes, tokens=tokens,
                          explanation=EXPLANATION, explanation_truncated=truncated,
                          latency_ms=round((time.perf_counter() - started) * 1000, 1),
                          warning=primary.metadata.get("warning"))
    finally:
        state["lock"].release()


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
    # Check the socket peer, not X-Forwarded-For: run uvicorn with --no-proxy-headers and bind 127.0.0.1.
    if not private_available() or not is_loopback(request.client.host):
        raise HTTPException(403, "Competition posts are available only in explicitly enabled local mode.")
    return json.loads(PRIVATE.read_text(encoding="utf-8"))


# Serve only the compiled UI, never the repository, data directory or model folder.
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "Unknown API route")
    if path == "favicon.svg" and (DIST / "favicon.svg").is_file():
        return FileResponse(DIST / "favicon.svg")
    if path in SPA_ROUTES and (DIST / "index.html").is_file():
        return FileResponse(DIST / "index.html")
    raise HTTPException(404, "Build the front end with npm run build, or use the Vite dev server.")
