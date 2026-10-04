"""FastAPI back end for the Live Demo. Run: uvicorn app.main:app --reload (from backend/)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .explain import occlusion
from .predictors import load_predictors
from .text import MAX_CHARS, normalize, script_bucket

LABELS = ["explicit", "subtle", "neutral"]  # 0, 1, 2 as in the competition
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", Path(__file__).resolve().parents[1] / "model"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictors"] = load_predictors(MODEL_ROOT)
    yield
    state.clear()


app = FastAPI(title="HerWILL Toxicity Showcase API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS * 2)


class ModelVote(BaseModel):
    id: str
    name: str
    label: int
    probs: list[float]


class Token(BaseModel):
    text: str
    start: int
    end: int
    weight: float


class PredictOut(BaseModel):
    text: str  # normalized text; token offsets refer to this string
    script: str
    primary: str  # id of the model whose label is the headline answer and whose tokens are shown
    label: int
    label_name: str
    models: list[ModelVote]
    tokens: list[Token]


@app.get("/api/health")
def health():
    return {"ok": True, "models": [p.id for p in state.get("predictors", [])]}


@app.post("/api/predict", response_model=PredictOut)
def predict(body: PredictIn):
    text = normalize(body.text)
    if not text:
        raise HTTPException(422, "text is empty after normalization")
    predictors = state["predictors"]
    votes = []
    for p in predictors:
        probs = p.predict_proba([text])[0]
        votes.append(ModelVote(id=p.id, name=p.name, label=int(probs.argmax()),
                               probs=[round(float(x), 4) for x in probs]))
    primary = predictors[-1]
    label = votes[-1].label
    return PredictOut(text=text, script=script_bucket(text), primary=primary.id, label=label,
                      label_name=LABELS[label], models=votes, tokens=occlusion(primary, text, label))
