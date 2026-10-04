import json

import pytest
from fastapi.testclient import TestClient

from backend.main import ROOT, app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SPARSE_MODEL_PATH", raising=False)
    monkeypatch.delenv("TRANSFORMER_MODEL_PATH", raising=False)
    monkeypatch.delenv("ENABLE_PRIVATE_CASES", raising=False)
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    "text,script",
    [
        ("This is a test.", "latin"),
        ("এটি একটি পরীক্ষা।", "bangla"),
        ("এটি a test", "mixed"),
        ("🙂?!", "other"),
    ],
)
def test_multilingual_predictions(client, text, script):
    response = client.post("/api/predict", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert result["script"]["id"] == script
    assert "".join(t["text"] for t in result["tokens"]) == text
    for m in result["models"]:
        assert m["label"] in [0, 1, 2]
        assert len(m["probs"]) == 3
        assert abs(sum(m["probs"]) - 1) < 1e-8
        assert m["probs"][m["label"]] == max(m["probs"])
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "  \n  "},
        {"text": "a" * 2001},
        {"text": 42},
        {"text": "hello", "model": "invented"},
        {},
    ],
)
def test_reject_invalid_input(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_public_allowlist(client):
    for name in ("models", "calibration", "examples"):
        assert client.get("/api/data/" + name).status_code == 200
    for path in (
        "/api/data/hard_cases",
        "/api/data/oof_V0.0",
        "/private/hard_cases.json",
        "/model/sparse.joblib",
        "/data/models.json",
        "/api/unknown",
    ):
        assert client.get(path).status_code == 404
    health = client.get("/api/health").json()
    assert not health["private_cases"]
    assert not health["stores_user_text"]
    assert not health["transformer_available"]


def test_private_default_and_spoofed_forwarded_header(client, monkeypatch, tmp_path):
    from backend import main

    private = tmp_path / "hard_cases.json"
    private.write_text('{"cases":[]}')
    monkeypatch.setattr(main, "PRIVATE", private)
    assert client.get("/api/hard-cases").status_code == 403
    monkeypatch.setenv("ENABLE_PRIVATE_CASES", "1")
    assert (
        client.get(
            "/api/hard-cases", headers={"X-Forwarded-For": "127.0.0.1"}
        ).status_code
        == 403
    )
    with TestClient(app, client=("127.0.0.1", 12345)) as local:
        assert local.get("/api/hard-cases").status_code == 200
        assert local.get("/api/health").json()["private_cases"]
        monkeypatch.setenv("PUBLIC_DEPLOYMENT", "1")
        assert local.get("/api/hard-cases").status_code == 403
        assert not local.get("/api/health").json()["private_cases"]


def test_busy_inference(client):
    with app.state.inference_lock:
        assert client.post("/api/predict", json={"text": "Hello"}).status_code == 429


def test_deletion_influence_is_reproducible(client):
    text = "You are an idiot."
    result = client.post("/api/predict", json={"text": text}).json()
    sparse = app.state.sparse
    label = result["models"][0]["label"]
    base = sparse.probabilities([text])[0, label]
    position = 0
    for token in result["tokens"]:
        if not token["text"].isspace():
            from backend.inference import normalize

            removed = text[:position] + text[position + len(token["text"]) :]
            actual = base - sparse.probabilities([normalize(removed)])[0, label]
            assert abs(actual - token["weight"]) < 1e-10
        position += len(token["text"])


def test_public_data_contract():
    from scripts.check_public_data import check

    check()


def test_private_export_vote_definitions():
    path = ROOT / "private/hard_cases.json"
    if not path.is_file():
        pytest.skip("Private workspace export is intentionally absent from public CI")
    value = json.loads(path.read_text())
    counts = {"all-wrong": 0, "llm-only": 0}
    for case in value["cases"]:
        counts[case["category"]] += 1
        label = case["label"]
        votes = {v["id"]: v["label"] for v in case["votes"]}
        if case["category"] == "all-wrong":
            assert all(v is None or v != label for v in votes.values())
        else:
            assert votes["V5.0"] == label
            assert all(
                v is None or v != label
                for name, v in votes.items()
                if name not in {"V5.0", "V3.7"}
            )
    assert all(count <= 40 for count in counts.values())


def test_saved_local_model_when_present():
    from backend.inference import SparsePredictor

    path = ROOT / "model/sparse.joblib"
    if not path.is_file():
        pytest.skip("Fitted competition artifacts are not included in public CI")
    predictor = SparsePredictor(path)
    assert predictor.metadata["mode"] == "local-trained"
    for text in [
        "Thanks for the thoughtful explanation.",
        "আজকের আলোচনা ভালো লেগেছে।",
        "Ajker discussion bhalo legeche.",
    ]:
        vote = predictor.predict(text)
        assert vote["label"] in [0, 1, 2]
        assert abs(sum(vote["probs"]) - 1) < 1e-8
