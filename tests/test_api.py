"""API tests. They use the illustrative model (fitted on data/examples.json) or a toy model, so they run
without competition data, as in CI."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.app import main
from backend.app.main import app
from backend.app.predictors import ROOT, TfidfPredictor
from backend.app.text import normalize


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SPARSE_MODEL_PATH", "")  # force the illustrative model
    for var in ("TRANSFORMER_MODEL_PATH", "MODEL_DIR", "ENABLE_PRIVATE_CASES", "PUBLIC_DEPLOYMENT"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize(
    "text,script",
    [("This is a test.", "latin"), ("এটি একটি পরীক্ষা।", "bangla"), ("এটি a test", "mixed"), ("🙂?!", "none")],
)
def test_multilingual_predictions(client, text, script):
    response = client.post("/api/predict", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert result["script"] == script
    assert result["primary"] == "authored-tfidf" and result["warning"]
    for t in result["tokens"]:
        assert result["text"][t["start"]:t["end"]] == t["text"]
    for m in result["models"]:
        assert m["label"] in [0, 1, 2] and m["mode"] == "illustrative"
        assert abs(sum(m["probs"]) - 1) < 1e-3
        assert m["probs"][m["label"]] == max(m["probs"])
    assert response.headers["cache-control"] == "no-store"


def test_text_is_normalized_and_offsets_match(client):
    body = client.post("/api/predict", json={"text": "  you   stupid idiot  "}).json()
    assert body["text"] == "you stupid idiot"
    assert [t["text"] for t in body["tokens"]] == ["you", "stupid", "idiot"]


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {"text": "  \n  "}, {"text": "a" * 2001}, {"text": 42}, {"text": "hello", "model": "invented"}, {}],
)
def test_reject_invalid_input(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_public_allowlist(client):
    for name in ("models", "calibration", "examples"):
        assert client.get("/api/data/" + name).status_code == 200
    for path in ("/api/data/hard_cases", "/api/data/oof_V0.0", "/private/hard_cases.json", "/model/sparse.joblib",
                 "/backend/model/tfidf.joblib", "/data/models.json", "/api/unknown"):
        assert client.get(path).status_code == 404
    health = client.get("/api/health").json()
    assert not health["private_cases"] and not health["stores_user_text"] and not health["transformer_available"]


def test_private_default_and_spoofed_forwarded_header(client, monkeypatch, tmp_path):
    private = tmp_path / "hard_cases.json"
    private.write_text('{"cases":[]}')
    monkeypatch.setattr(main, "PRIVATE", private)
    assert client.get("/api/hard-cases").status_code == 403
    monkeypatch.setenv("ENABLE_PRIVATE_CASES", "1")
    assert client.get("/api/hard-cases", headers={"X-Forwarded-For": "127.0.0.1"}).status_code == 403
    with TestClient(app, client=("127.0.0.1", 12345)) as local:
        assert local.get("/api/hard-cases").status_code == 200
        assert local.get("/api/health").json()["private_cases"]
        monkeypatch.setenv("PUBLIC_DEPLOYMENT", "1")
        assert local.get("/api/hard-cases").status_code == 403
        assert not local.get("/api/health").json()["private_cases"]


def test_busy_inference(client):
    with main.state["lock"]:
        assert client.post("/api/predict", json={"text": "Hello"}).status_code == 429


def test_deletion_influence_is_reproducible(client):
    text = "You are an idiot."
    result = client.post("/api/predict", json={"text": text}).json()
    model = main.state["predictors"][-1]
    label = result["label"]
    base = model.predict_proba([text])[0, label]
    for t in result["tokens"]:
        removed = normalize(text[:t["start"]] + text[t["end"]:])
        assert abs((base - model.predict_proba([removed])[0, label]) - t["weight"]) < 1e-4


def test_saved_model_formats_load(tmp_path):
    texts, labels = ["you stupid idiot", "nice try genius", "thanks for sharing"], [0, 1, 2]
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    clf = LogisticRegression(max_iter=500).fit(v.fit_transform(texts), labels)
    ours = tmp_path / "tfidf.joblib"
    joblib.dump({"vectorizers": [v], "clf": clf}, ours)  # train_tfidf.py format
    assert TfidfPredictor.from_file(ours).mode == "local-trained"
    w = TfidfVectorizer(analyzer="word")
    from scipy.sparse import hstack
    clf2 = LogisticRegression(max_iter=500).fit(hstack([v.transform(texts), w.fit_transform(texts)]), labels)
    theirs = tmp_path / "sparse.joblib"
    joblib.dump({"char": v, "word": w, "model": clf2, "metadata": {"id": "local-tfidf", "name": "TF-IDF",
                                                                    "mode": "local-trained"}}, theirs)  # train_sparse.py
    p = TfidfPredictor.from_file(theirs)
    assert p.id == "local-tfidf" and abs(p.predict_proba(["idiot"])[0].sum() - 1) < 1e-6


def test_public_deployment_ignores_local_model(monkeypatch):
    from backend.app import predictors

    monkeypatch.delenv("SPARSE_MODEL_PATH", raising=False)
    monkeypatch.setenv("PUBLIC_DEPLOYMENT", "1")
    assert predictors.load_sparse().mode == "illustrative"


def test_public_data_contract():
    from scripts.check_public_data import check

    check()


def test_private_export_vote_definitions():
    path = ROOT / "private/hard_cases.json"
    if not path.is_file():
        pytest.skip("Private workspace export is intentionally absent from public CI")
    value = json.loads(path.read_text(encoding="utf-8"))
    counts = {"all-wrong": 0, "llm-only": 0}
    for case in value["cases"]:
        counts[case["category"]] += 1
        label = case["label"]
        votes = {v["id"]: v["label"] for v in case["votes"]}
        if case["category"] == "all-wrong":
            assert all(v is None or v != label for v in votes.values())
        else:
            # V5.11 (best single LLM) is right; every non-LLM member is wrong.
            assert votes["V5.11"] == label
            llms = {"V5.0", "V5.11", "V5.2avg", "V5.3avg", "V3.11i"}
            assert all(v is None or v != label for name, v in votes.items() if name not in llms)
    assert all(count <= 40 for count in counts.values())


def test_saved_local_model_when_present():
    path = ROOT / "backend/model/tfidf.joblib"
    if not path.is_file():
        pytest.skip("Fitted competition artifacts are not included in public CI")
    predictor = TfidfPredictor.from_file(path)
    for text in ["Thanks for the thoughtful explanation.", "আজকের আলোচনা ভালো লেগেছে।", "Ajker discussion bhalo legeche."]:
        probs = predictor.predict_proba([normalize(text)])[0]
        assert abs(probs.sum() - 1) < 1e-6
