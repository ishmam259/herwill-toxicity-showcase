"""API tests on a tiny toy model, so they run without the competition data (and in CI)."""
import importlib

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TOY = [("you are a stupid idiot", 0), ("idiot go away", 0), ("nice try i guess, genius", 1),
       ("sure, people like you always know best", 1), ("thanks for sharing this", 2),
       ("ধন্যবাদ ভাই", 2), ("good morning everyone", 2), ("what a stupid post", 0)]


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("model")
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = v.fit_transform([t for t, _ in TOY])
    clf = LogisticRegression(max_iter=500).fit(X, [y for _, y in TOY])
    joblib.dump({"vectorizers": [v], "clf": clf}, root / "tfidf.joblib")
    monkeypatch_module.setenv("MODEL_ROOT", str(root))
    monkeypatch_module.delenv("MODEL_DIR", raising=False)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True, "models": ["tfidf"]}


def test_predict_shape(client):
    r = client.post("/api/predict", json={"text": "  you   stupid idiot  "})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "you stupid idiot"
    assert body["script"] == "latin"
    assert body["primary"] == "tfidf" and body["label"] in (0, 1, 2)
    assert abs(sum(body["models"][0]["probs"]) - 1) < 1e-3
    assert [t["text"] for t in body["tokens"]] == ["you", "stupid", "idiot"]
    for t in body["tokens"]:
        assert body["text"][t["start"]:t["end"]] == t["text"]


def test_predict_bangla_script(client):
    assert client.post("/api/predict", json={"text": "ধন্যবাদ ভাই"}).json()["script"] == "bangla"


@pytest.mark.parametrize("text", ["", "   "])
def test_predict_rejects_empty(client, text):
    assert client.post("/api/predict", json={"text": text}).status_code == 422
