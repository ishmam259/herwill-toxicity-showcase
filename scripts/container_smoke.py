"""Pipe into the public image: docker run --rm -i IMAGE python - < this_file.
Checks both package exclusions and actual same-origin production HTTP routes.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

assert not Path("/app/private").exists(), "Private exports entered the image"
assert not Path("/app/model").exists(), "Competition weights entered the image"
assert not Path("/app/backend/model").exists(), "Competition weights entered the image"
assert not Path("/app/backend/.venv").exists(), "A local virtualenv entered the image"
assert {p.name for p in Path("/app/data").iterdir()} == {
    "models.json",
    "calibration.json",
    "examples.json",
}
env = dict(os.environ, PUBLIC_DEPLOYMENT="1", ENABLE_PRIVATE_CASES="1")
process = subprocess.Popen(
    [
        "python",
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
        "--no-proxy-headers",
    ],
    env=env,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
base = "http://127.0.0.1:7860"
try:
    for _ in range(100):
        try:
            health = json.load(urllib.request.urlopen(base + "/api/health", timeout=2))
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    else:
        raise RuntimeError("Container did not become ready")
    assert not health["private_cases"]
    assert health["models"][0]["mode"] == "illustrative"
    for path in [
        "/",
        "/compare",
        "/confusion",
        "/calibration",
        "/hard-cases",
        "/favicon.svg",
        "/api/data/models",
    ]:
        assert urllib.request.urlopen(base + path, timeout=3).status == 200, path
    for path, expected in [
        ("/api/hard-cases", 403),
        ("/private/hard_cases.json", 404),
        ("/model/sparse.joblib", 404),
        ("/api/data/oof_V0.0", 404),
    ]:
        try:
            urllib.request.urlopen(base + path, timeout=3)
        except urllib.error.HTTPError as error:
            assert error.code == expected, (path, error.code)
        else:
            raise AssertionError(f"{path} was exposed")
    request = urllib.request.Request(
        base + "/api/predict",
        data=json.dumps({"text": "Thanks for explaining your perspective."}).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = json.load(urllib.request.urlopen(request, timeout=10))
    assert result["models"][0]["label"] in [0, 1, 2]
    assert abs(sum(result["models"][0]["probs"]) - 1) < 1e-3  # API rounds to 4 decimals
    print(
        "Container verified: compiled routes, inference, private denial, public allowlist, no private data or fitted weights."
    )
finally:
    process.terminate()
    process.wait(timeout=10)
