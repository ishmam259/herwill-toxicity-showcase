"""Validate the public allowlist, schema and absence of row-level fields."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"models.json", "calibration.json", "examples.json"}
FORBIDDEN = {"text", "idx", "fold", "probs", "posts", "cases", "train_ids", "test_ids"}


def scan(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in FORBIDDEN:
                raise ValueError(f"Row-level field {path}/{k} in public aggregates")
            scan(v, f"{path}/{k}")
    elif isinstance(value, list):
        for v in value:
            scan(v, path)


def check():
    actual = {p.name for p in (ROOT / "data").iterdir() if p.is_file()}
    if actual != ALLOWED:
        raise ValueError(f"Public data allowlist mismatch: {actual}")
    for name in ("models.json", "calibration.json"):
        value = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
        scan(value)
        if value["raw_text_included"] is not False or value["class_order"] != [0, 1, 2]:
            raise ValueError("Invalid public contract")
    models = json.loads((ROOT / "data/models.json").read_text(encoding="utf-8"))["models"]
    calibration = json.loads((ROOT / "data/calibration.json").read_text(encoding="utf-8"))
    for m in models:
        if sum(m["support"]) != m["rows"]:
            raise ValueError("Support mismatch")
        if sum(sum(row) for row in m["confusion"]) != m["rows"]:
            raise ValueError("Confusion mismatch")
        grid = calibration["models"][m["id"]]["macro_f1_grid"]
        if len(grid) != 21 or any(len(row) != 21 for row in grid):
            raise ValueError("Offset grid mismatch")
        if abs(grid[10][10] - m["macro_f1"]) > 1e-8:
            raise ValueError("Zero-offset score mismatch")
    ex = json.loads((ROOT / "data/examples.json").read_text(encoding="utf-8"))
    if "hand-written" not in ex["provenance"]:
        raise ValueError("Missing example provenance")
    if len({e["id"] for e in ex["examples"]}) != 18:
        raise ValueError("Expected 18 unique authored examples")
    print(
        "Public data verified: aggregate-only results and 18 attributed authored examples."
    )


if __name__ == "__main__":
    try:
        check()
    except (ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
