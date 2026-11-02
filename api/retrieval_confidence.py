"""Empirical name-acceptance filter, never a displayed probability guarantee."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


@lru_cache(maxsize=1)
def parameters():
    result = json.loads((ROOT / "retrieval_confidence.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "retrieval_index.json").read_text(encoding="utf-8"))
    if result["index_sha256"] != index["sha256"]:
        raise RuntimeError("Confidence model belongs to another retrieval index")
    return result


def evidence_score(features):
    model = parameters()
    normalized = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logit = float(normalized @ np.asarray(model["coefs"]) + model["intercept"])
    return float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))


def accept(features):
    model = parameters()
    # Keep a direct-shape floor in addition to the fitted correctness model.
    return features[1] >= model["minimum_pixel_score"] and evidence_score(features) >= model["threshold"]
