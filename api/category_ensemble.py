"""Small learned combiner of classifier outputs and reference similarity scores."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from api.compact import CLASSES

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


def feature_vector(scores, records, classifier, normalized, text, excluded_family=None):
    """No source family/path/label enters the learner. Exclusion is training-only."""
    values = []
    for category in CLASSES:
        families = {}
        for score, record in zip(scores, records):
            if record["family"] == excluded_family or record["category"] != category:
                continue
            families[record["family"]] = max(float(score), families.get(record["family"], -1))
        ordered = sorted(families.values(), reverse=True)
        values.extend((ordered + [-1] * 3)[:3])
    probabilities = [classifier["probabilities"][category] for category in CLASSES]
    values.extend(probabilities)
    values.extend(np.log(np.clip(probabilities, 1e-6, 1)))
    ink = 255 - np.asarray(normalized.convert("L"), dtype=np.uint8)
    columns = np.any(ink > 40, axis=0)
    runs = int(np.count_nonzero(columns & np.concatenate(([True], ~columns[:-1]))))
    values.extend([float((ink > 40).mean()), runs / max(len(text.replace(" ", "")), 1),
                   min(normalized.width / normalized.height, 25) / 25])
    return np.asarray(values, dtype=np.float32)


@lru_cache(maxsize=1)
def model():
    contract = json.loads((ROOT / "category_ensemble.json").read_text())
    path = ROOT / "category_ensemble.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"] or contract["classes"] != list(CLASSES):
        raise RuntimeError("Category combiner does not match its verified export")
    semantic = hashlib.sha256(json.dumps(json.loads((ROOT / "font_catalog.json").read_text()), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if semantic != contract["catalog_semantic_sha256"]:
        raise RuntimeError("Font catalog does not match the category model")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def predict(features):
    parameters = model()
    standardized = (features - parameters["mean"]) / parameters["scale"]
    logits = parameters["coefs"] @ standardized + parameters["intercept"]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return CLASSES[int(probabilities.argmax())], float(probabilities.max())
