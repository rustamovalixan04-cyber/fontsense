"""Conservative font-name retrieval from whole-word image reference statistics."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from api.compact import CHECKPOINT_SHA256, get_model
from api.name_encoder import encoder, forward_features, patches

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


def geometry(image, text):
    from PIL import ImageOps
    from api.compact import text_crop
    gray = text_crop(image).convert("L")
    bounds = ImageOps.invert(gray).point(lambda value: 255 if value > 40 else 0).getbbox()
    gray = gray.crop(bounds)
    pixels = np.asarray(gray, dtype=np.float32) / 255
    return np.asarray([gray.width / gray.height / max(len(text.replace(" ", "")), 1),
                       1 - pixels.mean(), (pixels < .5).mean(), (pixels < .1).mean()], dtype=np.float32)


@lru_cache(maxsize=1)
def gallery():
    metadata = json.loads((ROOT / "font_name_gallery.json").read_text())
    path = ROOT / "font_name_gallery.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise RuntimeError("Font-name gallery does not match its export")
    encoder_metadata = json.loads((ROOT / "font_name_encoder.json").read_text())
    if metadata["encoder_sha256"] != encoder_metadata["sha256"]:
        raise RuntimeError("Font-name gallery belongs to another encoder")
    if metadata["legacy_checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("Font-name gallery belongs to another base model")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    return metadata, arrays


def case_id(text):
    letters = [char for char in text if char.isalpha()]
    upper = sum(char.isupper() for char in letters)
    return 2 if upper >= max(1, len(letters) * .8) else 0 if upper == 0 else 1


def reference_scores(new_vectors, old_vectors, cases, arrays, weight, case_penalty):
    scores = weight * (new_vectors @ arrays["new_vectors"].T)
    if weight < 1:
        scores += (1 - weight) * (old_vectors @ arrays["old_vectors"].T)
    scores -= case_penalty * (np.asarray(cases)[:, None] != arrays["cases"][None, :])
    return np.stack([scores[:, arrays["families"] == index].max(axis=1) for index in range(int(arrays["families"].max()) + 1)], axis=1)


def candidates(image, text, limit=5):
    if len([char for char in text if char.isalpha()]) < 4 or len(set(text.casefold()) - {" "}) < 3:
        return []
    metadata, arrays = gallery()
    new_vector = encoder().encode(image)
    old_vector = None
    if metadata["neural_weight"] < 1:
        legacy = np.mean([forward_features(patch, get_model().blocks) for patch in patches(image)], axis=0)
        raw = np.concatenate((np.log1p(legacy), geometry(image, text)))
        old_vector = (raw - arrays["mean"]) / arrays["scale"]
        old_vector /= max(float(np.linalg.norm(old_vector)), 1e-6)
        old_vector = old_vector[None]
    scores = reference_scores(new_vector[None], old_vector, [case_id(text)], arrays,
                              metadata["neural_weight"], metadata["case_penalty"])[0]
    order = np.argsort(-scores)
    margin = float(scores[order[0]] - scores[order[1]])
    result = []
    for index in order[:limit]:
        font = metadata["families"][int(index)]
        result.append({**font, "score": float(scores[index]), "margin": margin,
                       "strength": "candidate", "method": "whole-word-reference-model"})
    return result


def recognize(image, text, ranked=None):
    if ranked is None:
        ranked = candidates(image, text, limit=2)
    if not ranked:
        return None
    metadata, _ = gallery()
    if ranked[0]["score"] < metadata["minimum_similarity"] or ranked[0]["margin"] < metadata["minimum_margin"]:
        return None
    font = ranked[0]
    return {"family": font["family"], "category": font["category"], "url": font["url"],
            "strength": "likely", "method": "whole-word-reference-model"}
