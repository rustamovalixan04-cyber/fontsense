"""Text-specific rendered comparisons across the extended open-font catalog.

No network calls, font downloads, font-name generation or new learned classifier
are needed during requests. Coarse vectors shortlist real styles for full-size
comparison. These similarity scores are not probabilities.
"""
from functools import lru_cache
from pathlib import Path
import json
import os
from threading import RLock

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"
_LOCK = RLock()


@lru_cache(maxsize=1)
def catalog():
    fonts = json.loads((ROOT / "extended_catalog.json").read_text(encoding="utf-8"))["fonts"]
    independent = ROOT / "independent_catalog.json"
    if independent.exists():
        fonts += json.loads(independent.read_text(encoding="utf-8"))["fonts"]
    if len({row['id'] for row in fonts}) != len(fonts):
        raise ValueError('Duplicate font identifier in reference catalogs')
    return tuple(fonts)


@lru_cache(maxsize=512)
def font_face(font_id, size=64):
    record = records_by_id().get(font_id)
    if record is None:
        raise ValueError("Unknown font")
    font = ImageFont.truetype(str(ROOT / record["file"]), size)
    if record["variable_weight"]:
        font.set_variation_by_axes([record["weight"] if axis["name"] == b"Weight" else axis["default"] for axis in font.get_variation_axes()])
    return font


@lru_cache(maxsize=1)
def records_by_id():
    return {font["id"]: font for font in catalog()}


@lru_cache(maxsize=1)
def primary_styles():
    families = {}
    for font in catalog():
        key = (font["italic"], abs(font["weight"] - 400))
        previous = families.get(font["family"])
        if previous is None or key < (previous["italic"], abs(previous["weight"] - 400)):
            families[font["family"]] = font
    return tuple(families.values())


def render_reference(font_id, text, size=64):
    with _LOCK:
        return _render_reference(font_id, text, size)


def _render_reference(font_id, text, size=64):
    font = font_face(font_id, size)
    box = font.getbbox(text)
    width, height = box[2] - box[0], box[3] - box[1]
    if width < 1 or height < 1 or width > 20000 or height > 512:
        raise ValueError("No visible reference text")
    image = Image.new("L", (width + 12, height + 12), 255)
    ImageDraw.Draw(image).text((6 - box[0], 6 - box[1]), text, font=font, fill=0)
    return image


def descriptor(image, width=96, height=24):
    gray = image.convert("L")
    bounds = ImageOps.invert(gray).point(lambda value: 255 if value > 40 else 0).getbbox()
    if bounds is None:
        raise ValueError("No visible text")
    gray = gray.crop(bounds)
    ink = 255 - np.asarray(gray.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)
    return ink.reshape(-1), gray.width / gray.height


@lru_cache(maxsize=2)
def coarse_references(text):
    vectors, aspects = [], []
    for font in catalog():
        try:
            vector, aspect = descriptor(render_reference(font["id"], text, 48))
        except (ValueError, OSError):
            # A font may have Latin letters but no digits/punctuation used in
            # this query. It must not break matching for the remaining catalog.
            vector, aspect = np.zeros(96 * 24, dtype=np.uint8), 1.
        vectors.append(vector); aspects.append(aspect)
    vectors = np.stack(vectors)
    norms = np.sqrt(np.square(vectors.astype(np.float32)).sum(axis=1))
    return vectors, np.maximum(norms, 1e-6), np.asarray(aspects, dtype=np.float32)


def exhaustive_rank(image, text, limit=8):
    # Lock the font engine and expensive cache fill, bounding duplicate work
    # when several requests arrive together on one warm worker.
    with _LOCK:
        query, aspect = descriptor(image)
        vector = query.astype(np.float32)
        vector /= max(float(np.linalg.norm(vector)), 1e-6)
        matrix, norms, aspects = coarse_references(text)
        scores = np.empty(len(matrix), dtype=np.float32)
        # Keep temporary float copies bounded, rather than caching hundreds MB.
        for start in range(0, len(matrix), 256):
            stop = min(start + 256, len(matrix))
            scores[start:stop] = matrix[start:stop].astype(np.float32) @ vector / norms[start:stop]
        scores -= .08 * np.minimum(np.abs(np.log(aspects / aspect)), 1.5)
        shortlist = np.argsort(-scores)[:140]
        # Use the same higher-resolution full/spacing comparison as the earlier
        # product. Imports here avoid a module-import cycle.
        from api.font_match import descriptor as full_descriptor, spacing_descriptor
        full_query, full_aspect = full_descriptor(image)
        compact_query, compact_aspect = spacing_descriptor(image)
        found = {}
        for index in shortlist:
            record = catalog()[int(index)]
            try:
                rendered = render_reference(record["id"], text, 96)
                reference, reference_aspect = full_descriptor(rendered)
                compact, aspect_compact = spacing_descriptor(rendered)
            except (ValueError, OSError):
                continue
            full = float(reference @ full_query) - .08 * min(abs(np.log(reference_aspect / full_aspect)), 1.5)
            spaced = float(compact @ compact_query) - .08 * min(abs(np.log(aspect_compact / compact_aspect)), 1.5)
            score = max(full, .5 * (full + spaced) - .04)
            if record["family"] not in found or score > found[record["family"]]["score"]:
                found[record["family"]] = {**record, "score": float(score)}
        return sorted(found.values(), key=lambda record: -record["score"])[:limit]


def learned_enabled():
    choice = os.environ.get("FONTSENSE_RETRIEVAL_ENGINE", "auto")
    if choice == "legacy": return False
    if choice == "candidate": return True
    policy = ROOT / "retrieval_policy.json"
    return policy.exists() and json.loads(policy.read_text()).get("enabled", False)


def rank(image, text, limit=8):
    native = None
    if learned_enabled():
        from api.retrieval_match import rank as learned_rank
        result = learned_rank(image, text, limit)
        if result and result[0]["pixel_score"] >= .8:
            return result
        if result and result[0].get('appearance_candidates'):
            return result[0]['appearance_candidates'][:limit]
        if result: native = result[0].get("native_candidates", [])
    fallback = exhaustive_rank(image, text, limit)
    if native is not None:
        for row in fallback:
            row["native_candidates"] = native
            row["method"] = "exhaustive-with-trained-native-references"
    return fallback


def likely_match(ranked, text):
    if ranked and ranked[0].get('method') == 'trained-appearance-alternatives':
        return None
    if ranked and ranked[0].get("method") == "trained-shortlist-and-glyphs":
        from api.retrieval_match import likely_match as learned_name
        return learned_name(ranked, text)
    # Conservative direct-evidence gate; close, compatible fonts stay candidates.
    # This is a retrieval decision, never an exact-identity guarantee.
    if len(ranked) < 2 or sum(char.isalpha() for char in text) < 5 or len(set(text.casefold())) < 4:
        return None
    best = ranked[0]
    if best["score"] < .96 or best["score"] - ranked[1]["score"] < .025:
        return None
    return {"family": best["family"], "category": best["category"], "url": best["url"],
            "style": best["style"], "strength": "likely", "method": "same-text-rendered-reference"}
