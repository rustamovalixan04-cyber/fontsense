"""OCR-assisted visual retrieval from a licensed reference font catalog.

Font names come from actual rendered references, not generated guesses.
Similarity measures are retrieval scores, not calibrated probabilities.
"""
from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import json
import math
from pathlib import Path
import time
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from api.compact import get_model, text_crop
from api.category_ensemble import feature_vector, predict as predict_category
from api.font_identity import recognize as recognize_font_name
from api.font_identity import candidates as name_candidates
from api import extended_match

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


@lru_cache(maxsize=1)
def catalog() -> tuple[dict, ...]:
    return tuple(json.loads((ROOT / "font_catalog.json").read_text(encoding="utf-8"))["fonts"])


@lru_cache(maxsize=1)
def recognizer():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / "text_recognition.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    characters = session.get_modelmeta().custom_metadata_map["character"].rstrip("\n").split("\n")
    count = session.get_outputs()[0].shape[-1]
    if len(characters) == count - 2:
        characters.append(" ")
    assert len(characters) == count - 1, "OCR character dictionary does not match the model"
    return session, [""] + characters


def read_text(image: Image.Image) -> tuple[str, float]:
    session, characters = recognizer()
    height = 48
    width = min(1536, max(320, math.ceil(height * image.width / image.height)))
    resized_width = min(width, math.ceil(height * image.width / image.height))
    resized = image.resize((resized_width, height), Image.Resampling.BILINEAR).convert("RGB")
    pixels = np.asarray(resized, dtype=np.float32) / 127.5 - 1
    tensor = np.zeros((1, 3, height, width), dtype=np.float32)
    tensor[0, :, :, :resized_width] = pixels.transpose(2, 0, 1)
    predictions = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    indices = predictions.argmax(axis=1)
    selected = (indices != 0) & np.concatenate(([True], indices[1:] != indices[:-1]))
    text = "".join(characters[int(i)] for i in indices[selected]).strip()
    confidence = float(predictions.max(axis=1)[selected].mean()) if selected.any() else 0
    return text[:80], confidence


@lru_cache(maxsize=256)
def font_face(font_id: str) -> ImageFont.FreeTypeFont:
    record = next((f for f in catalog() if f["id"] == font_id), None)
    if record is None:
        return extended_match.font_face(font_id, 96)
    font = ImageFont.truetype(str(ROOT / record["file"]), 96)
    if record.get("variable_weight"):
        axes = font.get_variation_axes()
        font.set_variation_by_axes([record["weight"] if axis["name"] == b"Weight" else axis["default"] for axis in axes])
    return font


@lru_cache(maxsize=32)
def render_reference(font_id: str, text: str) -> Image.Image:
    font = font_face(font_id)
    box = font.getbbox(text)
    width, height = box[2] - box[0], box[3] - box[1]
    if width <= 0 or height <= 0:
        raise ValueError("Enter some visible text.")
    image = Image.new("RGB", (width + 12, height + 12), "white")
    ImageDraw.Draw(image).text((6 - box[0], 6 - box[1]), text, font=font, fill="black")
    return image


def descriptor(image: Image.Image) -> tuple[np.ndarray, float]:
    gray = image.convert("L")
    bounds = ImageOps.invert(gray).point(lambda value: 255 if value > 40 else 0).getbbox()
    if not bounds:
        raise ValueError("No visible text found.")
    gray = gray.crop(bounds)
    aspect = gray.width / gray.height
    vector = (1 - np.asarray(gray.resize((384, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255).reshape(-1)
    vector /= max(float(np.linalg.norm(vector)), 1e-6)
    return vector, aspect


def spacing_descriptor(image: Image.Image) -> tuple[np.ndarray, float]:
    """Keep letter shapes while removing empty columns, including missed spaces."""
    ink = 255 - np.asarray(image.convert("L"), dtype=np.uint8)
    rows = np.any(ink > 40, axis=1)
    columns = np.any(ink > 40, axis=0)
    compact = ink[rows][:, columns]
    if not compact.size:
        raise ValueError("No visible text found.")
    aspect = compact.shape[1] / compact.shape[0]
    vector = np.asarray(Image.fromarray(compact).resize((192, 48), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1)
    vector /= max(float(np.linalg.norm(vector)), 1e-6)
    return vector, aspect


@lru_cache(maxsize=4)
def reference_vectors(text: str):
    values, compact = [], []
    for font in catalog():
        image = render_reference(font["id"], text)
        values.append(descriptor(image))
        compact.append(spacing_descriptor(image))
    return (np.stack([value[0] for value in values]), np.array([value[1] for value in values]),
            np.stack([value[0] for value in compact]), np.array([value[1] for value in compact]))


def rank_families(scores: np.ndarray) -> list[int]:
    """Return each family once, keeping its closest real weight/italic style."""
    seen = set()
    order = []
    for index in np.argsort(-scores):
        family = catalog()[int(index)]["family"]
        if family not in seen:
            seen.add(family)
            order.append(int(index))
    return order


def straighten_line(image: Image.Image) -> Image.Image:
    """Correct small screenshot rotation only when row alignment clearly improves."""
    small = image.convert("L")
    small.thumbnail((600, 150), Image.Resampling.BILINEAR)
    padded = ImageOps.expand(small, 20, "white")

    def energy(angle):
        rotated = padded.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=255)
        ink = 1 - np.asarray(rotated, dtype=np.float32) / 255
        return float(np.square(np.diff(ink.sum(axis=1))).sum())

    original = energy(0)
    angles = np.arange(-4, 4.01, .5)
    values = [energy(float(angle)) for angle in angles]
    index = int(np.argmax(values))
    angle = float(angles[index])
    if abs(angle) < .5 or values[index] < original * 1.12:
        return image
    return text_crop(image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor="white"))


def alignment_scores(normalized: Image.Image, text: str) -> np.ndarray:
    query, aspect = descriptor(normalized)
    compact_query, compact_aspect = spacing_descriptor(normalized)
    vectors, aspects, compact_vectors, compact_aspects = reference_vectors(text)
    full = vectors @ query - .08 * np.minimum(np.abs(np.log(aspects / aspect)), 1.5)
    compact = compact_vectors @ compact_query - .08 * np.minimum(np.abs(np.log(compact_aspects / compact_aspect)), 1.5)
    return np.maximum(full, .5 * (full + compact) - .04)


def identify(image: Image.Image, supplied_text: str = "") -> dict:
    started = time.perf_counter()
    normalized = straighten_line(text_crop(image))
    supplied_text = " ".join(supplied_text.split())
    if len(supplied_text) > 80 or any(ord(c) < 32 or ord(c) > 126 for c in supplied_text):
        raise ValueError("Use up to 80 Latin letters, numbers or punctuation.")
    if supplied_text:
        text, text_confidence = supplied_text, None
    else:
        text, text_confidence = read_text(normalized)
    classifier = get_model().predict(image)
    if len(text.strip()) < 2 or (text_confidence is not None and text_confidence < 0.55):
        return {"category": classifier["predicted_category"], "matches": [],
                "text": text, "needs_text": True, "match_strength": "uncertain",
                "identified_font": None,
                "classifier": classifier, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    # Spacing invariance is an alternate alignment, never a penalty on a good
    # full-line match. Its small cost favors an equally good original layout.
    scores = alignment_scores(normalized, text)
    order = rank_families(scores)
    best = float(scores[order[0]])
    category, category_score = predict_category(feature_vector(scores, catalog(), classifier, normalized, text))
    # An exceptionally close rendered reference is stronger direct evidence
    # than an ensemble trained with its own source-family references excluded.
    other = max(float(scores[i]) for i in order if catalog()[i]["category"] != catalog()[order[0]]["category"])
    if best >= .90 and best - other >= .06:
        category = catalog()[order[0]]["category"]
    # Keep the category combiner's original reference distribution unchanged.
    # The larger library is a separate, text-specific retrieval stage.
    extended = extended_match.rank(normalized, text)
    broad_name = extended_match.likely_match(extended, text)
    embedding_candidates = name_candidates(image, text)
    identified_font = recognize_font_name(image, text, ranked=embedding_candidates)
    # The older encoder was calibrated against just 86 names. It must not
    # overrule the larger open-font comparison with a familiar OFL lookalike.
    # Native references remain useful where no font program can be deployed.
    if identified_font and not embedding_candidates[0].get("native"):
        identified_font = None
    trained_native = None
    if extended and "native_candidates" in extended[0]:
        from api.retrieval_match import likely_native
        fresh_native = extended[0].get("native_candidates", [])
        if identified_font and (not fresh_native or fresh_native[0]["family"] != identified_font["family"] or fresh_native[0]["embedding_similarity"] < .9):
            identified_font = None
        trained_native = likely_native(extended, text)
        if trained_native is not None:
            identified_font = trained_native
    if broad_name is not None:
        if identified_font is not None and identified_font["family"] != broad_name["family"]:
            identified_font = None
        else:
            identified_font = broad_name
    # Known compatible alternatives are not evidence of which vendor's font
    # produced the pixels. Keep both names possible, even on a very close match.
    compatible = {"Tinos": "Times New Roman", "Arimo": "Arial", "Cousine": "Courier New", "Carlito": "Calibri"}
    related = compatible.get(extended[0]["family"]) if extended else None
    native_neighbor = next((font for font in embedding_candidates if font["family"] == related), None)
    if related and native_neighbor is None:
        native_neighbor = {"family": related, "category": extended[0]["category"],
                           "url": "https://learn.microsoft.com/en-us/typography/font-list/" + related.lower().replace(" ", "-"),
                           "basis": "compatible-design-alternative"}
    if native_neighbor and identified_font and identified_font["family"] in (extended[0]["family"], related):
        identified_font = None
    if identified_font is not None:
        category = identified_font["category"]
    elif extended and extended[0]["score"] >= .90:
        category = extended[0]["category"]
    # Keep the first suggestion consistent with the chosen broad category.
    order = [i for i in order if catalog()[i]["category"] == category] + [i for i in order if catalog()[i]["category"] != category]
    if best < .65:
        order = [i for i in order if catalog()[i]["category"] == category]
    if identified_font is not None:
        order = [i for i in order if catalog()[i]["family"] != identified_font["family"]]
    order = order[:5]
    selected_best = float(scores[order[0]])
    decorative = {"display", "handwriting"}
    ambiguous = (selected_best < .88 and
                 ((category in decorative and classifier["predicted_category"] in decorative and category != classifier["predicted_category"])
                  or category_score < .55))
    matches = []
    for index in order:
        font = catalog()[int(index)]
        matches.append({"id": font["id"], "family": font["family"], "category": font["category"], "style": font.get("style", "Regular"),
                        "similarity": round(float(np.clip(scores[index], 0, 1)) * 100, 1),
                        "url": "https://fonts.google.com/specimen/" + quote(font["family"]),
                        "preview": f"/font-preview/{font['id']}?text=" + quote(text)})
    broader_matches = [{"id": font["id"], "family": font["family"], "category": font["category"],
                        "style": font["style"], "similarity": round(float(np.clip(font["score"], 0, 1)) * 100, 1),
                        "url": font["url"], "preview": f"/font-preview/{font['id']}?text=" + quote(text) +
                        "".join("&" + key + "=" + str(value) for key, value in font.get("fit_axes", {}).items())}
                       for font in extended if not identified_font or font["family"] != identified_font["family"]]
    broader_matches.sort(key=lambda font: font["category"] != category)
    if broader_matches:
        matches = broader_matches[:5]
        selected_best = matches[0]["similarity"] / 100
        if selected_best >= .90:
            ambiguous = False
    possible = []
    if not identified_font and extended:
        native_options = []
        if "native_candidates" in extended[0]:
            from api.retrieval_confidence import parameters
            soft = parameters().get("native_candidate", {"minimum_similarity": .9, "maximum_family_rank": 10})
            native_options = [font for font in extended[0]["native_candidates"][:2]
                              if font["embedding_similarity"] >= soft["minimum_similarity"] and font["embedding_family_rank"] <= soft["maximum_family_rank"]]
            consensus = bool(native_options and any(font["family"] == native_options[0]["family"] and font.get("native") and font["score"] >= .86 for font in embedding_candidates))
            if native_options and (native_options[0]["embedding_family_rank"] == 1 or consensus):
                possible.append({key: native_options[0][key] for key in ("family", "category", "url")})
            if not possible and extended[0]["score"] >= .65:
                possible.append({key: extended[0][key] for key in ("family", "category", "url")})
            for font in native_options:
                if font["family"] not in {row["family"] for row in possible}:
                    possible.append({key: font[key] for key in ("family", "category", "url")})
        if native_neighbor:
            if native_neighbor["family"] not in {row["family"] for row in possible}:
                possible.append({key: native_neighbor[key] for key in ("family", "category", "url")})
        for font in embedding_candidates:
            if native_options: continue
            if font.get("native") and font["category"] == category and font["score"] >= max(.86, embedding_candidates[0]["score"] - .04):
                if font["family"] not in {row["family"] for row in possible}:
                    possible.append({key: font[key] for key in ("family", "category", "url")})
        for font in [font for font in extended if font["category"] == category][:3]:
            if font["score"] >= .65:
                if font["family"] not in {row["family"] for row in possible}:
                    possible.append({key: font[key] for key in ("family", "category", "url")})
        possible = possible[:5]
    return {"category": category, "matches": matches, "text": text,
            "identified_font": identified_font, "font_candidates": possible,
            "font_name_model": "trained-retrieval-1" if extended_match.learned_enabled() else "extended-rendered-reference-1",
            "needs_text": False, "match_strength": "uncertain" if ambiguous else "strong" if selected_best >= 0.88 else "possible" if selected_best >= 0.65 else "uncertain",
            "classifier": classifier, "catalog_size": len({font["family"] for font in extended_match.catalog()}),
            "category_model": "similarity-ensemble-1",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


@lru_cache(maxsize=128)
def preview_png(font_id: str, text: str, weight=None, width=None, optical_size=None) -> bytes:
    if not text or len(text) > 80:
        raise ValueError("Invalid preview text")
    if any(value is not None for value in (weight, width, optical_size)):
        from api.font_variations import render_variation
        image = render_variation(font_id, text, weight, width, optical_size).convert("RGB")
    else:
        image = render_reference(font_id, text).copy()
    image.thumbnail((900, 120), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (940, 160), "white")
    canvas.paste(image, ((940 - image.width) // 2, (160 - image.height) // 2))
    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()
