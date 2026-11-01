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


@lru_cache(maxsize=96)
def font_face(font_id: str) -> ImageFont.FreeTypeFont:
    record = next((f for f in catalog() if f["id"] == font_id), None)
    if record is None:
        raise ValueError("Unknown font")
    return ImageFont.truetype(str(ROOT / record["file"]), 96)


@lru_cache(maxsize=128)
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


@lru_cache(maxsize=8)
def reference_vectors(text: str):
    values = [descriptor(render_reference(font["id"], text)) for font in catalog()]
    return np.stack([value[0] for value in values]), np.array([value[1] for value in values])


def identify(image: Image.Image, supplied_text: str = "") -> dict:
    started = time.perf_counter()
    normalized = text_crop(image)
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
                "classifier": classifier, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    query, aspect = descriptor(normalized)
    vectors, aspects = reference_vectors(text)
    scores = vectors @ query - 0.08 * np.minimum(np.abs(np.log(aspects / aspect)), 1.5)
    order = np.argsort(-scores)
    best = float(scores[order[0]])
    category = catalog()[int(order[0])]["category"] if best >= 0.65 else classifier["predicted_category"]
    if best < 0.65:
        order = np.array([i for i in order if catalog()[int(i)]["category"] == category])
    order = order[:5]
    matches = []
    for index in order:
        font = catalog()[int(index)]
        matches.append({"id": font["id"], "family": font["family"], "category": font["category"],
                        "similarity": round(float(np.clip(scores[index], 0, 1)) * 100, 1),
                        "url": "https://fonts.google.com/specimen/" + quote(font["family"]),
                        "preview": f"/font-preview/{font['id']}?text=" + quote(text)})
    return {"category": category, "matches": matches, "text": text,
            "needs_text": False, "match_strength": "strong" if best >= 0.88 else "possible" if best >= 0.65 else "uncertain",
            "classifier": classifier, "catalog_size": len(catalog()),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


@lru_cache(maxsize=128)
def preview_png(font_id: str, text: str) -> bytes:
    if not text or len(text) > 80:
        raise ValueError("Invalid preview text")
    image = render_reference(font_id, text)
    image.thumbnail((900, 120), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (940, 160), "white")
    canvas.paste(image, ((940 - image.width) // 2, (160 - image.height) // 2))
    stream = BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()
