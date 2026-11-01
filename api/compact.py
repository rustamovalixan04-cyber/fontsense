"""Small NumPy runtime for the unchanged, frozen FontSense CNN weights."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("display", "handwriting", "monospace", "sans_serif", "serif")
CHECKPOINT_SHA256 = "c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f"
FINAL_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
FINAL_CLASS_ORDER = CLASSES
FINAL_MODEL_LABEL = "Final CNN"
FROZEN_THRESHOLD = 0.60


def text_crop(image: Image.Image) -> Image.Image:
    """Normalize polarity and crop excess background without losing thin strokes."""
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    rgb = Image.new("RGBA", rgba.size, "white")
    rgb.alpha_composite(rgba)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    contrast = np.abs(gray - background)
    peak = float(np.percentile(contrast, 99.5))
    if float(gray.std()) < 2 or peak < 8:
        raise ValueError("No readable text found. Try a closer crop.")
    mask = contrast > max(8, peak * 0.16)
    rows, cols = np.where(mask)
    if rows.size < 20:
        raise ValueError("No readable text found. Try a closer crop.")
    normalized = np.clip(255 - contrast * (255 / max(peak, 1)), 0, 255).astype(np.uint8)
    crop = Image.fromarray(normalized).crop((int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))
    padding = max(3, round(crop.height * 0.09))
    return ImageOps.expand(crop, padding, "white").convert("RGB")


class CompactCNN:
    def __init__(self):
        path = ROOT / "artifacts/web/cnn_v1.npz"
        contract = json.loads((path.parent / "web_runtime.json").read_text())
        if contract["source_checkpoint_sha256"] != CHECKPOINT_SHA256 or hashlib.sha256(path.read_bytes()).hexdigest() != contract["weights_sha256"]:
            raise RuntimeError("The compact model does not match its verified export.")
        with np.load(path, allow_pickle=False) as weights:
            self.blocks = []
            for index in (0, 4, 8, 12):
                stem = f"features.{index}"
                bn = f"features.{index + 1}"
                scale = weights[f"{bn}.weight"] / np.sqrt(weights[f"{bn}.running_var"] + np.float32(1e-5))
                kernel = weights[f"{stem}.weight"] * scale[:, None, None, None]
                bias = (weights[f"{stem}.bias"] - weights[f"{bn}.running_mean"]) * scale + weights[f"{bn}.bias"]
                self.blocks.append((kernel, bias))
            self.linear = weights["classifier.2.weight"].copy()
            self.bias = weights["classifier.2.bias"].copy()

    def predict(self, image: Image.Image) -> dict:
        started = time.perf_counter()
        if image.width < 20 or image.height < 20:
            raise ValueError("Use a text crop of at least 20 × 20 pixels.")
        gray = image.convert("L").resize((112, 48), Image.Resampling.BILINEAR)
        if float(np.asarray(gray).std()) < 2:
            raise ValueError("No readable text found. Try a closer crop.")
        x = (np.asarray(gray, dtype=np.float32)[None] / 255 - 0.5) / 0.5
        for index, (kernel, bias) in enumerate(self.blocks):
            padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
            windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
            x = np.einsum("ocij,chwij->ohw", kernel, windows, optimize=True) + bias[:, None, None]
            np.maximum(x, 0, out=x)
            if index < 3:
                channels, height, width = x.shape
                x = x.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))
        vector = x.mean(axis=(1, 2))
        logits = self.linear @ vector + self.bias
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(probabilities.argmax())
        return {
            "predicted_category": CLASSES[best],
            "confidence": float(probabilities[best]),
            "probabilities": dict(zip(CLASSES, map(float, probabilities))),
            "accepted": bool(probabilities[best] >= 0.6),
            "inference_ms": (time.perf_counter() - started) * 1000,
        }


@lru_cache(maxsize=1)
def get_model() -> CompactCNN:
    return CompactCNN()


def get_predictor(model_name: str) -> CompactCNN:
    if model_name != FINAL_MODEL_LABEL:
        raise ValueError("Choose the final CNN.")
    return get_model()
