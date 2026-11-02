"""Image-only font-style features; deployed inference uses NumPy."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from api.compact import text_crop

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


def patches(image: Image.Image, count=3):
    gray = text_crop(image).convert("L")
    bounds = ImageOps.invert(gray).point(lambda value: 255 if value > 40 else 0).getbbox()
    if bounds is None:
        raise ValueError("No visible text")
    gray = gray.crop(bounds)
    width = max(1, round(gray.width * 32 / gray.height))
    gray = gray.resize((width, 32), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (max(width + 16, 112), 48), 255)
    canvas.paste(gray, ((canvas.width - width) // 2, 8))
    starts = np.linspace(0, canvas.width - 112, count).round().astype(int)
    return np.stack([np.asarray(canvas.crop((int(x), 0, int(x) + 112, 48))) for x in starts])


def forward_features(pixels, blocks):
    x = (np.asarray(pixels, dtype=np.float32)[None] / 255 - .5) / .5
    for index, (kernel, bias) in enumerate(blocks):
        padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        x = np.einsum("ocij,chwij->ohw", kernel, windows, optimize=True) + bias[:, None, None]
        np.maximum(x, 0, out=x)
        if index < 3:
            channels, height, width = x.shape
            x = x.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))
    return x.mean(axis=(1, 2))


class NameEncoder:
    def __init__(self):
        contract = json.loads((ROOT / "font_name_encoder.json").read_text())
        path = ROOT / "font_name_encoder.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"]:
            raise RuntimeError("Font-name encoder does not match its export")
        self.blocks = []
        with np.load(path, allow_pickle=False) as weights:
            for index in (0, 4, 8, 12):
                stem, bn = f"features.{index}", f"features.{index + 1}"
                scale = weights[bn + ".weight"] / np.sqrt(weights[bn + ".running_var"] + 1e-5)
                kernel = weights[stem + ".weight"] * scale[:, None, None, None]
                bias = (weights[stem + ".bias"] - weights[bn + ".running_mean"]) * scale + weights[bn + ".bias"]
                self.blocks.append((kernel, bias))
            self.projection = weights["projection.weight"].copy()

    def encode(self, image):
        values = []
        for patch in patches(image):
            vector = self.projection @ forward_features(patch, self.blocks)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            values.append(vector)
        vector = np.mean(values, axis=0)
        return vector / max(float(np.linalg.norm(vector)), 1e-6)


@lru_cache(maxsize=1)
def encoder():
    return NameEncoder()
