from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog


def prepare_grayscale(image: Image.Image | str | Path, size: tuple[int, int] = (224, 96)) -> np.ndarray:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def extract_hog(
    image: Image.Image | str | Path,
    size: tuple[int, int] = (224, 96),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    array = prepare_grayscale(image, size=size)
    return hog(
        array,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
