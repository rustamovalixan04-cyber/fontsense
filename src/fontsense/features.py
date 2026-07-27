from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skimage.feature import hog
from tqdm import tqdm


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


class HOGTransformer(TransformerMixin, BaseEstimator):
    """Deterministically convert image paths or Pillow images into HOG vectors."""

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 96),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        show_progress: bool = False,
    ):
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.show_progress = show_progress

    def fit(self, x, y=None):
        self.fit_sample_count_ = len(x)
        self.is_fitted_ = True
        return self

    def transform(self, x) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        array = np.asarray(x)
        if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
            return array.astype(np.float32, copy=False)

        items = list(x)
        if not items:
            raise ValueError("HOGTransformer received zero images")
        iterator: Iterable = items
        if self.show_progress:
            iterator = tqdm(items, desc="Extracting HOG", leave=False)
        features = [
            extract_hog(
                image,
                size=tuple(self.image_size),
                orientations=int(self.orientations),
                pixels_per_cell=tuple(self.pixels_per_cell),
                cells_per_block=tuple(self.cells_per_block),
            )
            for image in iterator
        ]
        return np.stack(features)
