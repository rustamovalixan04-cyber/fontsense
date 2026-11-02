"""Compare separated glyph shapes when both lines admit a reliable alignment."""
import numpy as np
from PIL import Image, ImageOps


def glyphs(image):
    gray = image.convert("L")
    box = ImageOps.invert(gray).point(lambda x: 255 if x > 40 else 0).getbbox()
    if box is None: return None
    gray = gray.crop(box)
    ink = 255 - np.asarray(gray, dtype=np.uint8)
    occupied = np.any(ink > 40, axis=0)
    starts = np.flatnonzero(occupied & np.r_[True, ~occupied[:-1]])
    ends = np.flatnonzero(occupied & np.r_[~occupied[1:], True]) + 1
    if not 4 <= len(starts) <= 80: return None
    vectors, widths = [], []
    for start, end in zip(starts, ends):
        if end - start < 2: return None
        patch = Image.fromarray(ink[:, start:end]).resize((24, 48), Image.Resampling.BILINEAR)
        vector = np.asarray(patch, dtype=np.float32).reshape(-1)
        vector /= max(float(np.linalg.norm(vector)), 1e-6)
        vectors.append(vector); widths.append((end - start) / gray.height)
    return np.stack(vectors), np.asarray(widths)


def agreement(query, reference):
    if query is None or reference is None or len(query[0]) != len(reference[0]):
        return None
    similarities = (query[0] * reference[0]).sum(axis=1)
    # Individual glyph scaling removes kerning/letter-spacing differences; keep
    # relative letter widths so condensed designs still differ geometrically.
    ratios = np.log(reference[1] / query[1]); ratios -= np.median(ratios)
    return float(similarities.mean() - .08 * np.minimum(np.abs(ratios), 1).mean())
