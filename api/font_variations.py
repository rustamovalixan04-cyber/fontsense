"""Bounded variable-font fitting without mutating shared reference font faces."""
from functools import lru_cache
import math

from PIL import Image, ImageDraw, ImageFont

from api.extended_match import ROOT, records_by_id, _LOCK


@lru_cache(maxsize=128)
def variation_face(font_id, weight=None, width=None, optical_size=None):
    record = records_by_id().get(font_id)
    if record is None: raise ValueError("Unknown font")
    if any(value is not None and not math.isfinite(value) for value in (weight, width, optical_size)):
        raise ValueError("Invalid font variation")
    font = ImageFont.truetype(str(ROOT / record["file"]), 96)
    try: axes = font.get_variation_axes()
    except OSError: raise ValueError("This reference is not a variable font") from None
    selected = {b"Weight": weight if weight is not None else record["weight"], b"Width": width, b"Optical Size": optical_size}
    values = []
    for axis in axes:
        value = selected.get(axis["name"])
        value = axis["default"] if value is None else value
        if not axis["minimum"] <= value <= axis["maximum"]:
            raise ValueError("Font variation is outside the supported range")
        values.append(value)
    font.set_variation_by_axes(values)
    return font


def render_variation(font_id, text, weight=None, width=None, optical_size=None):
    with _LOCK:
        return _render_variation(font_id, text, weight, width, optical_size)


def _render_variation(font_id, text, weight=None, width=None, optical_size=None):
    font = variation_face(font_id, weight, width, optical_size)
    box = font.getbbox(text); w, h = box[2] - box[0], box[3] - box[1]
    if not 0 < w <= 20000 or not 0 < h <= 512: raise ValueError("No visible reference text")
    image = Image.new("L", (w + 12, h + 12), 255)
    ImageDraw.Draw(image).text((6 - box[0], 6 - box[1]), text, font=font, fill=0)
    return image


def fitting_axes(font_id):
    record = records_by_id().get(font_id)
    if record is None or not record.get("variable_weight"): return []
    font = ImageFont.truetype(str(ROOT / record["file"]), 96)
    try: axes = {axis["name"]: axis for axis in font.get_variation_axes()}
    except OSError: return []
    results = []
    # Fit one axis at a time to keep cost and geometric flexibility bounded.
    for name, key, trials in ((b"Weight", "weight", (300, 500, 600, 800)),
                              (b"Width", "width", (75, 90, 110, 125)),
                              (b"Optical Size", "optical_size", (12, 32, 64, 96))):
        axis = axes.get(name)
        if axis is None: continue
        for value in trials:
            if axis["minimum"] <= value <= axis["maximum"] and value != axis["default"]:
                results.append({key: value})
    return results
