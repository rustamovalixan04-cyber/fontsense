from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageOps

from api.font_match import render_reference
from api.font_variations import render_variation, variation_face
from api.glyph_match import glyphs, agreement


def test_glyph_alignment_keeps_same_shapes_with_changed_letter_spacing():
    source = render_reference("arimo", "Clear letters").convert("L")
    actual = glyphs(source)
    assert actual is not None
    assert agreement(actual, actual) == pytest.approx(1.)
    assert agreement(actual, glyphs(ImageOps.expand(source, (30, 10), "white"))) == pytest.approx(1.)
    assert agreement(actual, glyphs(render_reference("arimo", "Different shapes"))) is None


def test_variable_font_fitting_uses_real_glyph_weights_and_preserves_base_face():
    text = "Clear letters"
    before = render_reference("arimo", text).tobytes()
    regular = render_variation("x-arimo", text, weight=400)
    medium = render_variation("x-arimo", text, weight=600)
    assert (255 - np.asarray(medium, dtype=np.float32)).sum() > (255 - np.asarray(regular, dtype=np.float32)).sum() * 1.1
    assert render_reference("arimo", text).tobytes() == before


@pytest.mark.parametrize("font_id,weight", [("../../Windows/Fonts/arial.ttf", 400), ("x-arimo", float("nan")), ("x-arimo", 10000)])
def test_variation_parameters_cannot_select_files_or_invalid_axes(font_id, weight):
    with pytest.raises(ValueError): variation_face(font_id, weight)
