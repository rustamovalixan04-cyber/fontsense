"""Broader retrieval checks; no historical capstone data or results are used."""
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
from PIL import Image, ImageOps

from api import extended_match
from api.font_match import identify, render_reference

ROOT = Path(__file__).resolve().parents[1]


def test_extended_catalog_is_large_licensed_and_does_not_replace_category_catalog():
    metadata = json.loads((ROOT / "artifacts/web/extended_catalog.json").read_text())
    assert metadata["families"] >= 1500
    assert len(extended_match.primary_styles()) == metadata["families"]
    assert len(extended_match.catalog()) == metadata["styles"]
    assert all(source["download_url"].startswith("https://raw.githubusercontent.com/google/fonts/") for source in metadata["sources"])
    assert all("source_sha256" in source and "license_sha256" in source for source in metadata["sources"])
    original = json.loads((ROOT / "artifacts/web/font_catalog.json").read_text())
    assert len({font["family"] for font in original["fonts"]}) == 56


@pytest.mark.parametrize("family", ["Bangers", "Manrope", "Bitter", "Oxygen Mono", "Courgette"])
def test_families_outside_old_library_can_be_retrieved(family):
    font = next(font for font in extended_match.catalog() if font["family"] == family and not font["italic"])
    query = ImageOps.expand(extended_match.render_reference(font["id"], "Clear new letters", 73).convert("RGB"), 30, "white")
    result = identify(query, "Clear new letters")
    names = [font["family"] for font in result["matches"][:3]]
    if result["identified_font"]:
        names.insert(0, result["identified_font"]["family"])
    assert family in names
    if result["identified_font"]:
        assert result["identified_font"]["family"] == family


def test_times_new_roman_crop_shows_candidates_instead_of_a_false_exact_tinos():
    image = Image.open(ROOT / "tests/fixtures/times_new_roman_screenshot_crop.png").convert("RGB")
    result = identify(image)
    assert result["category"] == "serif"
    assert result["identified_font"] is None
    names = {font["family"] for font in result["font_candidates"]}
    assert "Times New Roman" in names
    assert result["matches"]


def test_compatible_fonts_stay_possible_and_not_certain():
    result = identify(ImageOps.expand(render_reference("tinos", "Clear new letters"), 30, "white"), "Clear new letters")
    assert result["identified_font"] is None
    assert result["font_candidates"]


def test_short_repeated_text_does_not_force_broad_font_identity():
    ranked = [{"family": "Example", "score": 1.}, {"family": "Other", "score": .5}]
    assert extended_match.likely_match(ranked, "IIIIII") is None


def test_unknown_font_identifier_cannot_read_arbitrary_paths():
    with pytest.raises(ValueError, match="Unknown font"):
        extended_match.font_face("../../Windows/Fonts/times.ttf")


def test_renamed_subset_preserves_rendered_reference_shapes():
    original = render_reference("arimo", "Clear new letters").convert("L")
    record = next(font for font in extended_match.catalog() if font["family"] == "Arimo" and font["weight"] == 400 and not font["italic"])
    derived = extended_match.render_reference(record["id"], "Clear new letters", 96)
    np.testing.assert_array_equal(np.asarray(original), np.asarray(derived))


def test_extended_font_files_are_ignored_and_original_model_is_unchanged():
    ignored = (ROOT / ".gitignore").read_text()
    assert "artifacts/web/extended-fonts/" in ignored
    actual = hashlib.sha256((ROOT / "artifacts/cnn/cnn_model.pt").read_bytes()).hexdigest()
    assert actual == "c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f"
