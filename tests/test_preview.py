import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import get_data_path

from fontsense.generate_preview import (
    PREVIEW_IMAGE_COUNT,
    _load_readable_font,
    generate_preview_dataset,
    load_frozen_family_split,
)
from fontsense.split import FONT_CATEGORIES


def _frozen_rows(font_path: str) -> list[dict]:
    rows = []
    for category in FONT_CATEGORIES:
        for index in range(18):
            if index < 12:
                split = "train"
            elif index < 15:
                split = "validation"
            else:
                split = "test"
            rows.append(
                {
                    "family": f"{category}_{index:02d}",
                    "category": category,
                    "split": split,
                    "path": font_path,
                    "source": "test source",
                    "license": "OFL",
                    "selection_seed": 42,
                }
            )
    return rows


def _preview_config() -> dict:
    return {
        "seed": 42,
        "images_per_family": 2,
        "image_width": 112,
        "image_height": 48,
        "font_size_min": 16,
        "font_size_max": 24,
        "letter_spacing_min": 0,
        "letter_spacing_max": 1,
        "rotation_degrees": 1.0,
        "blur_probability": 0.2,
        "blur_radius_min": 0.1,
        "blur_radius_max": 0.3,
        "jpeg_probability": 0.25,
        "jpeg_quality_min": 82,
        "jpeg_quality_max": 92,
        "dark_background_probability": 0.25,
        "contact_sheet_columns": 6,
        "phrases": ["Font Sense", "Type Study", "Hello World", "Clear Text"],
    }


def test_preview_generation_preserves_the_frozen_split_and_exact_counts(tmp_path):
    font_path = str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    split_path = tmp_path / "frozen_split.csv"
    config_path = tmp_path / "preview.json"
    output_dir = tmp_path / "preview_data"
    manifest_path = tmp_path / "reports" / "preview_manifest.csv"
    summary_path = tmp_path / "reports" / "preview_summary.json"
    contact_sheet_dir = tmp_path / "reports" / "contact_sheets"
    frozen = pd.DataFrame(_frozen_rows(font_path))
    frozen.to_csv(split_path, index=False)
    config_path.write_text(json.dumps(_preview_config()), encoding="utf-8")

    manifest, summary, contact_sheets = generate_preview_dataset(
        split_path,
        output_dir,
        config_path,
        manifest_path,
        summary_path,
        contact_sheet_dir,
        root=tmp_path,
    )

    assert len(manifest) == PREVIEW_IMAGE_COUNT
    assert manifest.groupby("family").size().eq(2).all()
    assert manifest.groupby("category").size().to_dict() == {
        category: 36 for category in FONT_CATEGORIES
    }
    assert manifest.groupby("split").size().to_dict() == {
        "test": 30,
        "train": 120,
        "validation": 30,
    }
    expected_assignments = frozen.set_index("family")[["category", "split"]].sort_index()
    actual_assignments = (
        manifest.drop_duplicates("family").set_index("family")[["category", "split"]].sort_index()
    )
    pd.testing.assert_frame_equal(actual_assignments, expected_assignments)
    assert len(list((output_dir / "images").rglob("*.png"))) == PREVIEW_IMAGE_COUNT
    assert summary["status"] == "passed"
    assert summary["family_overlap_count"] == 0
    assert summary["fully_blank_images"] == 0
    assert len(contact_sheets) == len(FONT_CATEGORIES)
    assert all(path.is_file() for path in contact_sheets)
    assert manifest_path.is_file()
    assert summary_path.is_file()


def test_frozen_split_rejects_a_non_42_selection_seed(tmp_path):
    font_path = str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    split_path = tmp_path / "wrong_seed.csv"
    rows = _frozen_rows(font_path)
    for row in rows:
        row["selection_seed"] = 7
    pd.DataFrame(rows).to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="selection seed 42"):
        load_frozen_family_split(split_path, root=tmp_path)


def test_variable_font_uses_normal_weight_when_available(monkeypatch):
    class FakeVariableFont:
        def __init__(self):
            self.selected_axes = None

        def get_variation_axes(self):
            return [
                {"minimum": 100, "default": 100, "maximum": 900, "name": b"Weight"},
                {"minimum": 50, "default": 100, "maximum": 200, "name": b"Width"},
            ]

        def set_variation_by_axes(self, values):
            self.selected_axes = values

    fake_font = FakeVariableFont()
    monkeypatch.setattr(
        "fontsense.generate_preview.ImageFont.truetype",
        lambda *_args, **_kwargs: fake_font,
    )

    loaded = _load_readable_font(Path("variable.ttf"), 24)

    assert loaded.selected_axes == [400, 100]
