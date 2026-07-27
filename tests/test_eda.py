import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

from fontsense.eda import (
    MODEL_FEATURE_AUDIT,
    audit_image_files,
    effect_balance_table,
    find_suspicious_pairs,
    parse_effects,
    phrase_balance,
    validate_dataset_structure,
)
from fontsense.split import FONT_CATEGORIES


def _effects() -> dict:
    return {
        "actual_font_size": 24,
        "background": "light",
        "blur_radius": None,
        "contrast_style": "strong",
        "horizontal_shift_px": 1,
        "jpeg_quality": None,
        "letter_spacing_px": 1,
        "luminance_difference": 200.0,
        "rotation_degrees": 0.5,
        "scale_factor": 1.0,
        "vertical_shift_px": 0,
    }


def _small_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = ["train", "train", "validation", "validation", "test"]
    rows = []
    frozen = []
    for index, (category, split) in enumerate(zip(FONT_CATEGORIES, splits)):
        family = f"Family {index}"
        rows.append(
            {
                "image_path": f"images/{index}.png",
                "family": family,
                "category": category,
                "split": split,
                "rendered_text": "Clear Type",
                "source_font": "font.ttf",
                "applied_effects": json.dumps(_effects()),
            }
        )
        frozen.append({"family": family, "category": category, "split": split})
    return pd.DataFrame(rows), pd.DataFrame(frozen)


def test_structure_uses_image_path_as_key_and_rejects_family_leakage():
    manifest, frozen = _small_manifest()

    result = validate_dataset_structure(
        manifest,
        frozen,
        expected_images=5,
        expected_families=5,
    )

    assert result["candidate_key"] == "image_path"
    assert result["family_overlap_count"] == 0

    leaked = pd.concat(
        [
            manifest,
            manifest.iloc[[0]].assign(
                image_path="images/leak.png",
                split="test",
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(AssertionError, match="Family leakage"):
        validate_dataset_structure(
            leaked,
            frozen,
            expected_images=6,
            expected_families=5,
        )


def test_image_audit_detects_blank_and_exact_duplicate_images(tmp_path: Path):
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    blank = tmp_path / "blank.png"
    image = Image.new("RGB", (64, 32), "white")
    draw = ImageDraw.Draw(image)
    draw.text((5, 8), "Type", fill="black")
    image.save(first)
    image.save(duplicate)
    Image.new("RGB", (64, 32), "white").save(blank)
    manifest = pd.DataFrame(
        [
            {
                "image_path": first.name,
                "family": "A",
                "category": "serif",
                "split": "train",
            },
            {
                "image_path": duplicate.name,
                "family": "B",
                "category": "sans_serif",
                "split": "validation",
            },
            {
                "image_path": blank.name,
                "family": "C",
                "category": "display",
                "split": "test",
            },
        ]
    )

    quality = audit_image_files(manifest, root=tmp_path)
    suspicious = find_suspicious_pairs(quality, root=tmp_path)

    assert quality["opens_successfully"].all()
    assert quality["blank"].sum() == 1
    assert quality.loc[0, "sha256"] == quality.loc[1, "sha256"]
    assert suspicious["exact_file_duplicate"].any()


def test_missing_image_is_recorded_without_fabricating_metrics(tmp_path: Path):
    manifest = pd.DataFrame(
        [
            {
                "image_path": "missing.png",
                "family": "Missing",
                "category": "serif",
                "split": "train",
            }
        ]
    )

    quality = audit_image_files(manifest, root=tmp_path)

    assert not bool(quality.loc[0, "exists"])
    assert not bool(quality.loc[0, "opens_successfully"])
    assert quality.loc[0, "error"] == "missing file"
    assert pd.isna(quality.loc[0].get("brightness_mean"))


def test_effect_and_phrase_balance_are_zero_for_equal_schedules():
    manifest, _ = _small_manifest()
    effects = parse_effects(manifest)

    effect_table, effect_summary = effect_balance_table(manifest, effects)
    _, phrase_summary = phrase_balance(manifest)

    assert len(effect_table) == len(FONT_CATEGORIES)
    assert effect_summary["maximum_binary_effect_rate_spread"] == 0
    assert not effect_summary["serious_category_dependent_effect_imbalance"]
    assert phrase_summary["cramers_v"] == 0
    assert not phrase_summary["strong_category_association"]


def test_model_feature_audit_excludes_manifest_metadata():
    excluded = set(MODEL_FEATURE_AUDIT["excluded_from_model_features"])

    assert MODEL_FEATURE_AUDIT["status"] == "passed_by_code_inspection"
    assert {"family", "rendered_text", "source_font", "file name"} <= excluded
