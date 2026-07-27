import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import get_data_path
from PIL import ImageStat

from fontsense.generate_full_dataset import (
    FULL_IMAGE_COUNT,
    plan_full_effects,
    render_full_image,
    validate_full_config,
    validate_full_dataset_structure,
)
from fontsense.split import FONT_CATEGORIES


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "full_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_split() -> pd.DataFrame:
    rows = []
    for category in FONT_CATEGORIES:
        for family_index in range(18):
            if family_index < 12:
                split = "train"
            elif family_index < 15:
                split = "validation"
            else:
                split = "test"
            rows.append(
                {
                    "family": f"{category}_{family_index:02d}",
                    "category": category,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _structural_manifest(frozen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in frozen.itertuples(index=False):
        family_rank = int(str(family.family).rsplit("_", 1)[1])
        for image_index in range(40):
            random_seed = 42 * 2_000_003 + family_rank * 10_007 + image_index
            effects = plan_full_effects(_config(), random_seed)
            effects["actual_font_size"] = effects["requested_font_size"]
            rows.append(
                {
                    "image_path": f"images/{family.family}_{image_index:04d}.png",
                    "family": family.family,
                    "category": family.category,
                    "split": family.split,
                    "rendered_text": f"phrase_{family_rank}_{image_index % 10}",
                    "source_font": "font.ttf",
                    "random_seed": random_seed,
                    "image_size": "224x96",
                    "applied_effects": json.dumps(effects, sort_keys=True),
                    "image_index": image_index,
                }
            )
    return pd.DataFrame(rows)


def test_full_structure_has_exact_counts_and_category_balanced_effects():
    frozen = _frozen_split()
    manifest = _structural_manifest(frozen)

    effect_balance = validate_full_dataset_structure(frozen, manifest)

    assert len(manifest) == FULL_IMAGE_COUNT
    assert manifest.groupby("family").size().eq(40).all()
    assert effect_balance["images"].tolist() == [720] * len(FONT_CATEGORIES)
    compared_columns = [
        "light_background",
        "dark_background",
        "soft_contrast",
        "strong_contrast",
        "mild_blur",
        "jpeg_compression",
        "mean_actual_font_size",
    ]
    for column in compared_columns:
        assert effect_balance[column].nunique() == 1


def test_full_structure_rejects_category_dependent_effect_change():
    frozen = _frozen_split()
    manifest = _structural_manifest(frozen)
    changed_index = manifest.index[manifest["category"] == "display"][0]
    effects = json.loads(manifest.at[changed_index, "applied_effects"])
    effects["background"] = "changed_only_for_display"
    manifest.at[changed_index, "applied_effects"] = json.dumps(effects, sort_keys=True)

    with pytest.raises(AssertionError, match="Category-dependent effect imbalance"):
        validate_full_dataset_structure(frozen, manifest)


def test_full_renderer_outputs_readable_sized_image_and_records_effects():
    font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"

    image, effects = render_full_image(
        "Clear Message",
        font_path,
        _config(),
        random_seed=42,
    )

    minimum, maximum = ImageStat.Stat(image.convert("L")).extrema[0]
    assert image.size == (224, 96)
    assert maximum > minimum
    assert set(effects) >= {
        "background",
        "contrast_style",
        "letter_spacing_px",
        "horizontal_shift_px",
        "scale_factor",
        "rotation_degrees",
        "blur_radius",
        "jpeg_quality",
        "actual_font_size",
    }


def test_full_config_rejects_wrong_seed_or_image_count():
    wrong_seed = _config()
    wrong_seed["seed"] = 7
    with pytest.raises(ValueError, match="fixed seed 42"):
        validate_full_config(wrong_seed)

    wrong_count = _config()
    wrong_count["images_per_family"] = 39
    with pytest.raises(ValueError, match="exactly 40"):
        validate_full_config(wrong_count)
