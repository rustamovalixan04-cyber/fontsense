from __future__ import annotations

import json
from pathlib import Path

from matplotlib import get_data_path
from PIL import ImageStat

from fontsense.v2_generate import (
    V2_IMAGE_COUNT,
    phrase_pool,
    plan_v2_effects,
    render_v2_image,
    validate_v2_config,
)


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "config/v2/dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_config_has_the_primary_20k_plan_and_phrase_diversity():
    config = validate_v2_config(_config())
    assert config["images_per_family"] == 100
    assert V2_IMAGE_COUNT == 20_000
    assert len(phrase_pool(config)) >= 100


def test_v2_effects_are_deterministic_and_include_realistic_options():
    first, first_text = plan_v2_effects(_config(), 42, "Modern Design")
    second, second_text = plan_v2_effects(_config(), 42, "Modern Design")
    assert first == second
    assert first_text == second_text
    assert set(first) >= {
        "resample_scale", "perspective_x_shear", "stroke_width", "noise_std", "case_style"
    }


def test_v2_renderer_creates_a_nonblank_expected_size_image():
    config = _config()
    effects, text = plan_v2_effects(config, 42, "Clear Message")
    effects["actual_font_size"] = 20
    effects["random_seed_for_noise"] = 42 + 31_337
    font = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    image = render_v2_image(text, font, config, effects)
    minimum, maximum = ImageStat.Stat(image.convert("L")).extrema[0]
    assert image.size == (224, 96)
    assert maximum > minimum
