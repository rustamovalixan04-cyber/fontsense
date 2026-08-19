from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat
from tqdm import tqdm

from .generate_dataset import jpeg_roundtrip, safe_slug
from .generate_full_dataset import (
    EFFECT_SIGNATURE_KEYS,
    _largest_fitting_font_size,
    _scale_transparent_layer,
    plan_full_effects,
)
from .generate_preview import (
    _load_readable_font,
    _portable_path,
    _resolve_path,
    _text_size,
    create_category_contact_sheets,
)
from .google_fonts import CATEGORY_ORDER
from .split import assert_no_family_leakage
from .utils import load_json, project_root, save_json
from .v2_data import V2_SEED, sha256_file

V2_FAMILY_COUNT = 200
V2_IMAGES_PER_FAMILY = 100
V2_IMAGE_COUNT = 20_000
V2_FAMILIES_PER_CATEGORY = 40
V2_SPLIT_FAMILY_COUNTS = {"train": 140, "validation": 30, "test": 30}
V2_SPLIT_IMAGE_COUNTS = {"train": 14_000, "validation": 3_000, "test": 3_000}
V2_EXTRA_EFFECT_KEYS = (
    "resample_scale",
    "perspective_x_shear",
    "perspective_y_shear",
    "stroke_width",
    "noise_std",
    "case_style",
)


def validate_v2_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "seed", "images_per_family", "image_width", "image_height", "font_size_min",
        "font_size_max", "letter_spacing_min", "letter_spacing_max",
        "horizontal_shift_max_px", "vertical_shift_max_px", "rotation_degrees",
        "scale_min", "scale_max", "blur_probability", "blur_radius_min",
        "blur_radius_max", "jpeg_probability", "jpeg_quality_min", "jpeg_quality_max",
        "dark_background_probability", "soft_contrast_probability", "resample_probability",
        "resample_scale_min", "resample_scale_max", "perspective_probability",
        "perspective_max_shear", "stroke_probability", "noise_probability",
        "noise_std_min", "noise_std_max", "case_variation_probability",
        "max_phrase_characters",
        "phrase_adjectives", "phrase_nouns", "fixed_phrases",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"V2 dataset config is missing settings: {sorted(missing)}")
    if int(config["seed"]) != V2_SEED:
        raise ValueError("V2 dataset generation must use seed 42")
    if int(config["images_per_family"]) != V2_IMAGES_PER_FAMILY:
        raise ValueError("Primary V2 generation must create exactly 100 images per family")
    if len(phrase_pool(config)) < V2_IMAGES_PER_FAMILY:
        raise ValueError("V2 dataset config needs at least 100 unique phrases")
    if float(config["rotation_degrees"]) > 3.0 or float(config["blur_radius_max"]) > 0.75:
        raise ValueError("V2 rotation and blur must remain mild")
    if not 0.75 <= float(config["resample_scale_min"]) <= float(config["resample_scale_max"]) <= 1:
        raise ValueError("V2 resampling must remain mild")
    for key in (
        "blur_probability", "jpeg_probability", "dark_background_probability",
        "soft_contrast_probability", "resample_probability", "perspective_probability",
        "stroke_probability", "noise_probability", "case_variation_probability",
    ):
        if not 0 < float(config[key]) < 1:
            raise ValueError(f"{key} must be between zero and one")
    return config


def phrase_pool(config: dict[str, Any]) -> list[str]:
    phrases = [str(value).strip() for value in config["fixed_phrases"]]
    phrases.extend(
        f"{adjective} {noun}"
        for adjective in config["phrase_adjectives"]
        for noun in config["phrase_nouns"]
    )
    maximum = int(config["max_phrase_characters"])
    return list(dict.fromkeys(value for value in phrases if value and len(value) <= maximum))


def phrases_for_rank(config: dict[str, Any], effect_rank: int) -> list[str]:
    rng = random.Random(V2_SEED * 7_919 + effect_rank)
    return rng.sample(phrase_pool(config), V2_IMAGES_PER_FAMILY)


def plan_v2_effects(config: dict[str, Any], random_seed: int, text: str) -> tuple[dict, str]:
    effects = plan_full_effects(config, random_seed)
    rng = random.Random(random_seed + 91_337)
    resample_scale = None
    if rng.random() < float(config["resample_probability"]):
        resample_scale = round(
            rng.uniform(float(config["resample_scale_min"]), float(config["resample_scale_max"])),
            4,
        )
    x_shear = y_shear = None
    if rng.random() < float(config["perspective_probability"]):
        limit = float(config["perspective_max_shear"])
        x_shear = round(rng.uniform(-limit, limit), 5)
        y_shear = round(rng.uniform(-limit, limit), 5)
    stroke_width = 1 if rng.random() < float(config["stroke_probability"]) else 0
    noise_std = None
    if rng.random() < float(config["noise_probability"]):
        noise_std = round(rng.uniform(float(config["noise_std_min"]), float(config["noise_std_max"])), 3)
    case_style = "original"
    rendered_text = text
    if rng.random() < float(config["case_variation_probability"]):
        case_style = rng.choice(["upper", "lower", "title"])
        rendered_text = {
            "upper": text.upper(),
            "lower": text.lower(),
            "title": text.title(),
        }[case_style]
    effects.update(
        resample_scale=resample_scale,
        perspective_x_shear=x_shear,
        perspective_y_shear=y_shear,
        stroke_width=stroke_width,
        noise_std=noise_std,
        case_style=case_style,
    )
    return effects, rendered_text


def _draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    letter_spacing: int,
    stroke_width: int,
) -> None:
    x, y = position
    for character in text:
        draw.text(
            (x, y),
            character,
            fill=fill,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=fill,
        )
        x += draw.textlength(character, font=font) + letter_spacing


def render_v2_image(
    text: str,
    font_path: str | Path,
    config: dict[str, Any],
    effects: dict[str, Any],
) -> Image.Image:
    width = int(config["image_width"])
    height = int(config["image_height"])
    background = tuple(effects["background_rgb"])
    foreground = tuple(effects["foreground_rgb"])
    text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    font = _load_readable_font(Path(font_path), int(effects["actual_font_size"]))
    text_width, text_height, bbox = _text_size(
        draw, text, font, int(effects["letter_spacing_px"])
    )
    if text_width > int(width * 0.78) or text_height > int(height * 0.52):
        raise ValueError(f"Balanced V2 font size does not safely fit {text!r}")
    x = (width - text_width) / 2 + int(effects["horizontal_shift_px"])
    y = (height - text_height) / 2 - bbox[1] + int(effects["vertical_shift_px"])
    _draw_spaced_text(
        draw,
        (x, y),
        text,
        font,
        (*foreground, 255),
        int(effects["letter_spacing_px"]),
        int(effects["stroke_width"]),
    )
    text_layer = _scale_transparent_layer(text_layer, float(effects["scale_factor"]))
    text_layer = text_layer.rotate(
        float(effects["rotation_degrees"]),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0, 0),
    )
    canvas = Image.alpha_composite(
        Image.new("RGBA", (width, height), (*background, 255)), text_layer
    ).convert("RGB")

    x_shear = effects["perspective_x_shear"]
    y_shear = effects["perspective_y_shear"]
    if x_shear is not None:
        canvas = canvas.transform(
            canvas.size,
            Image.Transform.AFFINE,
            (1, float(x_shear), -float(x_shear) * height / 2,
             float(y_shear), 1, -float(y_shear) * width / 2),
            resample=Image.Resampling.BICUBIC,
            fillcolor=background,
        )
    if effects["resample_scale"] is not None:
        scale = float(effects["resample_scale"])
        small = canvas.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = small.resize((width, height), Image.Resampling.BICUBIC)
    if effects["blur_radius"] is not None:
        canvas = canvas.filter(ImageFilter.GaussianBlur(float(effects["blur_radius"])))
    if effects["noise_std"] is not None:
        noise_rng = np.random.default_rng(int(effects["random_seed_for_noise"]))
        values = np.asarray(canvas, dtype=np.float32)
        values += noise_rng.normal(0, float(effects["noise_std"]), values.shape)
        canvas = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8), mode="RGB")
    if effects["jpeg_quality"] is not None:
        canvas = jpeg_roundtrip(canvas, int(effects["jpeg_quality"]))
    return canvas


def _balanced_font_sizes(
    frozen: pd.DataFrame,
    config: dict[str, Any],
    root: Path,
) -> dict[tuple[int, int], int]:
    canvas = Image.new("RGB", (int(config["image_width"]), int(config["image_height"])))
    draw = ImageDraw.Draw(canvas)
    schedule: dict[tuple[int, int], int] = {}
    for effect_rank in tqdm(range(V2_FAMILIES_PER_CATEGORY), desc="Balancing V2 font sizes"):
        fonts = frozen.loc[frozen["effect_rank"] == effect_rank, "path"]
        if len(fonts) != len(CATEGORY_ORDER):
            raise AssertionError(f"V2 effect rank {effect_rank} must contain five fonts")
        for image_index, source_text in enumerate(phrases_for_rank(config, effect_rank)):
            random_seed = V2_SEED * 2_000_003 + effect_rank * 10_007 + image_index
            effects, rendered_text = plan_v2_effects(config, random_seed, source_text)
            sizes = [
                _largest_fitting_font_size(
                    draw,
                    _resolve_path(path, root),
                    rendered_text,
                    int(effects["requested_font_size"]),
                    int(effects["letter_spacing_px"]),
                    int(config["image_width"] * 0.78),
                    int(config["image_height"] * 0.52),
                )
                for path in fonts
            ]
            schedule[(effect_rank, image_index)] = min(sizes)
    return schedule


def build_v2_plan(
    frozen: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    root: Path,
) -> pd.DataFrame:
    working = frozen.copy()
    working["effect_rank"] = -1
    # Every category receives exactly the same 0..39 schedule, while a fixed
    # offset prevents look-alike fonts in different categories from receiving
    # identical text/effect positions.
    for category_index, category in enumerate(CATEGORY_ORDER):
        category_indices = (
            working.loc[working["category"] == category]
            .sort_values("family", key=lambda values: values.str.casefold())
            .index.tolist()
        )
        if len(category_indices) != V2_FAMILIES_PER_CATEGORY:
            raise AssertionError(f"V2 category {category} must contain 40 families")
        for family_position, frame_index in enumerate(category_indices):
            working.at[frame_index, "effect_rank"] = (
                family_position + category_index * 7
            ) % V2_FAMILIES_PER_CATEGORY
    if (working["effect_rank"] < 0).any():
        raise AssertionError("Every V2 family must receive one effect rank")
    sizes = _balanced_font_sizes(working, config, root)
    rows: list[dict] = []
    for row in working.itertuples(index=False):
        effect_rank = int(row.effect_rank)
        for image_index, source_text in enumerate(phrases_for_rank(config, effect_rank)):
            random_seed = V2_SEED * 2_000_003 + effect_rank * 10_007 + image_index
            effects, rendered_text = plan_v2_effects(config, random_seed, source_text)
            effects["actual_font_size"] = sizes[(effect_rank, image_index)]
            effects["random_seed_for_noise"] = random_seed + 31_337
            family_slug = safe_slug(str(row.family))
            image_path = (
                output_dir / "images" / str(row.split) / str(row.category) / family_slug
                / f"{family_slug}_{image_index:04d}.png"
            )
            rows.append(
                {
                    "image_path": _portable_path(image_path, root),
                    "family": str(row.family),
                    "category": str(row.category),
                    "split": str(row.split),
                    "rendered_text": rendered_text,
                    "source_font": str(row.path),
                    "random_seed": random_seed,
                    "image_size": f"{config['image_width']}x{config['image_height']}",
                    "applied_effects": json.dumps(effects, sort_keys=True),
                    "image_index": image_index,
                    "effect_rank": effect_rank,
                }
            )
    return pd.DataFrame(rows)


def validate_v2_structure(frozen: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image_path", "family", "category", "split", "rendered_text", "source_font",
        "random_seed", "image_size", "applied_effects", "image_index", "effect_rank",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"V2 manifest is missing columns: {sorted(missing)}")
    if len(manifest) != V2_IMAGE_COUNT or manifest["family"].nunique() != V2_FAMILY_COUNT:
        raise AssertionError("V2 manifest must contain 20,000 images from 200 families")
    if not manifest.groupby("family").size().eq(V2_IMAGES_PER_FAMILY).all():
        raise AssertionError("Every V2 family must contain exactly 100 images")
    if manifest.groupby("category").size().to_dict() != {
        category: 4_000 for category in CATEGORY_ORDER
    }:
        raise AssertionError("Every V2 category must contain exactly 4,000 images")
    if manifest.groupby("split").size().to_dict() != V2_SPLIT_IMAGE_COUNTS:
        raise AssertionError("V2 split image counts are incorrect")
    if manifest["image_path"].duplicated().any():
        raise AssertionError("V2 manifest contains duplicate image paths")
    assert_no_family_leakage(manifest)
    expected = frozen[["family", "category", "split"]].sort_values("family").reset_index(drop=True)
    actual = (
        manifest[["family", "category", "split"]].drop_duplicates()
        .sort_values("family").reset_index(drop=True)
    )
    if not actual.equals(expected):
        raise AssertionError("V2 manifest does not match the frozen family split")

    signatures: dict[str, Counter] = {}
    phrase_counts: dict[str, Counter] = {}
    rows = []
    signature_keys = EFFECT_SIGNATURE_KEYS + V2_EXTRA_EFFECT_KEYS
    for category in CATEGORY_ORDER:
        group = manifest.loc[manifest["category"] == category]
        effects = [json.loads(value) for value in group["applied_effects"]]
        signatures[category] = Counter(
            json.dumps({key: item[key] for key in signature_keys}, sort_keys=True)
            for item in effects
        )
        phrase_counts[category] = Counter(group["rendered_text"])
        rows.append(
            {
                "category": category,
                "images": len(group),
                "dark_background": sum(item["background"] == "dark" for item in effects),
                "soft_contrast": sum(item["contrast_style"] == "soft" for item in effects),
                "blur": sum(item["blur_radius"] is not None for item in effects),
                "jpeg": sum(item["jpeg_quality"] is not None for item in effects),
                "resample": sum(item["resample_scale"] is not None for item in effects),
                "perspective": sum(item["perspective_x_shear"] is not None for item in effects),
                "stroke": sum(item["stroke_width"] > 0 for item in effects),
                "noise": sum(item["noise_std"] is not None for item in effects),
            }
        )
    reference = CATEGORY_ORDER[0]
    for category in CATEGORY_ORDER[1:]:
        if signatures[category] != signatures[reference]:
            raise AssertionError(f"Category-dependent V2 effect imbalance found for {category}")
        if phrase_counts[category] != phrase_counts[reference]:
            raise AssertionError(f"Category-dependent V2 phrase imbalance found for {category}")
    return pd.DataFrame(rows)


def _dhash(image: Image.Image) -> int:
    gray = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(gray, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def validate_v2_files(manifest: pd.DataFrame, root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    sha_groups: dict[str, list[int]] = defaultdict(list)
    perceptual: list[int] = []
    brightness: list[float] = []
    contrast: list[float] = []
    for index, row in enumerate(
        tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Validating V2 images")
    ):
        path = _resolve_path(row.image_path, root)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty V2 image: {path}")
        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if image.size != (224, 96):
                raise AssertionError(f"Unexpected V2 image dimensions: {path}")
            gray = image.convert("L")
            stats = ImageStat.Stat(gray)
            minimum, maximum = stats.extrema[0]
            if maximum == minimum:
                raise AssertionError(f"Blank V2 image: {path}")
            brightness.append(float(stats.mean[0]))
            contrast.append(float(stats.stddev[0]))
            perceptual.append(_dhash(image))
        sha_groups[sha256_file(path)].append(index)

    exact_rows = []
    for digest, indices in sha_groups.items():
        if len(indices) > 1:
            exact_rows.append(
                {"sha256": digest, "count": len(indices), "image_paths": "|".join(manifest.iloc[indices]["image_path"])}
            )
    exact = pd.DataFrame(exact_rows, columns=["sha256", "count", "image_paths"])
    if not exact.empty:
        raise AssertionError(f"V2 dataset contains {len(exact)} exact duplicate hash groups")

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    segments = ((0, 13), (13, 26), (26, 39), (39, 52), (52, 64))
    near_rows = []
    near_count = 0
    comparisons = 0
    report_limit = 5_000
    candidates_per_bucket = 40
    for index, value in enumerate(perceptual):
        compared_for_image: set[int] = set()
        for segment_index, (start, end) in enumerate(segments):
            mask = (1 << (end - start)) - 1
            key = (segment_index, (value >> start) & mask)
            bucket = buckets[key]
            if len(bucket) <= candidates_per_bucket:
                sampled = bucket
            else:
                sampled = [
                    bucket[round(position * (len(bucket) - 1) / (candidates_per_bucket - 1))]
                    for position in range(candidates_per_bucket)
                ]
            for other in sampled:
                if other in compared_for_image:
                    continue
                compared_for_image.add(other)
                comparisons += 1
                distance = (perceptual[other] ^ value).bit_count()
                if distance > 4:
                    continue
                near_count += 1
                if len(near_rows) >= report_limit:
                    continue
                left_row = manifest.iloc[other]
                right_row = manifest.iloc[index]
                near_rows.append(
                    {
                        "left_image": left_row["image_path"],
                        "right_image": right_row["image_path"],
                        "hamming_distance": distance,
                        "same_family": left_row["family"] == right_row["family"],
                        "same_split": left_row["split"] == right_row["split"],
                        "same_text": left_row["rendered_text"] == right_row["rendered_text"],
                    }
                )
            buckets[key].append(index)
    near = pd.DataFrame(
        near_rows,
        columns=["left_image", "right_image", "hamming_distance", "same_family", "same_split", "same_text"],
    ).sort_values(["hamming_distance", "left_image"], ignore_index=True)
    return (
        {
            "images_opened": len(manifest),
            "missing_images": 0,
            "empty_files": 0,
            "blank_images": 0,
            "corrupted_images": 0,
            "exact_duplicate_groups": 0,
            "near_duplicate_method": (
                "64-bit difference hash; up to 40 evenly spaced candidates from each of five "
                "shared hash segments per image; Hamming distance <= 4"
            ),
            "near_duplicate_comparisons": comparisons,
            "near_duplicate_pairs_distance_le_4": near_count,
            "near_duplicate_rows_saved": len(near),
            "near_duplicate_report_truncated": near_count > len(near),
            "mean_brightness": float(np.mean(brightness)),
            "mean_contrast": float(np.mean(contrast)),
        },
        exact,
        near,
    )


def _contact_sample(manifest: pd.DataFrame) -> pd.DataFrame:
    samples = []
    for category in CATEGORY_ORDER:
        for split in ("train", "validation", "test"):
            group = manifest.loc[(manifest["category"] == category) & (manifest["split"] == split)]
            samples.append(group.groupby("family", sort=True).head(1).head(4))
    return pd.concat(samples, ignore_index=True)


def generate_v2_dataset(
    split_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    report_dir: str | Path,
    *,
    reuse_existing_render: bool = False,
) -> dict:
    root = project_root()
    split_path = Path(split_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    report_dir = Path(report_dir)
    frozen = pd.read_csv(split_path, keep_default_na=False)
    assert_no_family_leakage(frozen)
    if len(frozen) != V2_FAMILY_COUNT:
        raise AssertionError("Frozen V2 split must contain exactly 200 families")
    config = validate_v2_config(load_json(config_path))
    split_hash_before = sha256_file(split_path)
    plan = build_v2_plan(frozen, config, output_dir, root)
    effect_balance = validate_v2_structure(frozen, plan)

    report_dir.mkdir(parents=True, exist_ok=True)
    plan_fingerprint = hashlib.sha256(
        plan.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    render_receipt_path = report_dir / "render_receipt.json"
    reuse_render = False
    if render_receipt_path.is_file():
        receipt = load_json(render_receipt_path)
        reuse_render = bool(
            receipt.get("status") == "completed"
            and receipt.get("plan_sha256") == plan_fingerprint
            and receipt.get("split_sha256") == split_hash_before
            and receipt.get("config_sha256") == sha256_file(config_path)
            and receipt.get("images_rendered") == V2_IMAGE_COUNT
            and all(_resolve_path(path, root).is_file() for path in plan["image_path"])
        )
    elif reuse_existing_render:
        reuse_render = all(
            _resolve_path(path, root).is_file() for path in plan["image_path"]
        )
        if not reuse_render:
            raise FileNotFoundError(
                "Cannot resume V2 validation because one or more rendered images are missing"
            )
        save_json(
            {
                "status": "completed",
                "images_rendered": V2_IMAGE_COUNT,
                "plan_sha256": plan_fingerprint,
                "split_sha256": split_hash_before,
                "config_sha256": sha256_file(config_path),
                "recovered_after_completed_render_validation_failure": True,
            },
            render_receipt_path,
        )
    if not reuse_render:
        for row in tqdm(plan.itertuples(index=False), total=len(plan), desc="Rendering V2 images"):
            image_path = _resolve_path(row.image_path, root)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            effects = json.loads(row.applied_effects)
            image = render_v2_image(
                row.rendered_text,
                _resolve_path(row.source_font, root),
                config,
                effects,
            )
            image.save(image_path, format="PNG", optimize=True)
        save_json(
            {
                "status": "completed",
                "images_rendered": V2_IMAGE_COUNT,
                "plan_sha256": plan_fingerprint,
                "split_sha256": split_hash_before,
                "config_sha256": sha256_file(config_path),
            },
            render_receipt_path,
        )

    file_results, exact, near = validate_v2_files(plan, root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(manifest_path, index=False)
    effect_balance_path = report_dir / "effect_balance.csv"
    exact_path = report_dir / "exact_duplicates.csv"
    near_path = report_dir / "near_duplicates.csv"
    effect_balance.to_csv(effect_balance_path, index=False)
    exact.to_csv(exact_path, index=False)
    near.to_csv(near_path, index=False)
    contact_sheets = create_category_contact_sheets(
        _contact_sample(plan),
        report_dir / "contact_sheets",
        root=root,
        columns=int(config.get("contact_sheet_columns", 6)),
        index_column="image_index",
        title_suffix="V2 samples across train, validation, and test",
    )

    rebuilt = build_v2_plan(frozen, config, output_dir, root)
    rebuilt_fingerprint = hashlib.sha256(
        rebuilt.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    if not rebuilt.equals(plan) or rebuilt_fingerprint != plan_fingerprint:
        raise AssertionError("Rerunning the V2 seed did not reproduce the manifest plan")
    sample_hashes = {}
    with tempfile.TemporaryDirectory(prefix="fontsense_v2_repro_"):
        for row in plan.groupby("category", sort=True).head(5).itertuples(index=False):
            effects = json.loads(row.applied_effects)
            regenerated = render_v2_image(
                row.rendered_text,
                _resolve_path(row.source_font, root),
                config,
                effects,
            )
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                regenerated.save(temporary_path, format="PNG", optimize=True)
                original_hash = sha256_file(_resolve_path(row.image_path, root))
                regenerated_hash = sha256_file(temporary_path)
                if original_hash != regenerated_hash:
                    raise AssertionError(f"V2 sample hash did not reproduce: {row.image_path}")
                sample_hashes[row.image_path] = original_hash
            finally:
                temporary_path.unlink(missing_ok=True)

    if sha256_file(split_path) != split_hash_before:
        raise AssertionError("Frozen V2 split changed during generation")
    summary = {
        "status": "passed",
        "seed": V2_SEED,
        "families_total": V2_FAMILY_COUNT,
        "images_per_family": V2_IMAGES_PER_FAMILY,
        "images_total": len(plan),
        "images_per_category": plan.groupby("category").size().to_dict(),
        "images_per_split": plan.groupby("split").size().to_dict(),
        "family_overlap_count": 0,
        **file_results,
        "category_effect_balance": "passed exact schedule match",
        "category_phrase_balance": "passed exact distribution match",
        "split_path": _portable_path(split_path, root),
        "split_sha256": split_hash_before,
        "config_path": _portable_path(config_path, root),
        "config_sha256": sha256_file(config_path),
        "manifest_path": _portable_path(manifest_path, root),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_plan_reproduced": True,
        "sample_images_reproduced": len(sample_hashes),
        "sample_image_hashes": sample_hashes,
        "contact_sheets": [_portable_path(path, root) for path in contact_sheets],
        "model_training_performed": False,
    }
    save_json(summary, report_dir / "dataset_validation_summary.json")
    save_json(
        {
            "status": "passed",
            "manifest_sha256_first": summary["manifest_sha256"],
            "manifest_plan_sha256_first": plan_fingerprint,
            "manifest_plan_sha256_second": rebuilt_fingerprint,
            "manifest_plan_reproduced": True,
            "checked_image_hashes": sample_hashes,
        },
        report_dir / "reproducibility.json",
    )
    return summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Generate and validate FontSense V2 20k dataset")
    parser.add_argument("--split", default=str(root / "data/v2/frozen_family_split.csv"))
    parser.add_argument("--config", default=str(root / "config/v2/dataset.json"))
    parser.add_argument("--output-dir", default=str(root / "data/v2/processed/fontsense_v2"))
    parser.add_argument("--manifest", default=str(root / "reports/v2/data/full_manifest.csv"))
    parser.add_argument("--report-dir", default=str(root / "reports/v2/data"))
    parser.add_argument(
        "--reuse-existing-render",
        action="store_true",
        help="Resume validation only after a prior run completed all 20,000 renders.",
    )
    args = parser.parse_args()
    summary_path = Path(args.report_dir) / "dataset_validation_summary.json"
    if summary_path.is_file() and Path(args.manifest).is_file():
        existing = load_json(summary_path)
        if (
            existing.get("status") == "passed"
            and existing.get("images_total") == V2_IMAGE_COUNT
            and sha256_file(args.manifest) == existing.get("manifest_sha256")
            and sha256_file(args.split) == existing.get("split_sha256")
        ):
            print(json.dumps({"status": "reused", "summary": existing}, indent=2))
            return
    result = generate_v2_dataset(
        args.split,
        args.config,
        args.output_dir,
        args.manifest,
        args.report_dir,
        reuse_existing_render=args.reuse_existing_render,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
