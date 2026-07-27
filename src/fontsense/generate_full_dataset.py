from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageStat
from tqdm import tqdm

from .generate_dataset import jpeg_roundtrip, safe_slug
from .generate_preview import (
    _draw_text_with_spacing,
    _load_readable_font,
    _portable_path,
    _resolve_path,
    _sha256,
    _text_size,
    create_category_contact_sheets,
    load_frozen_family_split,
)
from .split import FONT_CATEGORIES, assert_no_family_leakage
from .utils import load_json, project_root, save_json

FULL_SEED = 42
FULL_IMAGES_PER_FAMILY = 40
FULL_FAMILY_COUNT = 90
FULL_IMAGE_COUNT = FULL_FAMILY_COUNT * FULL_IMAGES_PER_FAMILY
EFFECT_SIGNATURE_KEYS = (
    "background",
    "background_rgb",
    "contrast_style",
    "foreground_rgb",
    "luminance_difference",
    "letter_spacing_px",
    "requested_font_size",
    "actual_font_size",
    "horizontal_shift_px",
    "vertical_shift_px",
    "scale_factor",
    "rotation_degrees",
    "blur_radius",
    "jpeg_quality",
)
REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "family",
    "category",
    "split",
    "rendered_text",
    "source_font",
    "random_seed",
    "image_size",
    "applied_effects",
}


def validate_full_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "seed",
        "images_per_family",
        "image_width",
        "image_height",
        "font_size_min",
        "font_size_max",
        "letter_spacing_min",
        "letter_spacing_max",
        "horizontal_shift_max_px",
        "vertical_shift_max_px",
        "rotation_degrees",
        "scale_min",
        "scale_max",
        "blur_probability",
        "blur_radius_min",
        "blur_radius_max",
        "jpeg_probability",
        "jpeg_quality_min",
        "jpeg_quality_max",
        "dark_background_probability",
        "soft_contrast_probability",
        "phrases",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Full dataset config is missing settings: {sorted(missing)}")
    if int(config["seed"]) != FULL_SEED:
        raise ValueError(f"Full dataset generation must use fixed seed {FULL_SEED}.")
    if int(config["images_per_family"]) != FULL_IMAGES_PER_FAMILY:
        raise ValueError(
            f"Full dataset generation must create exactly {FULL_IMAGES_PER_FAMILY} images per family."
        )
    if len(set(config["phrases"])) < FULL_IMAGES_PER_FAMILY:
        raise ValueError("Full dataset config needs at least 40 different words or phrases.")
    if int(config["font_size_min"]) > int(config["font_size_max"]):
        raise ValueError("font_size_min cannot be larger than font_size_max.")
    if int(config["letter_spacing_min"]) > int(config["letter_spacing_max"]):
        raise ValueError("letter_spacing_min cannot be larger than letter_spacing_max.")
    if not 0.9 <= float(config["scale_min"]) <= float(config["scale_max"]) <= 1.1:
        raise ValueError("Full dataset scaling must stay within the mild range 0.9 to 1.1.")
    if float(config["rotation_degrees"]) > 3.0:
        raise ValueError("Full dataset rotation must not exceed 3 degrees.")
    if float(config["blur_radius_max"]) > 0.75:
        raise ValueError("Full dataset blur must remain mild (0.75 pixels or less).")
    for key in (
        "blur_probability",
        "jpeg_probability",
        "dark_background_probability",
        "soft_contrast_probability",
    ):
        if not 0.0 < float(config[key]) < 1.0:
            raise ValueError(f"{key} must be between 0 and 1.")
    return config


def plan_full_effects(config: dict[str, Any], random_seed: int) -> dict[str, Any]:
    """Choose effects using only a seed, never the font category."""
    rng = random.Random(random_seed)
    is_dark = rng.random() < float(config["dark_background_probability"])
    is_soft = rng.random() < float(config["soft_contrast_probability"])

    if is_dark:
        background = [rng.randint(5, 45) for _ in range(3)]
        foreground_range = (170, 210) if is_soft else (220, 255)
        background_name = "dark"
    else:
        background = [rng.randint(232, 255) for _ in range(3)]
        foreground_range = (70, 110) if is_soft else (0, 55)
        background_name = "light"
    foreground = [rng.randint(*foreground_range) for _ in range(3)]
    luminance_difference = round(
        abs(sum(background) / 3 - sum(foreground) / 3),
        2,
    )

    blur_radius: float | None = None
    if rng.random() < float(config["blur_probability"]):
        blur_radius = round(
            rng.uniform(
                float(config["blur_radius_min"]),
                float(config["blur_radius_max"]),
            ),
            3,
        )
    jpeg_quality: int | None = None
    if rng.random() < float(config["jpeg_probability"]):
        jpeg_quality = rng.randint(
            int(config["jpeg_quality_min"]),
            int(config["jpeg_quality_max"]),
        )

    return {
        "background": background_name,
        "background_rgb": background,
        "contrast_style": "soft" if is_soft else "strong",
        "foreground_rgb": foreground,
        "luminance_difference": luminance_difference,
        "letter_spacing_px": rng.randint(
            int(config["letter_spacing_min"]),
            int(config["letter_spacing_max"]),
        ),
        "requested_font_size": rng.randint(
            int(config["font_size_min"]),
            int(config["font_size_max"]),
        ),
        "horizontal_shift_px": rng.randint(
            -int(config["horizontal_shift_max_px"]),
            int(config["horizontal_shift_max_px"]),
        ),
        "vertical_shift_px": rng.randint(
            -int(config["vertical_shift_max_px"]),
            int(config["vertical_shift_max_px"]),
        ),
        "scale_factor": round(
            rng.uniform(float(config["scale_min"]), float(config["scale_max"])),
            4,
        ),
        "rotation_degrees": round(
            rng.uniform(
                -float(config["rotation_degrees"]),
                float(config["rotation_degrees"]),
            ),
            3,
        ),
        "blur_radius": blur_radius,
        "jpeg_quality": jpeg_quality,
    }


def _fit_full_font(
    draw: ImageDraw.ImageDraw,
    font_path: Path,
    text: str,
    requested_size: int,
    letter_spacing: int,
    max_width: int,
    max_height: int,
):
    for size in range(requested_size, 9, -1):
        font = _load_readable_font(font_path, size)
        text_width, text_height, bbox = _text_size(draw, text, font, letter_spacing)
        if text_width <= max_width and text_height <= max_height:
            return font, text_width, bbox
    raise ValueError(f"Text '{text}' could not be fitted with font file {font_path}.")


def _largest_fitting_font_size(
    draw: ImageDraw.ImageDraw,
    font_path: Path,
    text: str,
    requested_size: int,
    letter_spacing: int,
    max_width: int,
    max_height: int,
) -> int:
    font, _, _ = _fit_full_font(
        draw,
        font_path,
        text,
        requested_size,
        letter_spacing,
        max_width,
        max_height,
    )
    return font.size


def _phrases_for_rank(config: dict[str, Any], effect_rank: int) -> list[str]:
    phrase_rng = random.Random(FULL_SEED * 7_919 + effect_rank)
    return phrase_rng.sample(list(config["phrases"]), FULL_IMAGES_PER_FAMILY)


def balanced_font_size_schedule(
    frozen_split: pd.DataFrame,
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[tuple[int, int], int]:
    """Choose one safe actual size for each effect position across all categories."""
    width = int(config["image_width"])
    height = int(config["image_height"])
    measuring_canvas = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(measuring_canvas)
    schedule: dict[tuple[int, int], int] = {}

    for effect_rank in tqdm(range(18), desc="Balancing font sizes"):
        rank_fonts = frozen_split.loc[frozen_split["_effect_rank"] == effect_rank, "path"]
        if len(rank_fonts) != len(FONT_CATEGORIES):
            raise AssertionError(
                f"Effect rank {effect_rank} must contain one font from each category."
            )
        phrases = _phrases_for_rank(config, effect_rank)
        for image_index, text in enumerate(phrases):
            random_seed = FULL_SEED * 2_000_003 + effect_rank * 10_007 + image_index
            effects = plan_full_effects(config, random_seed)
            fitting_sizes = [
                _largest_fitting_font_size(
                    draw,
                    _resolve_path(font_path, root),
                    text,
                    int(effects["requested_font_size"]),
                    int(effects["letter_spacing_px"]),
                    int(width * 0.78),
                    int(height * 0.52),
                )
                for font_path in rank_fonts
            ]
            schedule[(effect_rank, image_index)] = min(fitting_sizes)
    return schedule


def _scale_transparent_layer(layer: Image.Image, scale_factor: float) -> Image.Image:
    width, height = layer.size
    scaled_size = (
        max(1, round(width * scale_factor)),
        max(1, round(height * scale_factor)),
    )
    scaled = layer.resize(scaled_size, resample=Image.Resampling.BICUBIC)
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if scale_factor >= 1.0:
        left = (scaled.width - width) // 2
        top = (scaled.height - height) // 2
        return scaled.crop((left, top, left + width, top + height))
    left = (width - scaled.width) // 2
    top = (height - scaled.height) // 2
    result.alpha_composite(scaled, (left, top))
    return result


def render_full_image(
    text: str,
    font_path: str | Path,
    config: dict[str, Any],
    *,
    random_seed: int,
    applied_font_size: int | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one readable full-dataset image with recorded mild effects."""
    effects = plan_full_effects(config, random_seed)
    width = int(config["image_width"])
    height = int(config["image_height"])
    background = tuple(effects["background_rgb"])
    foreground = tuple(effects["foreground_rgb"])
    font_path = Path(font_path)

    text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    if applied_font_size is None:
        font, text_width, bbox = _fit_full_font(
            draw,
            font_path,
            text,
            int(effects["requested_font_size"]),
            int(effects["letter_spacing_px"]),
            int(width * 0.78),
            int(height * 0.52),
        )
    else:
        font = _load_readable_font(font_path, applied_font_size)
        text_width, text_height, bbox = _text_size(
            draw,
            text,
            font,
            int(effects["letter_spacing_px"]),
        )
        if text_width > int(width * 0.78) or text_height > int(height * 0.52):
            raise ValueError(
                f"Balanced font size {applied_font_size} does not safely fit '{text}'."
            )
    text_height = bbox[3] - bbox[1]
    x = (
        (width - text_width) / 2
        + int(effects["horizontal_shift_px"])
    )
    y = (
        (height - text_height) / 2
        - bbox[1]
        + int(effects["vertical_shift_px"])
    )
    _draw_text_with_spacing(
        draw,
        (x, y),
        text,
        font,
        (*foreground, 255),
        int(effects["letter_spacing_px"]),
    )
    text_layer = _scale_transparent_layer(text_layer, float(effects["scale_factor"]))
    text_layer = text_layer.rotate(
        float(effects["rotation_degrees"]),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0, 0),
    )

    canvas = Image.new("RGBA", (width, height), (*background, 255))
    canvas = Image.alpha_composite(canvas, text_layer).convert("RGB")
    if effects["blur_radius"] is not None:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=float(effects["blur_radius"])))
    if effects["jpeg_quality"] is not None:
        canvas = jpeg_roundtrip(canvas, int(effects["jpeg_quality"]))
    effects["actual_font_size"] = font.size
    return canvas, effects


def _effect_signature(effects: dict[str, Any]) -> str:
    scheduled = {key: effects[key] for key in EFFECT_SIGNATURE_KEYS}
    return json.dumps(scheduled, sort_keys=True)


def validate_category_effect_balance(manifest: pd.DataFrame) -> pd.DataFrame:
    """Require every category to receive the same multiset of planned effects."""
    effect_rows: list[dict[str, Any]] = []
    signatures: dict[str, Counter] = {}
    text_counts: dict[str, Counter] = {}
    for category in FONT_CATEGORIES:
        group = manifest.loc[manifest["category"] == category]
        parsed = [json.loads(value) for value in group["applied_effects"]]
        signatures[category] = Counter(_effect_signature(item) for item in parsed)
        text_counts[category] = Counter(group["rendered_text"])
        effect_rows.append(
            {
                "category": category,
                "images": len(group),
                "light_background": sum(item["background"] == "light" for item in parsed),
                "dark_background": sum(item["background"] == "dark" for item in parsed),
                "soft_contrast": sum(item["contrast_style"] == "soft" for item in parsed),
                "strong_contrast": sum(item["contrast_style"] == "strong" for item in parsed),
                "letter_spacing_applied": sum(item["letter_spacing_px"] != 0 for item in parsed),
                "mild_rotation_applied": sum(
                    abs(item["rotation_degrees"]) > 0 for item in parsed
                ),
                "mild_scaling_applied": sum(
                    item["scale_factor"] != 1.0 for item in parsed
                ),
                "horizontal_shift_applied": sum(
                    item["horizontal_shift_px"] != 0 for item in parsed
                ),
                "vertical_shift_applied": sum(
                    item["vertical_shift_px"] != 0 for item in parsed
                ),
                "mild_blur": sum(item["blur_radius"] is not None for item in parsed),
                "jpeg_compression": sum(item["jpeg_quality"] is not None for item in parsed),
                "mean_actual_font_size": round(
                    sum(item["actual_font_size"] for item in parsed) / len(parsed),
                    3,
                ),
                "minimum_luminance_difference": min(
                    item["luminance_difference"] for item in parsed
                ),
            }
        )

    reference_category = FONT_CATEGORIES[0]
    for category in FONT_CATEGORIES[1:]:
        if signatures[category] != signatures[reference_category]:
            raise AssertionError(
                f"Category-dependent effect imbalance found for '{category}'."
            )
        if text_counts[category] != text_counts[reference_category]:
            raise AssertionError(
                f"Category-dependent rendered-text imbalance found for '{category}'."
            )
    return pd.DataFrame(effect_rows)


def validate_full_dataset_structure(
    frozen_split: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Validate counts, frozen assignments, leakage, text, and effect balance."""
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Full manifest is missing columns: {sorted(missing)}")
    if len(manifest) != FULL_IMAGE_COUNT:
        raise AssertionError(f"Expected {FULL_IMAGE_COUNT} images; found {len(manifest)}.")
    if manifest["image_path"].duplicated().any():
        raise AssertionError("Full manifest contains duplicate image paths.")

    per_family = manifest.groupby("family").size()
    if len(per_family) != FULL_FAMILY_COUNT or not per_family.eq(FULL_IMAGES_PER_FAMILY).all():
        raise AssertionError("Every one of the 90 frozen families must have exactly 40 images.")
    expected_categories = {category: 720 for category in FONT_CATEGORIES}
    actual_categories = manifest.groupby("category").size().to_dict()
    if actual_categories != expected_categories:
        raise AssertionError(f"Expected 720 images per category; found {actual_categories}.")
    expected_splits = {"train": 2400, "validation": 600, "test": 600}
    actual_splits = manifest.groupby("split").size().to_dict()
    if actual_splits != expected_splits:
        raise AssertionError(f"Expected split counts {expected_splits}; found {actual_splits}.")
    assert_no_family_leakage(manifest)

    expected_assignments = (
        frozen_split[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values("family")
        .reset_index(drop=True)
    )
    actual_assignments = (
        manifest[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values("family")
        .reset_index(drop=True)
    )
    if not actual_assignments.equals(expected_assignments):
        raise AssertionError("Full dataset family assignments do not match the frozen split CSV.")
    return validate_category_effect_balance(manifest)


def validate_full_image_files(
    manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, int]:
    """Open every listed image and reject missing, empty, blank, or unlisted files."""
    root_path = Path(root) if root is not None else project_root()
    image_root = Path(output_dir) / "images"
    manifest_paths: list[Path] = []
    blank_images: list[str] = []
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Validating full images",
    ):
        image_path = _resolve_path(row.image_path, root_path).resolve()
        manifest_paths.append(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Full manifest path does not exist: {image_path}")
        if not image_path.is_relative_to(image_root.resolve()):
            raise AssertionError(f"Full image is outside the full dataset folder: {image_path}")
        if image_path.stat().st_size == 0:
            raise AssertionError(f"Full image file is empty: {image_path}")
        with Image.open(image_path) as opened:
            opened.verify()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            if f"{image.width}x{image.height}" != str(row.image_size):
                raise AssertionError(f"Image size does not match the manifest: {image_path}")
            minimum, maximum = ImageStat.Stat(image.convert("L")).extrema[0]
            if maximum == minimum:
                blank_images.append(str(image_path))
    if blank_images:
        raise AssertionError(f"Fully blank full-dataset images found: {blank_images[:5]}")

    disk_paths = {path.resolve() for path in image_root.rglob("*.png")}
    expected_paths = set(manifest_paths)
    if disk_paths != expected_paths:
        raise AssertionError(
            "The full image folder and manifest differ: "
            f"{len(disk_paths)} files on disk, {len(expected_paths)} listed."
        )
    return {
        "manifest_paths_existing": len(manifest_paths),
        "images_opened_successfully": len(manifest_paths),
        "empty_image_files": 0,
        "fully_blank_images": 0,
        "unlisted_image_files": 0,
    }


def _contact_sheet_sample(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = manifest.loc[manifest["image_index"].isin({0, 20})].copy()
    expected = len(FONT_CATEGORIES) * 18 * 2
    if len(selected) != expected:
        raise AssertionError(f"Expected {expected} contact-sheet samples; found {len(selected)}.")
    return selected


def generate_full_dataset(
    split_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    effect_balance_path: str | Path,
    contact_sheet_dir: str | Path,
    *,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    """Generate 3,600 images from the frozen split without assigning families."""
    root_path = Path(root) if root is not None else project_root()
    resolved_split = _resolve_path(split_path, root_path)
    resolved_config = _resolve_path(config_path, root_path)
    output_path = _resolve_path(output_dir, root_path)
    resolved_manifest = _resolve_path(manifest_path, root_path)
    resolved_summary = _resolve_path(summary_path, root_path)
    resolved_effect_balance = _resolve_path(effect_balance_path, root_path)
    resolved_contact_sheets = _resolve_path(contact_sheet_dir, root_path)

    split_hash_before = _sha256(resolved_split)
    frozen = load_frozen_family_split(resolved_split, root=root_path).copy()
    config = validate_full_config(load_json(resolved_config))
    frozen["_effect_rank"] = frozen.groupby("category").cumcount()
    font_size_schedule = balanced_font_size_schedule(
        frozen,
        config,
        root=root_path,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    image_root = output_path / "images"
    rows: list[dict[str, Any]] = []

    for family_index, row in tqdm(
        frozen.iterrows(),
        total=len(frozen),
        desc="Generating full families",
    ):
        family = str(row["family"])
        category = str(row["category"])
        split = str(row["split"])
        source_font = str(row["path"])
        resolved_font = _resolve_path(source_font, root_path)
        if not resolved_font.is_file():
            raise FileNotFoundError(f"Frozen font file does not exist for {family}: {resolved_font}")

        effect_rank = int(row["_effect_rank"])
        phrases = _phrases_for_rank(config, effect_rank)
        family_slug = safe_slug(family)
        family_output = image_root / split / category / family_slug
        family_output.mkdir(parents=True, exist_ok=True)

        for image_index, text in enumerate(phrases):
            random_seed = FULL_SEED * 2_000_003 + effect_rank * 10_007 + image_index
            try:
                image, effects = render_full_image(
                    text,
                    resolved_font,
                    config,
                    random_seed=random_seed,
                    applied_font_size=font_size_schedule[(effect_rank, image_index)],
                )
            except Exception as error:
                raise ValueError(
                    f"Could not open and render frozen font '{family}' at {resolved_font}: {error}"
                ) from error
            image_path = family_output / f"{family_slug}_{image_index:04d}.png"
            image.save(image_path, format="PNG", optimize=True)
            rows.append(
                {
                    "image_path": _portable_path(image_path, root_path),
                    "family": family,
                    "category": category,
                    "split": split,
                    "rendered_text": text,
                    "source_font": source_font,
                    "random_seed": random_seed,
                    "image_size": f"{image.width}x{image.height}",
                    "applied_effects": json.dumps(effects, sort_keys=True),
                    "image_index": image_index,
                }
            )

    manifest = pd.DataFrame(rows)
    effect_balance = validate_full_dataset_structure(frozen, manifest)
    file_results = validate_full_image_files(
        manifest,
        output_dir=output_path,
        root=root_path,
    )
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(resolved_manifest, index=False)
    resolved_effect_balance.parent.mkdir(parents=True, exist_ok=True)
    effect_balance.to_csv(resolved_effect_balance, index=False)
    contact_sheets = create_category_contact_sheets(
        _contact_sheet_sample(manifest),
        resolved_contact_sheets,
        root=root_path,
        columns=int(config.get("contact_sheet_columns", 6)),
        index_column="image_index",
        title_suffix="full-dataset samples",
    )

    split_hash_after = _sha256(resolved_split)
    if split_hash_after != split_hash_before:
        raise AssertionError("The frozen family split CSV changed during full dataset generation.")
    summary = {
        "status": "passed",
        "purpose": "Final FontSense image dataset generation and data validation; no model results.",
        "seed": FULL_SEED,
        "families_total": int(manifest["family"].nunique()),
        "images_per_family": FULL_IMAGES_PER_FAMILY,
        "images_total": len(manifest),
        "families_per_category": {
            key: int(value)
            for key, value in frozen.groupby("category")["family"].nunique().to_dict().items()
        },
        "images_per_category": {
            key: int(value)
            for key, value in manifest.groupby("category").size().to_dict().items()
        },
        "images_per_split": {
            key: int(value)
            for key, value in manifest.groupby("split").size().to_dict().items()
        },
        "family_overlap_count": 0,
        **file_results,
        "font_files_rendered": int(manifest["source_font"].nunique()),
        "category_effect_balance": "passed_exact_schedule_match",
        "category_text_balance": "passed_exact_distribution_match",
        "minimum_luminance_difference": float(
            effect_balance["minimum_luminance_difference"].min()
        ),
        "frozen_split_file": _portable_path(resolved_split, root_path),
        "frozen_split_sha256": split_hash_after,
        "full_config_file": _portable_path(resolved_config, root_path),
        "full_config_sha256": _sha256(resolved_config),
        "full_image_root": _portable_path(image_root, root_path),
        "full_manifest_file": _portable_path(resolved_manifest, root_path),
        "effect_balance_file": _portable_path(resolved_effect_balance, root_path),
        "contact_sheets": [
            _portable_path(path, root_path) for path in contact_sheets
        ],
        "model_training_performed": False,
    }
    save_json(summary, resolved_summary)
    return manifest, summary, contact_sheets


def _sample_image_hashes(
    manifest: pd.DataFrame,
    *,
    root: Path,
    per_category: int = 6,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for category_index, category in enumerate(FONT_CATEGORIES):
        group = manifest.loc[manifest["category"] == category]
        sample = group.sample(
            n=per_category,
            random_state=FULL_SEED + category_index,
        )
        for path in sorted(sample["image_path"]):
            resolved = _resolve_path(path, root)
            hashes[str(path)] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return hashes


def verify_full_reproducibility(
    generation_args: dict[str, Any],
    reproducibility_path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Rerun generation and compare the manifest plus 30 deterministic image hashes."""
    root_path = Path(root) if root is not None else project_root()
    manifest_path = _resolve_path(generation_args["manifest_path"], root_path)
    summary_path = _resolve_path(generation_args["summary_path"], root_path)
    before_manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest_hash_before = _sha256(manifest_path)
    image_hashes_before = _sample_image_hashes(before_manifest, root=root_path)

    after_manifest, _, _ = generate_full_dataset(**generation_args, root=root_path)
    manifest_hash_after = _sha256(manifest_path)
    image_hashes_after = _sample_image_hashes(after_manifest, root=root_path)
    mismatched_images = sorted(
        path
        for path, before_hash in image_hashes_before.items()
        if image_hashes_after.get(path) != before_hash
    )
    if manifest_hash_before != manifest_hash_after:
        raise AssertionError("The full manifest changed after a same-seed rerun.")
    if mismatched_images:
        raise AssertionError(
            f"Sample image hashes changed after a same-seed rerun: {mismatched_images[:5]}"
        )

    report = {
        "status": "passed",
        "seed": FULL_SEED,
        "reruns_compared": 2,
        "manifest_sha256_before": manifest_hash_before,
        "manifest_sha256_after": manifest_hash_after,
        "manifest_identical": True,
        "sample_images_checked": len(image_hashes_before),
        "sample_image_hashes_identical": True,
        "sample_image_hashes": image_hashes_after,
    }
    resolved_reproducibility = _resolve_path(reproducibility_path, root_path)
    save_json(report, resolved_reproducibility)
    summary = load_json(summary_path)
    summary["reproducibility_status"] = "passed"
    summary["reproducibility_check_file"] = _portable_path(
        resolved_reproducibility,
        root_path,
    )
    summary["same_seed_manifest_identical"] = True
    summary["same_seed_sample_images_identical"] = True
    summary["same_seed_sample_images_checked"] = len(image_hashes_before)
    save_json(summary, summary_path)
    return report


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Generate and validate the full 3,600-image FontSense dataset."
    )
    parser.add_argument(
        "--split",
        default=str(root / "data/interim/google_fonts_final_family_split.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "data/processed/fontsense_full"),
    )
    parser.add_argument(
        "--config",
        default=str(root / "config/full_dataset.json"),
    )
    parser.add_argument(
        "--manifest",
        default=str(root / "reports/dataset/full_manifest.csv"),
    )
    parser.add_argument(
        "--summary",
        default=str(root / "reports/dataset/full_validation_summary.json"),
    )
    parser.add_argument(
        "--effect-balance",
        default=str(root / "reports/dataset/full_effect_balance.csv"),
    )
    parser.add_argument(
        "--contact-sheet-dir",
        default=str(root / "reports/dataset/contact_sheets"),
    )
    parser.add_argument(
        "--reproducibility-report",
        default=str(root / "reports/dataset/full_reproducibility_check.json"),
    )
    parser.add_argument(
        "--verify-reproducibility",
        action="store_true",
        help="Generate a second time and compare the manifest plus 30 image hashes.",
    )
    args = parser.parse_args()
    generation_args = {
        "split_path": args.split,
        "output_dir": args.output_dir,
        "config_path": args.config,
        "manifest_path": args.manifest,
        "summary_path": args.summary,
        "effect_balance_path": args.effect_balance,
        "contact_sheet_dir": args.contact_sheet_dir,
    }
    _, summary, contact_sheets = generate_full_dataset(**generation_args, root=root)
    print(json.dumps(summary, indent=2))
    print(f"Created {len(contact_sheets)} full-dataset contact sheets.")
    if args.verify_reproducibility:
        report = verify_full_reproducibility(
            generation_args,
            args.reproducibility_report,
            root=root,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
