from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat
from tqdm import tqdm

from .generate_dataset import jpeg_roundtrip, safe_slug
from .split import FONT_CATEGORIES, assert_balanced_family_split, assert_no_family_leakage
from .utils import load_json, project_root, save_json

PREVIEW_SEED = 42
PREVIEW_IMAGES_PER_FAMILY = 2
PREVIEW_FAMILY_COUNT = 90
PREVIEW_IMAGE_COUNT = PREVIEW_FAMILY_COUNT * PREVIEW_IMAGES_PER_FAMILY
REQUIRED_SPLIT_COLUMNS = {"family", "category", "split", "path", "source", "license"}
REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "family",
    "category",
    "split",
    "rendered_text",
    "font_file",
    "applied_effects",
}


def _resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _portable_path(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_family_split(
    split_path: str | Path,
    *,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Read and validate the already-created family split without changing it."""
    root_path = Path(root) if root is not None else project_root()
    resolved_split = _resolve_path(split_path, root_path)
    frozen = pd.read_csv(resolved_split, keep_default_na=False)

    missing = REQUIRED_SPLIT_COLUMNS - set(frozen.columns)
    if missing:
        raise ValueError(f"Frozen family split is missing columns: {sorted(missing)}")
    if len(frozen) != PREVIEW_FAMILY_COUNT:
        raise ValueError(
            f"Frozen family split must contain exactly {PREVIEW_FAMILY_COUNT} rows; "
            f"found {len(frozen)}."
        )
    if frozen["family"].duplicated().any():
        duplicates = frozen.loc[frozen["family"].duplicated(), "family"].tolist()
        raise ValueError(f"Frozen family split contains duplicate families: {duplicates[:5]}")
    for column in REQUIRED_SPLIT_COLUMNS:
        if frozen[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Frozen family split contains an empty '{column}' value.")
    if "selection_seed" in frozen.columns:
        seeds = set(pd.to_numeric(frozen["selection_seed"], errors="coerce").dropna().astype(int))
        if seeds != {PREVIEW_SEED}:
            raise ValueError(
                f"Frozen family split must record selection seed {PREVIEW_SEED}; found {sorted(seeds)}."
            )

    assert_balanced_family_split(frozen)
    return frozen


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    letter_spacing: int,
) -> tuple[float, int, tuple[int, int, int, int]]:
    advances = [draw.textlength(character, font=font) for character in text]
    width = sum(advances) + max(0, len(text) - 1) * letter_spacing
    bbox = draw.textbbox((0, 0), text, font=font)
    return width, bbox[3] - bbox[1], bbox


def _load_readable_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    """Open a font and use normal weight when a variable weight axis is available."""
    font = ImageFont.truetype(str(font_path), size=size)
    try:
        axes = font.get_variation_axes()
        values = []
        for axis in axes:
            axis_name = axis["name"].decode("utf-8", errors="ignore").casefold()
            value = axis["default"]
            if "weight" in axis_name:
                value = min(max(400, axis["minimum"]), axis["maximum"])
            values.append(value)
        if values:
            font.set_variation_by_axes(values)
    except (AttributeError, OSError):
        pass
    return font


def _fit_preview_font(
    draw: ImageDraw.ImageDraw,
    font_path: Path,
    text: str,
    requested_size: int,
    letter_spacing: int,
    max_width: int,
    max_height: int,
) -> tuple[ImageFont.FreeTypeFont, float, tuple[int, int, int, int]]:
    for size in range(requested_size, 9, -1):
        font = _load_readable_font(font_path, size)
        text_width, text_height, bbox = _text_size(draw, text, font, letter_spacing)
        if text_width <= max_width and text_height <= max_height:
            return font, text_width, bbox
    raise ValueError(f"Text '{text}' could not be fitted with font file {font_path}.")


def _draw_text_with_spacing(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    letter_spacing: int,
) -> None:
    x, y = position
    for character in text:
        draw.text((x, y), character, fill=fill, font=font)
        x += draw.textlength(character, font=font) + letter_spacing


def _validated_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "seed",
        "images_per_family",
        "image_width",
        "image_height",
        "font_size_min",
        "font_size_max",
        "letter_spacing_min",
        "letter_spacing_max",
        "rotation_degrees",
        "blur_probability",
        "blur_radius_min",
        "blur_radius_max",
        "jpeg_probability",
        "jpeg_quality_min",
        "jpeg_quality_max",
        "dark_background_probability",
        "phrases",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Preview config is missing settings: {sorted(missing)}")
    if int(config["seed"]) != PREVIEW_SEED:
        raise ValueError(f"Preview generation must use fixed seed {PREVIEW_SEED}.")
    if int(config["images_per_family"]) != PREVIEW_IMAGES_PER_FAMILY:
        raise ValueError(
            f"Preview generation must create exactly {PREVIEW_IMAGES_PER_FAMILY} images per family."
        )
    if int(config["image_width"]) <= 0 or int(config["image_height"]) <= 0:
        raise ValueError("Preview image dimensions must be positive.")
    if int(config["font_size_min"]) > int(config["font_size_max"]):
        raise ValueError("font_size_min cannot be larger than font_size_max.")
    if int(config["letter_spacing_min"]) > int(config["letter_spacing_max"]):
        raise ValueError("letter_spacing_min cannot be larger than letter_spacing_max.")
    if len(config["phrases"]) < PREVIEW_IMAGES_PER_FAMILY:
        raise ValueError("Preview config needs at least two different phrases.")
    for key in ("blur_probability", "jpeg_probability", "dark_background_probability"):
        if not 0.0 < float(config[key]) < 1.0:
            raise ValueError(f"{key} must be between 0 and 1 so the effect is used on only some images.")
    return config


def render_preview_image(
    text: str,
    font_path: str | Path,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one readable preview image and return an honest record of its effects."""
    rng = random.Random(seed)
    font_path = Path(font_path)
    width = int(config["image_width"])
    height = int(config["image_height"])
    is_dark = rng.random() < float(config["dark_background_probability"])

    if is_dark:
        background = tuple(rng.randint(8, 42) for _ in range(3))
        foreground = tuple(rng.randint(220, 255) for _ in range(3))
        background_name = "dark"
    else:
        background = tuple(rng.randint(235, 255) for _ in range(3))
        foreground = tuple(rng.randint(0, 55) for _ in range(3))
        background_name = "light"

    letter_spacing = rng.randint(
        int(config["letter_spacing_min"]),
        int(config["letter_spacing_max"]),
    )
    requested_size = rng.randint(int(config["font_size_min"]), int(config["font_size_max"]))
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    font, text_width, bbox = _fit_preview_font(
        draw,
        font_path,
        text,
        requested_size,
        letter_spacing,
        int(width * 0.90),
        int(height * 0.62),
    )
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) / 2 + rng.randint(-3, 3)
    y = (height - text_height) / 2 - bbox[1] + rng.randint(-2, 2)
    _draw_text_with_spacing(draw, (x, y), text, font, foreground, letter_spacing)

    rotation = rng.uniform(-float(config["rotation_degrees"]), float(config["rotation_degrees"]))
    canvas = canvas.rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=background,
    )

    blur_radius: float | None = None
    if rng.random() < float(config["blur_probability"]):
        blur_radius = rng.uniform(
            float(config["blur_radius_min"]),
            float(config["blur_radius_max"]),
        )
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    jpeg_quality: int | None = None
    if rng.random() < float(config["jpeg_probability"]):
        jpeg_quality = rng.randint(
            int(config["jpeg_quality_min"]),
            int(config["jpeg_quality_max"]),
        )
        canvas = jpeg_roundtrip(canvas, jpeg_quality)

    effects = {
        "font_size": font.size,
        "background": background_name,
        "letter_spacing_px": letter_spacing,
        "rotation_degrees": round(rotation, 3),
        "blur_radius": None if blur_radius is None else round(blur_radius, 3),
        "jpeg_quality": jpeg_quality,
    }
    return canvas, effects


def _effect_counts(manifest: pd.DataFrame) -> dict[str, int]:
    effects = [json.loads(value) for value in manifest["applied_effects"]]
    return {
        "light_background": sum(item["background"] == "light" for item in effects),
        "dark_background": sum(item["background"] == "dark" for item in effects),
        "letter_spacing": sum(item["letter_spacing_px"] > 0 for item in effects),
        "mild_rotation": sum(abs(item["rotation_degrees"]) > 0 for item in effects),
        "mild_blur": sum(item["blur_radius"] is not None for item in effects),
        "jpeg_compression": sum(item["jpeg_quality"] is not None for item in effects),
    }


def validate_preview_dataset(
    frozen_split: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the exact count, leakage, path, image, and blank-image checks."""
    root_path = Path(root) if root is not None else project_root()
    output_path = Path(output_dir)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Preview manifest is missing columns: {sorted(missing)}")
    if len(manifest) != PREVIEW_IMAGE_COUNT:
        raise AssertionError(f"Expected {PREVIEW_IMAGE_COUNT} preview images; found {len(manifest)}.")
    if manifest["image_path"].duplicated().any():
        raise AssertionError("Preview manifest contains duplicate image paths.")

    per_family = manifest.groupby("family").size()
    if len(per_family) != PREVIEW_FAMILY_COUNT or not per_family.eq(PREVIEW_IMAGES_PER_FAMILY).all():
        raise AssertionError("Every one of the 90 frozen families must have exactly two preview images.")
    expected_category_counts = {category: 36 for category in FONT_CATEGORIES}
    actual_category_counts = manifest.groupby("category").size().to_dict()
    if actual_category_counts != expected_category_counts:
        raise AssertionError(
            f"Expected 36 images per category; found {actual_category_counts}."
        )
    expected_split_counts = {"train": 120, "validation": 30, "test": 30}
    actual_split_counts = manifest.groupby("split").size().to_dict()
    if actual_split_counts != expected_split_counts:
        raise AssertionError(
            f"Expected split image counts {expected_split_counts}; found {actual_split_counts}."
        )
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
        raise AssertionError("Preview family assignments do not match the frozen split CSV.")

    image_root = output_path / "images"
    manifest_paths: list[Path] = []
    blank_images: list[str] = []
    for row in manifest.itertuples(index=False):
        image_path = _resolve_path(row.image_path, root_path).resolve()
        manifest_paths.append(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Preview manifest path does not exist: {image_path}")
        if not image_path.is_relative_to(image_root.resolve()):
            raise AssertionError(f"Preview image is outside the preview folder: {image_path}")
        if image_path.stat().st_size == 0:
            raise AssertionError(f"Preview image file is empty: {image_path}")
        with Image.open(image_path) as opened:
            opened.verify()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            if image.width == 0 or image.height == 0:
                raise AssertionError(f"Preview image has empty dimensions: {image_path}")
            gray = image.convert("L")
            minimum, maximum = ImageStat.Stat(gray).extrema[0]
            if maximum == minimum:
                blank_images.append(str(image_path))
    if blank_images:
        raise AssertionError(f"Fully blank preview images found: {blank_images[:5]}")

    saved_paths = {path.resolve() for path in image_root.rglob("*.png")}
    expected_paths = set(manifest_paths)
    if saved_paths != expected_paths:
        raise AssertionError(
            "The preview image folder and manifest differ: "
            f"{len(saved_paths)} files on disk, {len(expected_paths)} listed."
        )

    effect_counts = _effect_counts(manifest)
    for effect in ("light_background", "dark_background", "mild_blur", "jpeg_compression"):
        if not 0 < effect_counts[effect] < PREVIEW_IMAGE_COUNT:
            raise AssertionError(f"Effect '{effect}' must be present on only some preview images.")

    return {
        "status": "passed",
        "purpose": "Small preview dataset for data inspection; not final training data or final results.",
        "seed": PREVIEW_SEED,
        "families_total": int(manifest["family"].nunique()),
        "images_per_family": PREVIEW_IMAGES_PER_FAMILY,
        "images_total": len(manifest),
        "families_per_category": {
            key: int(value)
            for key, value in frozen_split.groupby("category")["family"].nunique().to_dict().items()
        },
        "images_per_category": {
            key: int(value) for key, value in actual_category_counts.items()
        },
        "images_per_split": {
            key: int(value) for key, value in actual_split_counts.items()
        },
        "family_overlap_count": 0,
        "manifest_paths_existing": len(manifest_paths),
        "images_opened_successfully": len(manifest_paths),
        "empty_image_files": 0,
        "fully_blank_images": 0,
        "font_files_rendered": int(manifest["font_file"].nunique()),
        "effect_counts": effect_counts,
    }


def _short_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if draw.textlength(label, font=font) <= max_width:
        return label
    shortened = label
    while shortened and draw.textlength(f"{shortened}...", font=font) > max_width:
        shortened = shortened[:-1]
    return f"{shortened}..."


def create_category_contact_sheets(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    *,
    root: str | Path | None = None,
    columns: int = 6,
) -> list[Path]:
    """Create one 36-image contact sheet for each category."""
    root_path = Path(root) if root is not None else project_root()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_font = ImageFont.load_default()
    saved: list[Path] = []

    for category in FONT_CATEGORIES:
        group = manifest.loc[manifest["category"] == category].copy()
        split_order = {"train": 0, "validation": 1, "test": 2}
        group["_split_order"] = group["split"].map(split_order)
        group = group.sort_values(["_split_order", "family", "preview_index"])
        first_image_path = _resolve_path(group.iloc[0]["image_path"], root_path)
        with Image.open(first_image_path) as first:
            image_width, image_height = first.size

        rows = (len(group) + columns - 1) // columns
        padding = 6
        title_height = 28
        label_height = 24
        cell_width = image_width + padding * 2
        cell_height = image_height + label_height + padding * 2
        sheet = Image.new(
            "RGB",
            (columns * cell_width, title_height + rows * cell_height),
            (232, 234, 238),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (padding, 8),
            f"{category}: {len(group)} preview images",
            fill=(20, 24, 32),
            font=label_font,
        )

        for tile_index, row in enumerate(group.itertuples(index=False)):
            column_index = tile_index % columns
            row_index = tile_index // columns
            x = column_index * cell_width + padding
            y = title_height + row_index * cell_height + padding
            image_path = _resolve_path(row.image_path, root_path)
            with Image.open(image_path) as opened:
                sheet.paste(opened.convert("RGB"), (x, y))
            label = _short_label(
                draw,
                f"{row.family} | {row.split}",
                label_font,
                image_width,
            )
            draw.text(
                (x, y + image_height + 5),
                label,
                fill=(25, 28, 36),
                font=label_font,
            )

        contact_sheet_path = output_path / f"{category}.png"
        sheet.save(contact_sheet_path, optimize=True)
        saved.append(contact_sheet_path)
    return saved


def generate_preview_dataset(
    split_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
    contact_sheet_dir: str | Path,
    *,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    """Generate the fixed 180-image preview without creating a new family split."""
    root_path = Path(root) if root is not None else project_root()
    resolved_split = _resolve_path(split_path, root_path)
    resolved_config = _resolve_path(config_path, root_path)
    output_path = _resolve_path(output_dir, root_path)
    resolved_manifest = _resolve_path(manifest_path, root_path)
    resolved_summary = _resolve_path(summary_path, root_path)
    resolved_contact_sheets = _resolve_path(contact_sheet_dir, root_path)

    split_hash_before = _sha256(resolved_split)
    frozen = load_frozen_family_split(resolved_split, root=root_path)
    config = _validated_config(load_json(resolved_config))
    output_path.mkdir(parents=True, exist_ok=True)
    image_root = output_path / "images"
    rows: list[dict[str, Any]] = []

    for family_index, row in tqdm(
        frozen.iterrows(),
        total=len(frozen),
        desc="Generating preview families",
    ):
        family = str(row["family"])
        category = str(row["category"])
        split = str(row["split"])
        font_file = str(row["path"])
        resolved_font = _resolve_path(font_file, root_path)
        if not resolved_font.is_file():
            raise FileNotFoundError(f"Frozen font file does not exist for {family}: {resolved_font}")

        family_slug = safe_slug(family)
        family_output = image_root / split / category / family_slug
        family_output.mkdir(parents=True, exist_ok=True)
        phrase_rng = random.Random(PREVIEW_SEED * 1009 + family_index)
        phrases = phrase_rng.sample(list(config["phrases"]), PREVIEW_IMAGES_PER_FAMILY)

        for preview_index, text in enumerate(phrases):
            sample_seed = PREVIEW_SEED * 1_000_003 + family_index * 10_007 + preview_index
            try:
                image, effects = render_preview_image(
                    text,
                    resolved_font,
                    config,
                    seed=sample_seed,
                )
            except Exception as error:
                raise ValueError(
                    f"Could not open and render frozen font '{family}' at {resolved_font}: {error}"
                ) from error
            image_path = family_output / f"{family_slug}_{preview_index:02d}.png"
            image.save(image_path, format="PNG", optimize=True)
            rows.append(
                {
                    "image_path": _portable_path(image_path, root_path),
                    "family": family,
                    "category": category,
                    "split": split,
                    "rendered_text": text,
                    "font_file": font_file,
                    "applied_effects": json.dumps(effects, sort_keys=True),
                    "seed": sample_seed,
                    "preview_index": preview_index,
                }
            )

    manifest = pd.DataFrame(rows)
    summary = validate_preview_dataset(
        frozen,
        manifest,
        output_dir=output_path,
        root=root_path,
    )
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(resolved_manifest, index=False)
    contact_sheets = create_category_contact_sheets(
        manifest,
        resolved_contact_sheets,
        root=root_path,
        columns=int(config.get("contact_sheet_columns", 6)),
    )

    split_hash_after = _sha256(resolved_split)
    if split_hash_after != split_hash_before:
        raise AssertionError("The frozen family split CSV changed during preview generation.")
    summary.update(
        {
            "frozen_split_file": _portable_path(resolved_split, root_path),
            "frozen_split_sha256": split_hash_after,
            "preview_config_file": _portable_path(resolved_config, root_path),
            "preview_image_root": _portable_path(image_root, root_path),
            "preview_manifest_file": _portable_path(resolved_manifest, root_path),
            "contact_sheets": [
                _portable_path(path, root_path) for path in contact_sheets
            ],
        }
    )
    save_json(summary, resolved_summary)
    return manifest, summary, contact_sheets


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Generate and validate the small frozen-split FontSense preview dataset."
    )
    parser.add_argument(
        "--split",
        default=str(root / "data/interim/google_fonts_final_family_split.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "data/processed/fontsense_preview"),
    )
    parser.add_argument(
        "--config",
        default=str(root / "config/preview.json"),
    )
    parser.add_argument(
        "--manifest",
        default=str(root / "reports/preview/preview_manifest.csv"),
    )
    parser.add_argument(
        "--summary",
        default=str(root / "reports/preview/preview_validation_summary.json"),
    )
    parser.add_argument(
        "--contact-sheet-dir",
        default=str(root / "reports/preview/contact_sheets"),
    )
    args = parser.parse_args()

    _, summary, contact_sheets = generate_preview_dataset(
        args.split,
        args.output_dir,
        args.config,
        args.manifest,
        args.summary,
        args.contact_sheet_dir,
        root=root,
    )
    print(json.dumps(summary, indent=2))
    print(f"Created {len(contact_sheets)} category contact sheets.")


if __name__ == "__main__":
    main()
