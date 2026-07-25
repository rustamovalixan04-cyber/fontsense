from __future__ import annotations

import argparse
import io
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from tqdm import tqdm

from .split import SplitRatios, family_level_split
from .utils import load_json, project_root, set_seed


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def fit_font(font_path: str, text: str, max_width: int, max_height: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    start = rng.randint(34, 64)
    for size in range(start, 13, -2):
        font = ImageFont.truetype(font_path, size=size)
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= max_width and height <= max_height:
            return font
    return ImageFont.truetype(font_path, size=14)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def render_text_image(
    text: str,
    font_path: str,
    width: int = 224,
    height: int = 96,
    seed: int = 42,
    augment: bool = True,
) -> Image.Image:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if augment:
        light_background = rng.random() < 0.82
        if light_background:
            background = tuple(rng.randint(225, 255) for _ in range(3))
            foreground = tuple(rng.randint(0, 70) for _ in range(3))
        else:
            background = tuple(rng.randint(0, 55) for _ in range(3))
            foreground = tuple(rng.randint(205, 255) for _ in range(3))
    else:
        background, foreground = (255, 255, 255), (0, 0, 0)

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    font = fit_font(font_path, text, int(width * 0.90), int(height * 0.62), rng)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]
    if augment:
        x += rng.randint(-8, 8)
        y += rng.randint(-5, 5)
    draw.text((x, y), text, fill=foreground, font=font)

    if augment:
        angle = rng.uniform(-4.0, 4.0)
        canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=background)
        if rng.random() < 0.45:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 1.0)))
        if rng.random() < 0.45:
            canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.75, 1.30))
        if rng.random() < 0.35:
            array = np.asarray(canvas).astype(np.float32)
            noise = np_rng.normal(0.0, rng.uniform(1.0, 7.0), size=array.shape)
            array = np.clip(array + noise, 0, 255).astype(np.uint8)
            canvas = Image.fromarray(array, mode="RGB")
        if rng.random() < 0.35:
            canvas = jpeg_roundtrip(canvas, quality=rng.randint(48, 88))
    return canvas


def generate_dataset(
    font_manifest_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    images_per_family: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    config = load_json(config_path)
    width = int(config["image_width"])
    height = int(config["image_height"])
    phrases = list(config["phrases"])
    images_per_family = images_per_family or int(config["images_per_family"])

    fonts = pd.read_csv(font_manifest_path)
    required_columns = {"family", "category", "path"}
    missing_columns = required_columns - set(fonts.columns)
    if missing_columns:
        raise ValueError(f"Font manifest is missing columns: {sorted(missing_columns)}")
    if "usable" in fonts.columns:
        usable_mask = fonts["usable"].astype(str).str.lower().isin({"true", "1"})
        fonts = fonts.loc[usable_mask].copy()
    if fonts.empty:
        raise ValueError("Font manifest contains zero usable fonts; run the font audit before dataset generation.")
    existing_path_mask = fonts["path"].fillna("").astype(str).map(lambda path: Path(path).is_file())
    fonts = fonts.loc[existing_path_mask].copy()
    if fonts.empty:
        raise ValueError("Font manifest contains zero usable font files on this system.")
    fonts = fonts.drop_duplicates(["family", "category"]).reset_index(drop=True)
    family_manifest = family_level_split(
        fonts,
        SplitRatios(config["train_ratio"], config["validation_ratio"], config["test_ratio"]),
        seed=seed,
    )

    output_dir = Path(output_dir)
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for family_index, row in tqdm(family_manifest.iterrows(), total=len(family_manifest), desc="Generating families"):
        family = str(row["family"])
        category = str(row["category"])
        split = str(row["split"])
        font_path = str(row["path"])
        family_slug = safe_slug(family)
        folder = image_root / split / category / family_slug
        folder.mkdir(parents=True, exist_ok=True)
        for image_index in range(images_per_family):
            sample_seed = seed * 1_000_003 + family_index * 10_007 + image_index
            rng = random.Random(sample_seed)
            text = rng.choice(phrases)
            image = render_text_image(text, font_path, width, height, sample_seed, augment=True)
            filename = f"{family_slug}_{image_index:04d}.png"
            path = folder / filename
            image.save(path, optimize=True)
            rows.append({
                "image_path": str(path),
                "relative_image_path": str(path.relative_to(output_dir)),
                "family": family,
                "category": category,
                "split": split,
                "text": text,
                "source_font_path": font_path,
                "seed": sample_seed,
            })

    manifest = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    family_manifest.to_csv(output_dir / "families.csv", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return family_manifest, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a leakage-safe synthetic typeface image dataset.")
    parser.add_argument("--font-manifest", required=True)
    parser.add_argument("--output-dir", default=str(project_root() / "data/processed/fontsense"))
    parser.add_argument("--config", default=str(project_root() / "config/default.json"))
    parser.add_argument("--images-per-family", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    families, images = generate_dataset(args.font_manifest, args.output_dir, args.config, args.images_per_family, args.seed)
    print("Families by category and split:")
    print(families.groupby(["category", "split"]).size().unstack(fill_value=0))
    print(f"Generated {len(images):,} images at {args.output_dir}")


if __name__ == "__main__":
    main()
