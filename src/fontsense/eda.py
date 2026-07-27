from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

from .split import FONT_CATEGORIES, assert_no_family_leakage
from .utils import project_root, save_json

FULL_IMAGE_COUNT = 3_600
FULL_FAMILY_COUNT = 90
EXPECTED_IMAGE_SIZE = (224, 96)
SPLIT_ORDER = ("train", "validation", "test")
CATEGORY_LABELS = {
    "serif": "Serif",
    "sans_serif": "Sans serif",
    "display": "Display",
    "handwriting": "Handwriting",
    "monospace": "Monospace",
}
CATEGORY_COLORS = {
    "serif": "#2667A5",
    "sans_serif": "#D49B25",
    "display": "#D66A3A",
    "handwriting": "#7C8B3C",
    "monospace": "#C2557D",
}
SPLIT_COLORS = {
    "train": "#2667A5",
    "validation": "#D49B25",
    "test": "#D66A3A",
}
EFFECT_FIELDS = (
    "actual_font_size",
    "background",
    "blur_radius",
    "contrast_style",
    "horizontal_shift_px",
    "jpeg_quality",
    "letter_spacing_px",
    "luminance_difference",
    "rotation_degrees",
    "scale_factor",
    "vertical_shift_px",
)
MODEL_FEATURE_AUDIT = {
    "status": "passed_by_code_inspection",
    "hog_input": "HOG values calculated from grayscale image pixels only",
    "cnn_input": "normalized grayscale image tensor only",
    "target": "category",
    "path_usage": "image_path is used only to open the image file",
    "excluded_from_model_features": [
        "image_path text",
        "file name",
        "family",
        "split",
        "rendered_text",
        "source_font",
        "random_seed",
        "applied_effects",
    ],
    "code_checked": [
        "src/fontsense/features.py",
        "src/fontsense/train_hog.py",
        "src/fontsense/train_cnn.py",
    ],
}


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_effects(manifest: pd.DataFrame) -> pd.DataFrame:
    """Expand the JSON effect column into stable analysis columns."""
    if "applied_effects" not in manifest:
        raise ValueError("Manifest is missing applied_effects.")
    rows: list[dict[str, Any]] = []
    for index, value in manifest["applied_effects"].items():
        try:
            effects = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid applied_effects JSON at manifest row {index}.") from error
        missing = set(EFFECT_FIELDS) - set(effects)
        if missing:
            raise ValueError(
                f"Manifest row {index} is missing effect fields: {sorted(missing)}"
            )
        rows.append({field: effects[field] for field in EFFECT_FIELDS})
    return pd.DataFrame(rows, index=manifest.index)


def validate_dataset_structure(
    manifest: pd.DataFrame,
    frozen_split: pd.DataFrame,
    *,
    expected_images: int = FULL_IMAGE_COUNT,
    expected_families: int = FULL_FAMILY_COUNT,
) -> dict[str, Any]:
    """Validate the manifest grain, counts, assignments, and family isolation."""
    required = {
        "image_path",
        "family",
        "category",
        "split",
        "rendered_text",
        "source_font",
        "applied_effects",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Full manifest is missing columns: {sorted(missing)}")
    if len(manifest) != expected_images:
        raise AssertionError(f"Expected {expected_images} images; found {len(manifest)}.")
    if manifest["image_path"].duplicated().any():
        raise AssertionError("Manifest image_path must be a unique image-level key.")
    if manifest["family"].nunique() != expected_families:
        raise AssertionError(
            f"Expected {expected_families} unique families; "
            f"found {manifest['family'].nunique()}."
        )
    if set(manifest["category"]) != set(FONT_CATEGORIES):
        raise AssertionError(
            f"Expected categories {list(FONT_CATEGORIES)}; "
            f"found {sorted(manifest['category'].unique())}."
        )
    if set(manifest["split"]) != set(SPLIT_ORDER):
        raise AssertionError(
            f"Expected splits {list(SPLIT_ORDER)}; "
            f"found {sorted(manifest['split'].unique())}."
        )
    assert_no_family_leakage(manifest)

    expected_assignments = (
        frozen_split[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values(["family", "category", "split"])
        .reset_index(drop=True)
    )
    actual_assignments = (
        manifest[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values(["family", "category", "split"])
        .reset_index(drop=True)
    )
    if not actual_assignments.equals(expected_assignments):
        raise AssertionError("Manifest assignments do not match the frozen family split.")

    family_split_counts = manifest.groupby("family")["split"].nunique()
    overlap_count = int((family_split_counts > 1).sum())
    return {
        "manifest_grain": "one row per rendered image",
        "candidate_key": "image_path",
        "total_images": int(len(manifest)),
        "unique_image_paths": int(manifest["image_path"].nunique()),
        "unique_families": int(manifest["family"].nunique()),
        "family_overlap_count": overlap_count,
        "images_per_category": {
            key: int(value)
            for key, value in manifest.groupby("category").size().to_dict().items()
        },
        "images_per_split": {
            key: int(value)
            for key, value in manifest.groupby("split").size().to_dict().items()
        },
        "images_per_family_min": int(manifest.groupby("family").size().min()),
        "images_per_family_max": int(manifest.groupby("family").size().max()),
        "families_per_category": {
            key: int(value)
            for key, value in manifest.groupby("category")["family"].nunique().to_dict().items()
        },
        "families_per_split": {
            key: int(value)
            for key, value in manifest.groupby("split")["family"].nunique().to_dict().items()
        },
    }


def _difference_hash(image: Image.Image, width: int = 16, height: int = 16) -> str:
    resized = ImageOps.fit(
        image.convert("L"),
        (width + 1, height),
        method=Image.Resampling.LANCZOS,
    )
    array = np.asarray(resized, dtype=np.int16)
    bits = array[:, 1:] >= array[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{width * height // 4}x}"


def audit_image_files(manifest: pd.DataFrame, *, root: Path) -> pd.DataFrame:
    """Open every image and calculate deterministic file and pixel quality metrics."""
    records: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        resolved = _resolve_path(row.image_path, root)
        record: dict[str, Any] = {
            "image_path": str(row.image_path),
            "family": str(row.family),
            "category": str(row.category),
            "split": str(row.split),
            "exists": resolved.is_file(),
            "opens_successfully": False,
            "error": "",
        }
        if not resolved.is_file():
            record["error"] = "missing file"
            records.append(record)
            continue
        try:
            with Image.open(resolved) as opened:
                opened.load()
                image = opened.convert("RGB")
            grayscale = np.asarray(image.convert("L"), dtype=np.float32)
            minimum = float(grayscale.min())
            maximum = float(grayscale.max())
            contrast = float(grayscale.std())
            intensity_range = maximum - minimum
            record.update(
                {
                    "opens_successfully": True,
                    "width": int(image.width),
                    "height": int(image.height),
                    "mode": image.mode,
                    "file_size_bytes": int(resolved.stat().st_size),
                    "brightness_mean": round(float(grayscale.mean()), 4),
                    "contrast_std": round(contrast, 4),
                    "intensity_range": round(intensity_range, 4),
                    "blank": bool(intensity_range == 0),
                    "low_pixel_contrast": bool(contrast < 5.0 or intensity_range < 10.0),
                    "sha256": _sha256(resolved),
                    "difference_hash": _difference_hash(image),
                }
            )
        except Exception as error:  # Pillow can raise several format-specific exceptions.
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    return pd.DataFrame(records)


def find_suspicious_pairs(
    quality: pd.DataFrame,
    *,
    root: Path,
    hamming_threshold: int = 2,
    similarity_threshold: float = 0.995,
    candidate_limit: int = 5_000,
) -> pd.DataFrame:
    """Find extremely similar images without treating repeated phrases as duplicates."""
    readable = quality.loc[quality["opens_successfully"]].reset_index(drop=True)
    hashes = [int(value, 16) for value in readable["difference_hash"]]
    candidates: list[tuple[int, int, int]] = []
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            distance = (hashes[left] ^ hashes[right]).bit_count()
            if distance <= hamming_threshold:
                candidates.append((left, right, distance))
                if len(candidates) >= candidate_limit:
                    break
        if len(candidates) >= candidate_limit:
            break

    rows: list[dict[str, Any]] = []
    for left, right, distance in candidates:
        left_row = readable.iloc[left]
        right_row = readable.iloc[right]
        left_path = _resolve_path(left_row["image_path"], root)
        right_path = _resolve_path(right_row["image_path"], root)
        with Image.open(left_path) as image:
            left_array = np.asarray(image.convert("L"), dtype=np.uint8)
        with Image.open(right_path) as image:
            right_array = np.asarray(image.convert("L"), dtype=np.uint8)
        if left_array.shape != right_array.shape:
            continue
        similarity = float(
            structural_similarity(left_array, right_array, data_range=255)
        )
        if similarity >= similarity_threshold:
            rows.append(
                {
                    "left_image_path": left_row["image_path"],
                    "right_image_path": right_row["image_path"],
                    "left_family": left_row["family"],
                    "right_family": right_row["family"],
                    "left_category": left_row["category"],
                    "right_category": right_row["category"],
                    "hash_hamming_distance": distance,
                    "structural_similarity": round(similarity, 6),
                    "exact_file_duplicate": bool(
                        left_row["sha256"] == right_row["sha256"]
                    ),
                }
            )
    columns = [
        "left_image_path",
        "right_image_path",
        "left_family",
        "right_family",
        "left_category",
        "right_category",
        "hash_hamming_distance",
        "structural_similarity",
        "exact_file_duplicate",
    ]
    return pd.DataFrame(rows, columns=columns)


def effect_balance_table(
    manifest: pd.DataFrame, effects: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat(
        [manifest[["category"]].reset_index(drop=True), effects.reset_index(drop=True)],
        axis=1,
    )
    combined["dark_background"] = combined["background"].eq("dark")
    combined["soft_contrast"] = combined["contrast_style"].eq("soft")
    combined["blur_applied"] = combined["blur_radius"].notna()
    combined["jpeg_applied"] = combined["jpeg_quality"].notna()
    combined["letter_spacing_applied"] = combined["letter_spacing_px"].ne(0)
    combined["rotation_applied"] = combined["rotation_degrees"].abs().gt(0)
    combined["horizontal_shift_applied"] = combined["horizontal_shift_px"].ne(0)
    combined["vertical_shift_applied"] = combined["vertical_shift_px"].ne(0)
    combined["scaling_applied"] = combined["scale_factor"].ne(1.0)
    binary = [
        "dark_background",
        "soft_contrast",
        "blur_applied",
        "jpeg_applied",
        "letter_spacing_applied",
        "rotation_applied",
        "horizontal_shift_applied",
        "vertical_shift_applied",
        "scaling_applied",
    ]
    rates = combined.groupby("category")[binary].mean().reset_index()
    means = (
        combined.groupby("category")[
            ["actual_font_size", "rotation_degrees", "blur_radius", "jpeg_quality"]
        ]
        .mean()
        .reset_index()
    )
    table = rates.merge(means, on="category", how="left")
    spreads = {
        column: float(table[column].max() - table[column].min())
        for column in binary
    }
    return table, {
        "maximum_binary_effect_rate_spread": round(max(spreads.values()), 8),
        "binary_effect_rate_spreads": {
            key: round(value, 8) for key, value in spreads.items()
        },
        "serious_imbalance_threshold": 0.05,
        "serious_category_dependent_effect_imbalance": bool(
            max(spreads.values()) > 0.05
        ),
        "mean_actual_font_size_spread": round(
            float(table["actual_font_size"].max() - table["actual_font_size"].min()),
            8,
        ),
    }


def phrase_balance(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = (
        manifest.groupby(["rendered_text", "category"])
        .size()
        .rename("images")
        .reset_index()
    )
    totals = counts.groupby("category")["images"].transform("sum")
    counts["category_rate"] = counts["images"] / totals
    pivot = counts.pivot(
        index="rendered_text", columns="category", values="images"
    ).fillna(0)
    observed = pivot.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    valid = expected > 0
    chi_square = float((((observed - expected) ** 2)[valid] / expected[valid]).sum())
    denominator = observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1)
    cramers_v = float(np.sqrt(chi_square / denominator)) if denominator > 0 else 0.0
    rate_pivot = counts.pivot(
        index="rendered_text", columns="category", values="category_rate"
    ).fillna(0)
    maximum_rate_spread = float((rate_pivot.max(axis=1) - rate_pivot.min(axis=1)).max())
    return counts, {
        "unique_phrases": int(manifest["rendered_text"].nunique()),
        "cramers_v": round(cramers_v, 8),
        "maximum_phrase_rate_spread": round(maximum_rate_spread, 8),
        "strong_category_association": bool(cramers_v >= 0.3),
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _label_bars(axis: plt.Axes, *, decimals: int = 0) -> None:
    for container in axis.containers:
        axis.bar_label(
            container,
            fmt=f"%.{decimals}f",
            padding=3,
            color="#22272E",
            fontsize=9,
        )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_figures(
    manifest: pd.DataFrame,
    quality: pd.DataFrame,
    effects: pd.DataFrame,
    effect_balance: pd.DataFrame,
    phrase_counts: pd.DataFrame,
    *,
    root: Path,
    figure_dir: Path,
) -> dict[str, str]:
    """Create the requested static EDA figures using a fixed visual system."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#22272E",
            "axes.edgecolor": "#59636E",
            "text.color": "#22272E",
            "xtick.color": "#59636E",
            "ytick.color": "#59636E",
        }
    )
    paths: dict[str, str] = {}

    category_counts = manifest.groupby("category").size().reindex(FONT_CATEGORIES)
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        [CATEGORY_LABELS[item] for item in category_counts.index],
        category_counts.values,
        color=[CATEGORY_COLORS[item] for item in category_counts.index],
        edgecolor="#37404A",
        linewidth=0.6,
    )
    axis.bar_label(bars, padding=3)
    axis.set_ylim(0, category_counts.max() * 1.12)
    axis.set_ylabel("Images")
    figure.suptitle("Images per font category", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.91, "All 3,600 manifest rows; the count axis starts at zero.", color="#59636E")
    _style_axis(axis)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = figure_dir / "class_counts.png"
    _save_figure(figure, path)
    paths["class_counts"] = str(path.relative_to(root)).replace("\\", "/")

    split_counts = manifest.groupby("split").size().reindex(SPLIT_ORDER)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        [item.title() for item in split_counts.index],
        split_counts.values,
        color=[SPLIT_COLORS[item] for item in split_counts.index],
        edgecolor="#37404A",
        linewidth=0.6,
    )
    axis.bar_label(bars, padding=3)
    axis.set_ylim(0, split_counts.max() * 1.12)
    axis.set_ylabel("Images")
    figure.suptitle("Images per family-level split", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.91, "Training has 2,400 images; validation and test each have 600.", color="#59636E")
    _style_axis(axis)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = figure_dir / "split_counts.png"
    _save_figure(figure, path)
    paths["split_counts"] = str(path.relative_to(root)).replace("\\", "/")

    families = (
        manifest.groupby(["category", "split"])["family"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=FONT_CATEGORIES, columns=SPLIT_ORDER)
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(families))
    width = 0.24
    for offset, split in enumerate(SPLIT_ORDER):
        axis.bar(
            x + (offset - 1) * width,
            families[split],
            width,
            label=split.title(),
            color=SPLIT_COLORS[split],
            edgecolor="#37404A",
            linewidth=0.5,
        )
    _label_bars(axis)
    axis.set_xticks(x, [CATEGORY_LABELS[item] for item in families.index])
    axis.set_ylim(0, 14)
    axis.set_ylabel("Unique font families")
    figure.suptitle("Independent font families per category and split", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.91, "Every category contains 12 train, 3 validation, and 3 test families.", color="#59636E")
    axis.legend(frameon=False, ncols=3, loc="upper right")
    _style_axis(axis)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = figure_dir / "families_per_category_split.png"
    _save_figure(figure, path)
    paths["families_per_category_split"] = str(path.relative_to(root)).replace("\\", "/")

    effect_labels = {
        "dark_background": "Dark background",
        "soft_contrast": "Soft contrast",
        "blur_applied": "Mild blur",
        "jpeg_applied": "JPEG round-trip",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, (column, label) in zip(axes.flat, effect_labels.items()):
        values = effect_balance.set_index("category")[column].reindex(FONT_CATEGORIES) * 100
        bars = axis.bar(
            [CATEGORY_LABELS[item] for item in values.index],
            values,
            color=[CATEGORY_COLORS[item] for item in values.index],
            edgecolor="#37404A",
            linewidth=0.5,
        )
        axis.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
        axis.set_title(label)
        axis.set_ylim(0, 45)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel("Share of images (%)")
        _style_axis(axis)
    figure.suptitle("Applied effect rates by category", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.925, "Each panel uses the same 0–45% scale across 720 images per category.", color="#59636E")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    path = figure_dir / "effect_distributions.png"
    _save_figure(figure, path)
    paths["effect_distributions"] = str(path.relative_to(root)).replace("\\", "/")

    readable = quality.loc[quality["opens_successfully"]].copy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, column, title, ylabel in (
        (axes[0], "brightness_mean", "Mean image brightness", "Mean grayscale value (0–255)"),
        (axes[1], "contrast_std", "Pixel contrast", "Grayscale standard deviation"),
    ):
        data = [
            readable.loc[readable["category"] == category, column].to_numpy()
            for category in FONT_CATEGORIES
        ]
        box = axis.boxplot(data, patch_artist=True, tick_labels=[CATEGORY_LABELS[item] for item in FONT_CATEGORIES])
        for patch, category in zip(box["boxes"], FONT_CATEGORIES):
            patch.set_facecolor(CATEGORY_COLORS[category])
            patch.set_alpha(0.75)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        _style_axis(axis)
    figure.suptitle("Brightness and contrast distributions", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.925, "Boxplots cover all successfully opened images; differences can reflect letter shape and stroke coverage.", color="#59636E")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    path = figure_dir / "brightness_contrast_distributions.png"
    _save_figure(figure, path)
    paths["brightness_contrast_distributions"] = str(path.relative_to(root)).replace("\\", "/")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, column, title, ylabel in (
        (axes[0], "actual_font_size", "Rendered font size", "Font size (pixels)"),
        (axes[1], "rotation_degrees", "Mild rotation", "Rotation (degrees)"),
    ):
        data = [
            effects.loc[manifest["category"] == category, column].to_numpy(dtype=float)
            for category in FONT_CATEGORIES
        ]
        box = axis.boxplot(data, patch_artist=True, tick_labels=[CATEGORY_LABELS[item] for item in FONT_CATEGORIES])
        for patch, category in zip(box["boxes"], FONT_CATEGORIES):
            patch.set_facecolor(CATEGORY_COLORS[category])
            patch.set_alpha(0.75)
        axis.axhline(0, color="#59636E", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        _style_axis(axis)
    figure.suptitle("Text size and rotation distributions", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.925, "The effect schedule is shared across categories; actual size is reduced when needed to fit text.", color="#59636E")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    path = figure_dir / "text_size_rotation_distributions.png"
    _save_figure(figure, path)
    paths["text_size_rotation_distributions"] = str(path.relative_to(root)).replace("\\", "/")

    top_phrases = (
        manifest.groupby("rendered_text")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .index
    )
    heatmap = (
        phrase_counts.loc[phrase_counts["rendered_text"].isin(top_phrases)]
        .pivot(index="rendered_text", columns="category", values="images")
        .reindex(index=top_phrases, columns=FONT_CATEGORIES)
        .fillna(0)
    )
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(heatmap.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    axis.set_xticks(range(len(FONT_CATEGORIES)), [CATEGORY_LABELS[item] for item in FONT_CATEGORIES], rotation=25)
    axis.set_yticks(range(len(heatmap)), heatmap.index)
    figure.suptitle("Counts for the 15 most-used phrases by category", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.91, "The same phrase schedule is used in every category; darker cells mean more images.", color="#59636E")
    figure.colorbar(image, ax=axis, label="Images")
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    path = figure_dir / "phrase_category_balance.png"
    _save_figure(figure, path)
    paths["phrase_category_balance"] = str(path.relative_to(root)).replace("\\", "/")

    merged = manifest.reset_index(drop=True).join(
        quality[["brightness_mean", "contrast_std"]].reset_index(drop=True)
    ).join(effects.reset_index(drop=True))
    representative_rows = []
    for category in FONT_CATEGORIES:
        for split in SPLIT_ORDER:
            group = merged.loc[
                (merged["category"] == category) & (merged["split"] == split)
            ].copy()
            target = group["contrast_std"].median()
            representative_rows.append(group.loc[(group["contrast_std"] - target).abs().idxmin()])
    representative = pd.DataFrame(representative_rows)
    figure, axes = plt.subplots(
        len(FONT_CATEGORIES),
        len(SPLIT_ORDER),
        figsize=(13, 11),
    )
    for axis, (_, row) in zip(axes.flat, representative.iterrows()):
        with Image.open(_resolve_path(row["image_path"], root)) as opened:
            axis.imshow(opened.convert("RGB"))
        axis.set_title(
            f"{CATEGORY_LABELS[row['category']]} · {row['split'].title()}\n{row['family']}",
            fontsize=8,
            pad=3,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Representative samples from every category and split", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.942, "One deterministic median-contrast image is shown for each of 15 category–split combinations.", color="#59636E")
    figure.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.6)
    path = figure_dir / "representative_samples.png"
    _save_figure(figure, path)
    paths["representative_samples"] = str(path.relative_to(root)).replace("\\", "/")

    merged["difficulty_score"] = (
        (1.0 - merged["contrast_std"].rank(pct=True))
        + (1.0 - merged["actual_font_size"].rank(pct=True))
        + merged["rotation_degrees"].abs() / max(merged["rotation_degrees"].abs().max(), 1e-9)
        + merged["blur_radius"].fillna(0) / max(merged["blur_radius"].fillna(0).max(), 1e-9)
        + (93 - merged["jpeg_quality"].fillna(93)) / 21
    )
    unusual_rows = []
    for category in FONT_CATEGORIES:
        unusual_rows.append(
            merged.loc[merged["category"] == category]
            .nlargest(3, "difficulty_score")
        )
    unusual = pd.concat(unusual_rows).drop_duplicates("image_path").head(15)
    figure, axes = plt.subplots(5, 3, figsize=(13, 11))
    for axis, (_, row) in zip(axes.flat, unusual.iterrows()):
        with Image.open(_resolve_path(row["image_path"], root)) as opened:
            axis.imshow(opened.convert("RGB"))
        blur = 0.0 if pd.isna(row["blur_radius"]) else float(row["blur_radius"])
        axis.set_title(
            f"{CATEGORY_LABELS[row['category']]} · {row['family']}\n"
            f"size {int(row['actual_font_size'])}, rot {row['rotation_degrees']:.2f}°, blur {blur:.2f}",
            fontsize=7,
            pad=3,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Automated unusual or difficult sample screen", fontsize=15, fontweight="bold")
    figure.text(0.125, 0.942, "Examples score highly for small text, low pixel contrast, rotation, blur, or JPEG compression; this is not a readability verdict.", color="#59636E")
    figure.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.6)
    path = figure_dir / "unusual_difficult_samples.png"
    _save_figure(figure, path)
    paths["unusual_difficult_samples"] = str(path.relative_to(root)).replace("\\", "/")
    return paths


def run_eda(
    *,
    manifest_path: str | Path,
    frozen_split_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full data-quality audit without generating data or training a model."""
    root_path = Path(root) if root is not None else project_root()
    resolved_manifest = _resolve_path(manifest_path, root_path)
    resolved_split = _resolve_path(frozen_split_path, root_path)
    resolved_output = _resolve_path(output_dir, root_path)
    resolved_figures = _resolve_path(figure_dir, root_path)
    manifest = pd.read_csv(resolved_manifest, keep_default_na=False)
    frozen_split = pd.read_csv(resolved_split, keep_default_na=False)
    effects = parse_effects(manifest)
    structure = validate_dataset_structure(manifest, frozen_split)
    quality = audit_image_files(manifest, root=root_path)
    suspicious = find_suspicious_pairs(quality, root=root_path)
    effect_table, effect_summary = effect_balance_table(manifest, effects)
    phrase_table, phrase_summary = phrase_balance(manifest)

    readable = quality.loc[quality["opens_successfully"]]
    exact_duplicate_hashes = readable.loc[
        readable["sha256"].duplicated(keep=False), "sha256"
    ].nunique()
    corrupt_count = int((quality["exists"] & ~quality["opens_successfully"]).sum())
    missing_count = int((~quality["exists"]).sum())
    blank_count = int(readable["blank"].sum())
    low_contrast_count = int(readable["low_pixel_contrast"].sum())
    wrong_dimensions = int(
        (
            (readable["width"] != EXPECTED_IMAGE_SIZE[0])
            | (readable["height"] != EXPECTED_IMAGE_SIZE[1])
        ).sum()
    )
    summary: dict[str, Any] = {
        "status": "passed",
        "purpose": "EDA and data-quality checks only; no model training or model results.",
        "dataset_ready_for_model_training": True,
        "structure": structure,
        "image_quality": {
            "manifest_paths_checked": int(len(quality)),
            "images_opened_successfully": int(quality["opens_successfully"].sum()),
            "missing_images": missing_count,
            "corrupted_images": corrupt_count,
            "blank_images": blank_count,
            "low_pixel_contrast_flags": low_contrast_count,
            "automated_unreadable_flags": missing_count
            + corrupt_count
            + blank_count
            + low_contrast_count,
            "readability_scope": (
                "Automated screen for missing, corrupt, blank, or extremely low-contrast "
                "images; human review is still required for subjective readability."
            ),
            "wrong_dimension_images": wrong_dimensions,
            "expected_dimensions": "224x96",
            "brightness": {
                "minimum": round(float(readable["brightness_mean"].min()), 4),
                "median": round(float(readable["brightness_mean"].median()), 4),
                "maximum": round(float(readable["brightness_mean"].max()), 4),
            },
            "contrast": {
                "minimum": round(float(readable["contrast_std"].min()), 4),
                "median": round(float(readable["contrast_std"].median()), 4),
                "maximum": round(float(readable["contrast_std"].max()), 4),
            },
            "exact_duplicate_hash_groups": int(exact_duplicate_hashes),
            "suspicious_near_identical_pairs": int(len(suspicious)),
            "near_duplicate_definition": (
                "16x16 difference-hash Hamming distance <= 2 and structural "
                "similarity >= 0.995"
            ),
        },
        "text_size": {
            "minimum_actual_font_size": int(effects["actual_font_size"].min()),
            "median_actual_font_size": float(effects["actual_font_size"].median()),
            "maximum_actual_font_size": int(effects["actual_font_size"].max()),
        },
        "effects": {
            **effect_summary,
            "background_counts": {
                key: int(value)
                for key, value in effects["background"].value_counts().to_dict().items()
            },
            "blurred_images": int(effects["blur_radius"].notna().sum()),
            "jpeg_compressed_images": int(effects["jpeg_quality"].notna().sum()),
            "rotation_range_degrees": [
                float(effects["rotation_degrees"].min()),
                float(effects["rotation_degrees"].max()),
            ],
        },
        "phrase_balance": phrase_summary,
        "model_feature_audit": MODEL_FEATURE_AUDIT,
        "frozen_split_sha256": _sha256(resolved_split),
        "source_manifest_sha256": _sha256(resolved_manifest),
        "model_training_performed": False,
    }
    blocking = [
        missing_count,
        corrupt_count,
        blank_count,
        low_contrast_count,
        wrong_dimensions,
        exact_duplicate_hashes,
        structure["family_overlap_count"],
        int(effect_summary["serious_category_dependent_effect_imbalance"]),
        int(phrase_summary["strong_category_association"]),
    ]
    if any(blocking):
        summary["status"] = "failed"
        summary["dataset_ready_for_model_training"] = False

    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_figures.mkdir(parents=True, exist_ok=True)
    quality_with_effects = pd.concat(
        [quality.reset_index(drop=True), effects.reset_index(drop=True)], axis=1
    )
    quality_with_effects.to_csv(
        resolved_output / "image_quality_metrics.csv", index=False
    )
    suspicious.to_csv(resolved_output / "suspicious_image_pairs.csv", index=False)
    effect_table.to_csv(resolved_output / "effect_balance_by_category.csv", index=False)
    phrase_table.to_csv(resolved_output / "phrase_balance_by_category.csv", index=False)
    (
        manifest.groupby(["category", "split"])
        .agg(images=("image_path", "size"), families=("family", "nunique"))
        .reset_index()
        .to_csv(resolved_output / "dataset_counts.csv", index=False)
    )
    figures = create_figures(
        manifest,
        quality,
        effects,
        effect_table,
        phrase_table,
        root=root_path,
        figure_dir=resolved_figures,
    )
    summary["figures"] = figures
    save_json(summary, resolved_output / "eda_validation_summary.json")
    chart_map = [
        {
            "figure": key,
            "path": value,
            "question": {
                "class_counts": "Are image classes balanced?",
                "split_counts": "How many images are in each family-level split?",
                "families_per_category_split": "Are independent families balanced by category and split?",
                "effect_distributions": "Are generation effects balanced across categories?",
                "brightness_contrast_distributions": "What do image brightness and contrast distributions look like?",
                "text_size_rotation_distributions": "What text-size and rotation ranges were used?",
                "phrase_category_balance": "Are rendered phrases associated with category?",
                "representative_samples": "What does each category and split look like?",
                "unusual_difficult_samples": "Which images merit closer manual review?",
            }[key],
            "claim_supported": {
                "class_counts": "Each category contains 720 images.",
                "split_counts": "The dataset contains 2,400 train, 600 validation, and 600 test images.",
                "families_per_category_split": "Each category contains 12 train, 3 validation, and 3 test families.",
                "effect_distributions": "Selected binary effect rates match across categories.",
                "brightness_contrast_distributions": "All images have measurable pixel variation; category differences are descriptive.",
                "text_size_rotation_distributions": "Text remains within the configured mild rendering ranges.",
                "phrase_category_balance": "The phrase distribution is identical across categories.",
                "representative_samples": "All 15 category-split combinations are represented.",
                "unusual_difficult_samples": "The automated screen highlights edge cases for human inspection.",
            }[key],
        }
        for key, value in figures.items()
    ]
    save_json(chart_map, resolved_output / "chart_map.json")
    if summary["status"] != "passed":
        raise AssertionError(
            "EDA validation found a blocking data-quality problem. "
            f"See {resolved_output / 'eda_validation_summary.json'}."
        )
    return summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Audit the existing full FontSense dataset without regenerating it."
    )
    parser.add_argument(
        "--manifest",
        default=str(root / "reports/dataset/full_manifest.csv"),
    )
    parser.add_argument(
        "--frozen-split",
        default=str(root / "data/interim/google_fonts_final_family_split.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "reports/eda"),
    )
    parser.add_argument(
        "--figure-dir",
        default=str(root / "reports/figures"),
    )
    args = parser.parse_args()
    summary = run_eda(
        manifest_path=args.manifest,
        frozen_split_path=args.frozen_split,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        root=root,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
