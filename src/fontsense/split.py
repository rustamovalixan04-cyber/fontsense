from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import project_root

FONT_CATEGORIES = ("serif", "sans_serif", "display", "handwriting", "monospace")
FINAL_SPLIT_COLUMNS = ["family", "category", "split", "path", "source", "license"]


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        if not np.isclose(self.train + self.validation + self.test, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("All split ratios must be positive")


@dataclass(frozen=True)
class SplitCounts:
    train: int = 12
    validation: int = 3
    test: int = 3

    @property
    def total(self) -> int:
        return self.train + self.validation + self.test

    def validate(self, families_per_category: int) -> None:
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("Every split count must be positive")
        if self.total != families_per_category:
            raise ValueError(
                "Train, validation, and test counts must add up to families_per_category"
            )


def family_level_split(
    font_manifest: pd.DataFrame,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each font family to exactly one split inside its category."""
    ratios.validate()
    required = {"family", "category"}
    missing = required - set(font_manifest.columns)
    if missing:
        raise ValueError(f"Font manifest is missing columns: {sorted(missing)}")

    unique = font_manifest.drop_duplicates(["family", "category"]).copy()
    rng = np.random.default_rng(seed)
    assignments: list[pd.DataFrame] = []
    for category, group in unique.groupby("category", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        count = len(group)
        if count < 3:
            raise ValueError(f"Category '{category}' needs at least 3 independent font families; found {count}.")
        n_test = max(1, int(round(count * ratios.test)))
        n_val = max(1, int(round(count * ratios.validation)))
        if n_test + n_val >= count:
            n_test = 1
            n_val = 1
        split = np.array(["train"] * count, dtype=object)
        split[-n_test:] = "test"
        split[-(n_test + n_val):-n_test] = "validation"
        # category-specific deterministic shuffle, so the last rows are not tied to source ordering
        order = rng.permutation(count)
        group = group.iloc[order].reset_index(drop=True)
        group["split"] = split
        assignments.append(group)

    result = pd.concat(assignments, ignore_index=True)
    overlap = result.groupby("family")["split"].nunique()
    if (overlap > 1).any():
        bad = overlap[overlap > 1].index.tolist()
        raise AssertionError(f"Leakage detected: families assigned to multiple splits: {bad[:5]}")
    return result


def assert_balanced_family_split(
    manifest: pd.DataFrame,
    *,
    categories: tuple[str, ...] = FONT_CATEGORIES,
    families_per_category: int = 18,
    split_counts: SplitCounts = SplitCounts(),
) -> None:
    """Validate the exact family and split counts required for the final dataset."""
    split_counts.validate(families_per_category)
    required = {"family", "category", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Family split is missing columns: {sorted(missing)}")
    if manifest["family"].nunique() != len(categories) * families_per_category:
        raise AssertionError(
            f"Expected {len(categories) * families_per_category} unique families; "
            f"found {manifest['family'].nunique()}."
        )

    category_counts = manifest.groupby("category")["family"].nunique()
    expected_category_counts = pd.Series(
        {category: families_per_category for category in categories},
        dtype="int64",
    )
    if not category_counts.reindex(categories, fill_value=0).equals(expected_category_counts):
        raise AssertionError(
            f"Expected {families_per_category} families per category; "
            f"found {category_counts.to_dict()}."
        )

    expected_split_counts = {
        "train": split_counts.train * len(categories),
        "validation": split_counts.validation * len(categories),
        "test": split_counts.test * len(categories),
    }
    actual_split_counts = manifest.groupby("split")["family"].nunique().to_dict()
    if actual_split_counts != expected_split_counts:
        raise AssertionError(
            f"Expected split counts {expected_split_counts}; found {actual_split_counts}."
        )

    per_category = manifest.groupby(["category", "split"])["family"].nunique().unstack(fill_value=0)
    expected_per_category = {
        "train": split_counts.train,
        "validation": split_counts.validation,
        "test": split_counts.test,
    }
    for category in categories:
        actual = per_category.loc[category].to_dict()
        if actual != expected_per_category:
            raise AssertionError(
                f"Category '{category}' expected split counts {expected_per_category}; found {actual}."
            )
    assert_no_family_leakage(manifest)


def balanced_family_split(
    font_manifest: pd.DataFrame,
    *,
    categories: tuple[str, ...] = FONT_CATEGORIES,
    families_per_category: int = 18,
    split_counts: SplitCounts = SplitCounts(),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select equal category counts and assign whole families deterministically."""
    split_counts.validate(families_per_category)
    required = {"family", "category", "path", "source", "license"}
    missing = required - set(font_manifest.columns)
    if missing:
        raise ValueError(f"Font manifest is missing columns: {sorted(missing)}")

    fonts = font_manifest.copy()
    if "usable" in fonts.columns:
        usable_mask = fonts["usable"].astype(str).str.lower().isin({"true", "1"})
        fonts = fonts.loc[usable_mask].copy()
    for column in required:
        fonts[column] = fonts[column].fillna("").astype(str).str.strip()
    if fonts.empty:
        raise ValueError("Font manifest contains zero usable families")
    if fonts[list(required)].eq("").any().any():
        raise ValueError("Usable font rows must include family, category, path, source, and license")
    duplicate_families = fonts.loc[fonts.duplicated("family", keep=False), "family"].unique().tolist()
    if duplicate_families:
        raise ValueError(f"Font manifest contains duplicate usable families: {duplicate_families[:5]}")
    unsupported = sorted(set(fonts["category"]) - set(categories))
    if unsupported:
        raise ValueError(f"Font manifest contains unsupported categories: {unsupported}")

    rng = np.random.default_rng(seed)
    selected_groups: list[pd.DataFrame] = []
    excluded_groups: list[pd.DataFrame] = []
    split_labels = (
        ["train"] * split_counts.train
        + ["validation"] * split_counts.validation
        + ["test"] * split_counts.test
    )

    for category in categories:
        group = (
            fonts.loc[fonts["category"] == category]
            .sort_values("family", key=lambda values: values.str.casefold())
            .reset_index(drop=True)
        )
        if len(group) < families_per_category:
            raise ValueError(
                f"Category '{category}' needs {families_per_category} usable families; found {len(group)}."
            )
        shuffled = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        selected = shuffled.iloc[:families_per_category].copy()
        selected["split"] = split_labels
        selected["selection_seed"] = seed
        selected_groups.append(selected)

        excluded = shuffled.iloc[families_per_category:].copy()
        excluded["exclusion_reason"] = (
            f"not selected for balanced {families_per_category}-family category sample"
        )
        excluded["selection_seed"] = seed
        excluded_groups.append(excluded)

    selected_frame = pd.concat(selected_groups, ignore_index=True)
    excluded_frame = pd.concat(excluded_groups, ignore_index=True)
    assert_balanced_family_split(
        selected_frame,
        categories=categories,
        families_per_category=families_per_category,
        split_counts=split_counts,
    )

    category_order = {category: index for index, category in enumerate(categories)}
    split_order = {"train": 0, "validation": 1, "test": 2}
    selected_frame["_category_order"] = selected_frame["category"].map(category_order)
    selected_frame["_split_order"] = selected_frame["split"].map(split_order)
    selected_frame = (
        selected_frame.sort_values(
            ["_category_order", "_split_order", "family"],
            key=lambda values: values.str.casefold() if values.dtype == "object" else values,
        )
        .drop(columns=["_category_order", "_split_order"])
        .reset_index(drop=True)
    )
    excluded_frame["_category_order"] = excluded_frame["category"].map(category_order)
    excluded_frame = (
        excluded_frame.sort_values(
            ["_category_order", "family"],
            key=lambda values: values.str.casefold() if values.dtype == "object" else values,
        )
        .drop(columns=["_category_order"])
        .reset_index(drop=True)
    )

    selected_first = FINAL_SPLIT_COLUMNS + ["selection_seed"]
    selected_frame = selected_frame[
        selected_first + [column for column in selected_frame.columns if column not in selected_first]
    ]
    excluded_first = [
        "family",
        "category",
        "path",
        "source",
        "license",
        "exclusion_reason",
        "selection_seed",
    ]
    excluded_frame = excluded_frame[
        excluded_first + [column for column in excluded_frame.columns if column not in excluded_first]
    ]
    return selected_frame, excluded_frame


def validate_split_font_files(manifest: pd.DataFrame) -> None:
    """Confirm every selected font path exists and renders before saving the split."""
    from .google_fonts import validate_font_file

    failures: list[str] = []
    for row in manifest.itertuples(index=False):
        font_path = Path(str(row.path))
        resolved = font_path if font_path.is_absolute() else project_root() / font_path
        if not resolved.is_file():
            failures.append(f"{row.family}: file not found at {font_path}")
            continue
        valid, reason = validate_font_file(resolved)
        if not valid:
            failures.append(f"{row.family}: {reason}")
    if failures:
        raise ValueError(f"Selected font validation failed: {'; '.join(failures[:5])}")


def create_balanced_family_split(
    manifest_path: str | Path,
    output_path: str | Path,
    excluded_output_path: str | Path,
    *,
    families_per_category: int = 18,
    split_counts: SplitCounts = SplitCounts(),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the accepted audit, validate the selected files, and save both CSVs."""
    font_manifest = pd.read_csv(manifest_path, keep_default_na=False)
    selected, excluded = balanced_family_split(
        font_manifest,
        families_per_category=families_per_category,
        split_counts=split_counts,
        seed=seed,
    )
    validate_split_font_files(selected)

    output_path = Path(output_path)
    excluded_output_path = Path(excluded_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    excluded.to_csv(excluded_output_path, index=False)
    return selected, excluded


def assert_no_family_leakage(manifest: pd.DataFrame) -> None:
    required = {"family", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    counts = manifest.groupby("family")["split"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise AssertionError(f"Family leakage found: {leaked.index.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced leakage-safe Google Fonts family split.")
    parser.add_argument(
        "--manifest",
        default=str(project_root() / "data/interim/google_fonts_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(project_root() / "data/interim/google_fonts_final_family_split.csv"),
    )
    parser.add_argument(
        "--excluded-output",
        default=str(project_root() / "data/interim/google_fonts_balancing_exclusions.csv"),
    )
    parser.add_argument("--families-per-category", type=int, default=18)
    parser.add_argument("--train-families", type=int, default=12)
    parser.add_argument("--validation-families", type=int, default=3)
    parser.add_argument("--test-families", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_counts = SplitCounts(args.train_families, args.validation_families, args.test_families)
    selected, excluded = create_balanced_family_split(
        args.manifest,
        args.output,
        args.excluded_output,
        families_per_category=args.families_per_category,
        split_counts=split_counts,
        seed=args.seed,
    )
    print("Families by category and split:")
    print(selected.groupby(["category", "split"]).size().unstack(fill_value=0))
    print(f"Selected {selected['family'].nunique()} unique families.")
    print(f"Excluded {excluded['family'].nunique()} usable families from balancing.")
    print(f"Saved split: {args.output}")
    print(f"Saved exclusions: {args.excluded_output}")


if __name__ == "__main__":
    main()
