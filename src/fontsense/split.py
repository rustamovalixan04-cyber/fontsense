from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def assert_no_family_leakage(manifest: pd.DataFrame) -> None:
    required = {"family", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    counts = manifest.groupby("family")["split"].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        raise AssertionError(f"Family leakage found: {leaked.index.tolist()}")
