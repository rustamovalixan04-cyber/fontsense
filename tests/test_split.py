from pathlib import Path

import pandas as pd
import pytest
from matplotlib import get_data_path

from fontsense.split import (
    FONT_CATEGORIES,
    SplitCounts,
    assert_balanced_family_split,
    assert_no_family_leakage,
    balanced_family_split,
    create_balanced_family_split,
    family_level_split,
)


def test_family_split_has_no_overlap():
    rows = []
    for category in ["serif", "sans_serif", "display", "handwriting", "monospace"]:
        for index in range(8):
            rows.append({"family": f"{category}_{index}", "category": category, "path": f"/{index}.ttf"})
    split = family_level_split(pd.DataFrame(rows), seed=42)
    assert_no_family_leakage(split)
    assert set(split["split"]) == {"train", "validation", "test"}


def balanced_rows(families_per_category=25, path="/font.ttf"):
    rows = []
    for category in FONT_CATEGORIES:
        for index in range(families_per_category):
            rows.append(
                {
                    "family": f"{category}_{index:02d}",
                    "category": category,
                    "path": path,
                    "source": "Google Fonts official repository",
                    "license": "OFL",
                    "usable": True,
                }
            )
    return rows


def test_balanced_family_split_has_exact_counts_and_is_order_independent():
    frame = pd.DataFrame(balanced_rows())

    selected, excluded = balanced_family_split(frame, seed=42)
    shuffled_selected, shuffled_excluded = balanced_family_split(
        frame.sample(frac=1.0, random_state=7),
        seed=42,
    )

    assert_balanced_family_split(selected)
    assert selected["family"].nunique() == 90
    assert selected.groupby("category")["family"].nunique().to_dict() == {
        category: 18 for category in FONT_CATEGORIES
    }
    assert selected.groupby("split")["family"].nunique().to_dict() == {
        "test": 15,
        "train": 60,
        "validation": 15,
    }
    assert len(excluded) == 35
    pd.testing.assert_frame_equal(selected, shuffled_selected)
    pd.testing.assert_frame_equal(excluded, shuffled_excluded)


def test_balanced_family_split_rejects_a_category_with_fewer_than_18_families():
    frame = pd.DataFrame(balanced_rows())
    removed = {f"monospace_{index:02d}" for index in range(17, 25)}
    frame = frame.loc[
        ~((frame["category"] == "monospace") & (frame["family"].isin(removed)))
    ]

    with pytest.raises(ValueError, match="monospace.*needs 18 usable families"):
        balanced_family_split(frame, seed=42)


def test_create_balanced_family_split_saves_renderable_fonts(tmp_path):
    font_path = str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    manifest_path = tmp_path / "audit.csv"
    output_path = tmp_path / "final_split.csv"
    excluded_path = tmp_path / "excluded.csv"
    pd.DataFrame(balanced_rows(families_per_category=18, path=font_path)).to_csv(
        manifest_path,
        index=False,
    )

    selected, excluded = create_balanced_family_split(
        manifest_path,
        output_path,
        excluded_path,
        split_counts=SplitCounts(12, 3, 3),
        seed=42,
    )

    saved = pd.read_csv(output_path)
    assert output_path.is_file()
    assert excluded_path.is_file()
    assert len(saved) == 90
    assert len(selected) == 90
    assert excluded.empty
    assert saved.columns[:6].tolist() == ["family", "category", "split", "path", "source", "license"]
