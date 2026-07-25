import pandas as pd

from fontsense.split import family_level_split, assert_no_family_leakage


def test_family_split_has_no_overlap():
    rows = []
    for category in ["serif", "sans_serif", "display", "handwriting", "monospace"]:
        for index in range(8):
            rows.append({"family": f"{category}_{index}", "category": category, "path": f"/{index}.ttf"})
    split = family_level_split(pd.DataFrame(rows), seed=42)
    assert_no_family_leakage(split)
    assert set(split["split"]) == {"train", "validation", "test"}
