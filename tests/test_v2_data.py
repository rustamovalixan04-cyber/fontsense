from __future__ import annotations

from pathlib import Path

import pandas as pd

from fontsense.google_fonts import CATEGORY_ORDER
from fontsense.v2_data import V2_LICENSE_FILE_BY_CODE, create_v2_family_split


def _audit_frame(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for category in CATEGORY_ORDER:
        for index in range(48):
            family = f"{category} family {index:02d}"
            rows.append(
                {
                    "family": family,
                    "category": category,
                    "source": "Google Fonts official repository",
                    "license": "OFL",
                    "path": str(tmp_path / f"{category}-{index}.ttf"),
                    "latin_support": True,
                    "validation_status": "passed",
                    "failure_reason": "",
                    "usable": True,
                    "slug": f"{category}{index}",
                    "repository_folder": "ofl",
                    "source_commit": "abc123",
                    "metadata_url": "https://example.test/METADATA.pb",
                    "font_url": "https://example.test/font.ttf",
                    "license_url": "https://example.test/OFL.txt",
                    "license_path": "license.txt",
                    "subsets": "latin",
                    "font_sha256": "0" * 64,
                    "font_size_bytes": 1,
                }
            )
    return pd.DataFrame(rows)


def _v1_frame() -> pd.DataFrame:
    rows = []
    for category in CATEGORY_ORDER:
        for index in range(18):
            rows.append(
                {
                    "family": f"{category} family {index:02d}",
                    "category": category,
                    "split": "test" if index >= 15 else "train",
                }
            )
    return pd.DataFrame(rows)


def test_v2_split_is_balanced_deterministic_and_protects_v1_test(tmp_path: Path):
    audit_path = tmp_path / "audit.csv"
    v1_path = tmp_path / "v1.csv"
    _audit_frame(tmp_path).to_csv(audit_path, index=False)
    _v1_frame().to_csv(v1_path, index=False)

    first, _, summary = create_v2_family_split(
        audit_path,
        v1_path,
        tmp_path / "split.csv",
        tmp_path / "excluded.csv",
        tmp_path / "summary.json",
    )
    second, _, _ = create_v2_family_split(
        audit_path,
        v1_path,
        tmp_path / "split2.csv",
        tmp_path / "excluded2.csv",
        tmp_path / "summary2.json",
    )

    assert first[["family", "category", "split"]].equals(
        second[["family", "category", "split"]]
    )
    assert first["family"].nunique() == 200
    assert first.groupby("category")["family"].nunique().eq(40).all()
    assert first.groupby("split")["family"].nunique().to_dict() == {
        "test": 30,
        "train": 140,
        "validation": 30,
    }
    assert not first["was_in_v1_test"].any()
    assert first.loc[first["split"] == "test", "fresh_v2_test_family"].all()
    assert summary["planned_images_total"] == 20_000


def test_v2_uses_the_official_ubuntu_licence_filename():
    assert V2_LICENSE_FILE_BY_CODE["UFL"] == "LICENCE.txt"
