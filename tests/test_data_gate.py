from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_data_gate_artifacts_use_the_frozen_dataset() -> None:
    manifest = pd.read_csv(ROOT / "reports/dataset/full_manifest.csv")
    frozen_split = pd.read_csv(
        ROOT / "data/interim/google_fonts_final_family_split.csv"
    )
    preprocessing = json.loads(
        (ROOT / "reports/preprocessing_manifest.json").read_text(encoding="utf-8")
    )

    assert len(manifest) == 3600
    assert manifest["family"].nunique() == 90
    assert not manifest["image_path"].duplicated().any()
    assert manifest.groupby("family")["split"].nunique().max() == 1
    assert set(manifest["family"]) == set(frozen_split["family"])
    assert preprocessing["fit_boundary"] == "train"
    assert preprocessing["input"]["grayscale"] is True
    assert (
        preprocessing["input"]["resize_width"],
        preprocessing["input"]["resize_height"],
    ) == (112, 48)
    assert preprocessing["training_only_augmentation"]["validation_enabled"] is False
    assert preprocessing["training_only_augmentation"]["test_enabled"] is False


def test_data_gate_csv_schemas_and_evidence_paths() -> None:
    issue_path = ROOT / "reports/data_gate/issue_log.csv"
    split_path = ROOT / "reports/data_gate/split_summary.csv"

    with issue_path.open(encoding="utf-8-sig", newline="") as handle:
        issue_rows = list(csv.DictReader(handle))
    assert list(issue_rows[0]) == [
        "issue_id",
        "finding",
        "evidence_path",
        "risk",
        "decision",
        "action_or_limitation",
        "status",
        "owner",
    ]
    for row in issue_rows:
        for evidence_path in row["evidence_path"].split(";"):
            assert (ROOT / evidence_path.strip()).exists(), evidence_path

    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    assert list(split_rows[0]) == [
        "split",
        "image_count",
        "family_count",
        "images_per_category",
        "families_per_category",
        "overlap_result",
        "notes",
    ]
    assert sum(
        int(row["image_count"]) for row in split_rows if row["split"] != "total"
    ) == 3600
