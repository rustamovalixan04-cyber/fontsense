from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw

import fontsense.final_evaluation as final_evaluation
from fontsense.cnn_model import FontSenseCNN
from fontsense.final_evaluation import (
    evaluate_final_test,
    prepare_final_evaluation,
    select_validation_threshold,
    sha256_file,
)


ROOT = Path(__file__).parents[1]
CATEGORIES = list(final_evaluation.EXPECTED_CLASSES)


def test_validation_only_threshold_rule_selects_point_60():
    predictions = pd.read_csv(
        ROOT
        / "reports"
        / "cnn"
        / "best_cnn_validation_predictions.csv"
    )

    threshold, analysis, summary = select_validation_threshold(
        predictions,
        CATEGORIES,
    )

    assert threshold == pytest.approx(0.60)
    assert summary["source_split"] == "validation"
    assert summary["test_data_used"] is False
    selected = analysis.loc[analysis["threshold"] == threshold].iloc[0]
    assert selected["accepted_accuracy"] >= 0.90
    assert selected["coverage"] >= 0.50
    assert not analysis.loc[
        analysis["threshold"] < threshold,
        "qualifies",
    ].any()


def test_prepare_freezes_model_before_reading_test_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    read_paths: list[Path] = []
    original_read_csv = final_evaluation.pd.read_csv

    def recording_read_csv(path, *args, **kwargs):
        read_paths.append(Path(path).resolve())
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(
        final_evaluation.pd,
        "read_csv",
        recording_read_csv,
    )
    manifest_path = ROOT / "reports" / "dataset" / "full_manifest.csv"
    freeze = prepare_final_evaluation(
        checkpoint_path=ROOT / "artifacts" / "cnn" / "cnn_model.pt",
        metadata_path=ROOT / "artifacts" / "cnn" / "cnn_metadata.json",
        comparison_path=(
            ROOT / "reports" / "cnn" / "cnn_experiment_comparison.csv"
        ),
        validation_predictions_path=(
            ROOT
            / "reports"
            / "cnn"
            / "best_cnn_validation_predictions.csv"
        ),
        family_split_path=(
            ROOT
            / "data"
            / "interim"
            / "google_fonts_final_family_split.csv"
        ),
        manifest_path=manifest_path,
        report_dir=tmp_path,
    )

    assert manifest_path.resolve() not in read_paths
    assert freeze["status"] == "prepared"
    assert freeze["uncertainty_threshold"]["selected_threshold"] == 0.60
    assert freeze["uncertainty_threshold"]["test_data_used"] is False
    assert freeze["test_access_before_freeze"] == {
        "test_manifest_rows_loaded": 0,
        "test_images_loaded": 0,
        "test_predictions_made": 0,
        "test_metrics_recorded": False,
    }


def _write_image(path: Path, index: int) -> None:
    image = Image.new("RGB", (64, 32), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (5 + index * 2, 5, 15 + index * 2, 26),
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _tiny_final_contract(tmp_path: Path) -> Path:
    model = FontSenseCNN(num_classes=5, width=2, dropout=0.25)
    checkpoint_path = tmp_path / "cnn_model.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "classes": CATEGORIES,
        "architecture": {"width": 2, "dropout": 0.25},
        "preprocessing": {
            "image_size": [32, 16],
            "grayscale": True,
            "normalize_mean": [0.5],
            "normalize_std": [0.5],
        },
        "selected_validation_run": {
            "name": "Tiny test model",
            "best_epoch": 1,
            "validation_macro_f1": 0.2,
            "validation_accuracy": 0.2,
        },
        "training_data": {
            "train_rows": 10,
            "validation_rows": 5,
            "fit_splits": ["train"],
            "selection_split": "validation",
            "test_images_loaded": 0,
            "test_rows_evaluated": 0,
            "test_metrics_recorded": False,
        },
        "seed": 42,
    }
    torch.save(checkpoint, checkpoint_path)

    manifest_rows = []
    split_rows = []
    for index, category in enumerate(CATEGORIES):
        image_path = tmp_path / "images" / f"{category}.png"
        _write_image(image_path, index)
        family = f"{category}_test_family"
        manifest_rows.append(
            {
                "image_path": str(image_path),
                "family": family,
                "category": category,
                "split": "test",
            }
        )
        split_rows.append(
            {
                "family": family,
                "category": category,
                "split": "test",
            }
        )
    manifest_path = tmp_path / "manifest.csv"
    split_path = tmp_path / "family_split.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)

    freeze = {
        "status": "prepared",
        "selected_model": {
            "name": "Tiny test model",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "preprocessing": checkpoint["preprocessing"],
        "class_order": CATEGORIES,
        "random_seed": 42,
        "frozen_family_split": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
        },
        "full_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "uncertainty_threshold": {
            "selected_threshold": 0.60,
            "source_split": "validation",
            "test_data_used": False,
            "method": "Unit-test validation-only rule.",
        },
    }
    freeze_path = tmp_path / "reports" / "pre_test_freeze.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    return freeze_path


def test_final_evaluation_scores_tiny_test_once_and_blocks_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    freeze_path = _tiny_final_contract(tmp_path)
    monkeypatch.setattr(final_evaluation, "EXPECTED_TEST_IMAGES", 5)
    monkeypatch.setattr(
        final_evaluation,
        "EXPECTED_TEST_IMAGES_PER_CLASS",
        1,
    )
    monkeypatch.setattr(
        final_evaluation,
        "EXPECTED_TEST_FAMILIES_PER_CLASS",
        1,
    )
    monkeypatch.setattr(
        final_evaluation,
        "EXPECTED_IMAGE_SIZE",
        (32, 16),
    )

    metrics = evaluate_final_test(freeze_path)

    assert metrics["status"] == "completed"
    assert metrics["test_results"]["images"] == 5
    assert metrics["checks"]["checkpoint_loaded"] is True
    assert metrics["checks"]["test_used_for_threshold_selection"] is False
    assert (
        Path(freeze_path.parent / "final_test_predictions.csv").is_file()
    )
    assert len(list((freeze_path.parent / "figures").glob("*.png"))) == 3
    with pytest.raises(RuntimeError, match="already started or completed"):
        evaluate_final_test(freeze_path)


def test_frozen_hash_mismatch_stops_before_test_images_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    freeze_path = _tiny_final_contract(tmp_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["full_manifest"]["sha256"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    opened: list[str] = []

    def forbidden_open(*args, **kwargs):
        opened.append(str(args[0]))
        raise AssertionError("Test image should not be opened")

    monkeypatch.setattr(final_evaluation.Image, "open", forbidden_open)

    with pytest.raises(AssertionError, match="Frozen manifest hash changed"):
        evaluate_final_test(freeze_path)
    assert opened == []


def test_saved_final_evaluation_is_complete_and_non_rerunnable():
    report_dir = ROOT / "reports" / "final_evaluation"
    freeze_path = report_dir / "pre_test_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    receipt = json.loads(
        (report_dir / "evaluation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = json.loads(
        (report_dir / "final_test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = pd.read_csv(
        report_dir / "final_test_predictions.csv"
    )
    probability_columns = [
        f"probability_{category}" for category in CATEGORIES
    ]

    assert freeze["test_access_before_freeze"] == {
        "test_manifest_rows_loaded": 0,
        "test_images_loaded": 0,
        "test_predictions_made": 0,
        "test_metrics_recorded": False,
    }
    assert receipt["status"] == "completed"
    assert receipt["rerun_allowed"] is False
    assert metrics["status"] == "completed"
    assert metrics["test_results"]["images"] == 600
    assert metrics["test_results"]["families"] == 15
    assert set(metrics["test_results"]["images_per_category"].values()) == {
        120
    }
    assert set(
        metrics["test_results"]["families_per_category"].values()
    ) == {3}
    assert metrics["checks"]["family_overlap_count"] == 0
    assert metrics["uncertainty_threshold"]["selected_using"] == (
        "validation only"
    )
    assert metrics["uncertainty_threshold"][
        "test_results_used_for_selection"
    ] is False
    assert metrics["checks"]["post_test_retraining_or_tuning"] is False
    assert all(
        metrics["checks"][key] is True
        for key in (
            "checkpoint_loaded",
            "preprocessing_loaded",
            "checkpoint_hash_unchanged",
            "frozen_split_hash_unchanged",
            "manifest_hash_unchanged",
        )
    )
    assert len(predictions) == 600
    assert set(predictions["split"]) == {"test"}
    assert predictions.groupby("category").size().eq(120).all()
    assert predictions.groupby("category")["family"].nunique().eq(3).all()
    assert np.allclose(
        predictions[probability_columns].sum(axis=1),
        1.0,
    )
    assert (
        int(predictions["correct"].sum())
        == metrics["test_results"]["correct_predictions"]
    )
    with pytest.raises(RuntimeError, match="already started or completed"):
        evaluate_final_test(freeze_path)
