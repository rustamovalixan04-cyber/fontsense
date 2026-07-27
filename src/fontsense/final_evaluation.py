from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .split import assert_no_family_leakage
from .train_cnn import build_image_transform, load_cnn_checkpoint
from .utils import load_json, project_root, save_json, set_seed

matplotlib.use("Agg")
from matplotlib import pyplot as plt


EXPECTED_CLASSES = (
    "display",
    "handwriting",
    "monospace",
    "sans_serif",
    "serif",
)
EXPECTED_TEST_IMAGES = 600
EXPECTED_TEST_IMAGES_PER_CLASS = 120
EXPECTED_TEST_FAMILIES_PER_CLASS = 3
EXPECTED_SELECTED_RUN = "Reference small CNN"
EXPECTED_LEARNING_RATE = 0.001
EXPECTED_WIDTH = 16
EXPECTED_DROPOUT = 0.25
EXPECTED_BEST_EPOCH = 14
EXPECTED_IMAGE_SIZE = (112, 48)
EXPECTED_SEED = 42
THRESHOLD_CANDIDATES = tuple(value / 100 for value in range(50, 91, 5))
MIN_ACCEPTED_ACCURACY = 0.90
MIN_VALIDATION_COVERAGE = 0.50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_recorded_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def select_validation_threshold(
    predictions: pd.DataFrame,
    classes: list[str],
) -> tuple[float, pd.DataFrame, dict]:
    """Choose an uncertainty threshold using validation predictions only."""
    if len(predictions) != 600:
        raise AssertionError(
            f"Expected 600 validation predictions; found {len(predictions)}"
        )
    if set(predictions["split"]) != {"validation"}:
        raise AssertionError(
            "Threshold selection input must contain validation rows only"
        )

    probability_columns = [f"probability_{name}" for name in classes]
    missing = set(probability_columns) - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Validation predictions are missing columns: {sorted(missing)}"
        )
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    predicted_indices = probabilities.argmax(axis=1)
    predicted_classes = np.asarray(classes)[predicted_indices]
    if not np.array_equal(
        predicted_classes,
        predictions["predicted_category"].to_numpy(),
    ):
        raise AssertionError(
            "Saved validation labels do not match probability argmax"
        )

    confidence = probabilities.max(axis=1)
    correct = (
        predictions["category"].to_numpy()
        == predictions["predicted_category"].to_numpy()
    )
    rows: list[dict] = []
    for threshold in THRESHOLD_CANDIDATES:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        accepted_accuracy = (
            float(correct[accepted].mean()) if accepted_count else None
        )
        coverage = float(accepted.mean())
        qualifies = bool(
            accepted_accuracy is not None
            and accepted_accuracy >= MIN_ACCEPTED_ACCURACY
            and coverage >= MIN_VALIDATION_COVERAGE
        )
        rows.append(
            {
                "threshold": threshold,
                "validation_rows": len(predictions),
                "accepted_predictions": accepted_count,
                "uncertain_predictions": int((~accepted).sum()),
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "qualifies": qualifies,
            }
        )
    analysis = pd.DataFrame(rows)
    qualifying = analysis.loc[analysis["qualifies"]]
    if qualifying.empty:
        raise RuntimeError(
            "No validation-only uncertainty threshold met the frozen rule"
        )
    selected = qualifying.sort_values("threshold").iloc[0]
    summary = {
        "method": (
            "Choose the lowest candidate threshold with at least 90% "
            "accepted-prediction accuracy and at least 50% validation "
            "coverage."
        ),
        "source_split": "validation",
        "candidate_thresholds": list(THRESHOLD_CANDIDATES),
        "minimum_accepted_accuracy": MIN_ACCEPTED_ACCURACY,
        "minimum_coverage": MIN_VALIDATION_COVERAGE,
        "selected_threshold": float(selected["threshold"]),
        "validation_rows": int(selected["validation_rows"]),
        "validation_accepted_predictions": int(
            selected["accepted_predictions"]
        ),
        "validation_uncertain_predictions": int(
            selected["uncertain_predictions"]
        ),
        "validation_coverage": float(selected["coverage"]),
        "validation_accepted_accuracy": float(
            selected["accepted_accuracy"]
        ),
        "test_data_used": False,
    }
    return float(selected["threshold"]), analysis, summary


def _selected_comparison_row(
    comparison_path: Path,
) -> tuple[pd.Series, pd.DataFrame]:
    comparison = pd.read_csv(comparison_path)
    selected = comparison.loc[
        comparison["run_name"] == EXPECTED_SELECTED_RUN
    ]
    if len(selected) != 1:
        raise AssertionError(
            f"Expected one {EXPECTED_SELECTED_RUN!r} comparison row"
        )
    row = selected.iloc[0]
    expected = {
        "learning_rate": EXPECTED_LEARNING_RATE,
        "width": EXPECTED_WIDTH,
        "dropout": EXPECTED_DROPOUT,
        "best_epoch": EXPECTED_BEST_EPOCH,
        "image_width": EXPECTED_IMAGE_SIZE[0],
        "image_height": EXPECTED_IMAGE_SIZE[1],
    }
    for field, value in expected.items():
        if not np.isclose(float(row[field]), float(value)):
            raise AssertionError(
                f"Selected run {field} changed: expected {value}, "
                f"found {row[field]}"
            )
    return row, comparison


def prepare_final_evaluation(
    *,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    comparison_path: str | Path,
    validation_predictions_path: str | Path,
    family_split_path: str | Path,
    manifest_path: str | Path,
    report_dir: str | Path,
) -> dict:
    """Freeze the selected model and threshold without reading test rows."""
    root = project_root()
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(metadata_path)
    comparison_path = Path(comparison_path)
    validation_predictions_path = Path(validation_predictions_path)
    family_split_path = Path(family_split_path)
    manifest_path = Path(manifest_path)
    report_dir = Path(report_dir)
    freeze_path = report_dir / "pre_test_freeze.json"
    receipt_path = report_dir / "evaluation_receipt.json"
    if freeze_path.exists():
        raise FileExistsError(
            "Pre-test freeze already exists; refusing to overwrite it"
        )
    if receipt_path.exists():
        raise FileExistsError(
            "Final evaluation receipt already exists; refusing to prepare again"
        )

    model, checkpoint = load_cnn_checkpoint(checkpoint_path)
    del model
    metadata = load_json(metadata_path)
    selected_row, _ = _selected_comparison_row(comparison_path)
    classes = list(checkpoint["classes"])
    if classes != list(EXPECTED_CLASSES):
        raise AssertionError(
            f"Class order changed: expected {EXPECTED_CLASSES}, found {classes}"
        )
    architecture = checkpoint["architecture"]
    preprocessing = checkpoint["preprocessing"]
    selected_run = checkpoint["selected_validation_run"]
    training_data = checkpoint["training_data"]
    if int(architecture["width"]) != EXPECTED_WIDTH:
        raise AssertionError("Checkpoint filter width changed")
    if not np.isclose(
        float(architecture["dropout"]),
        EXPECTED_DROPOUT,
    ):
        raise AssertionError("Checkpoint dropout changed")
    if tuple(preprocessing["image_size"]) != EXPECTED_IMAGE_SIZE:
        raise AssertionError("Checkpoint image size changed")
    if preprocessing["grayscale"] is not True:
        raise AssertionError("Selected checkpoint must use grayscale input")
    if selected_run["name"] != EXPECTED_SELECTED_RUN:
        raise AssertionError("Checkpoint selected run name changed")
    if int(selected_run["best_epoch"]) != EXPECTED_BEST_EPOCH:
        raise AssertionError("Checkpoint selected epoch changed")
    if int(checkpoint["seed"]) != EXPECTED_SEED:
        raise AssertionError("Checkpoint random seed changed")
    if training_data["fit_splits"] != ["train"]:
        raise AssertionError("Checkpoint was not fitted only on train")
    if training_data["selection_split"] != "validation":
        raise AssertionError("Checkpoint was not selected on validation")
    if (
        training_data["test_images_loaded"] != 0
        or training_data["test_rows_evaluated"] != 0
        or training_data["test_metrics_recorded"] is not False
    ):
        raise AssertionError(
            "Checkpoint metadata indicates test access before selection"
        )
    if metadata["best_validation_run"]["name"] != EXPECTED_SELECTED_RUN:
        raise AssertionError("Saved CNN metadata selected run changed")
    if not np.isclose(
        float(selected_row["validation_macro_f1"]),
        float(selected_run["validation_macro_f1"]),
    ):
        raise AssertionError(
            "Checkpoint and comparison validation macro F1 differ"
        )

    validation_predictions = pd.read_csv(validation_predictions_path)
    threshold, threshold_analysis, threshold_summary = (
        select_validation_threshold(validation_predictions, classes)
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = report_dir / "validation_threshold_analysis.csv"
    threshold_analysis.to_csv(threshold_path, index=False)

    freeze = {
        "status": "prepared",
        "purpose": (
            "Immutable record created before loading the final test split."
        ),
        "prepared_at_utc": utc_now(),
        "selected_model": {
            "name": EXPECTED_SELECTED_RUN,
            "checkpoint": display_path(checkpoint_path, root),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metadata": display_path(metadata_path, root),
            "metadata_sha256": sha256_file(metadata_path),
            "comparison": display_path(comparison_path, root),
            "comparison_sha256": sha256_file(comparison_path),
            "learning_rate": EXPECTED_LEARNING_RATE,
            "filters": EXPECTED_WIDTH,
            "dropout": EXPECTED_DROPOUT,
            "selected_checkpoint_epoch": EXPECTED_BEST_EPOCH,
            "selection_metric": "validation_macro_f1",
            "selected_validation_macro_f1": float(
                selected_row["validation_macro_f1"]
            ),
            "selected_validation_accuracy": float(
                selected_row["validation_accuracy"]
            ),
        },
        "preprocessing": preprocessing,
        "class_order": classes,
        "random_seed": EXPECTED_SEED,
        "frozen_family_split": {
            "path": display_path(family_split_path, root),
            "sha256": sha256_file(family_split_path),
        },
        "full_manifest": {
            "path": display_path(manifest_path, root),
            "sha256": sha256_file(manifest_path),
        },
        "uncertainty_threshold": {
            **threshold_summary,
            "validation_predictions": display_path(
                validation_predictions_path,
                root,
            ),
            "validation_predictions_sha256": sha256_file(
                validation_predictions_path
            ),
            "analysis_path": display_path(threshold_path, root),
        },
        "test_access_before_freeze": {
            "test_manifest_rows_loaded": 0,
            "test_images_loaded": 0,
            "test_predictions_made": 0,
            "test_metrics_recorded": False,
        },
        "expected_final_test": {
            "images": EXPECTED_TEST_IMAGES,
            "images_per_category": EXPECTED_TEST_IMAGES_PER_CLASS,
            "families_per_category": EXPECTED_TEST_FAMILIES_PER_CLASS,
            "categories": list(EXPECTED_CLASSES),
        },
    }
    if not np.isclose(
        freeze["uncertainty_threshold"]["selected_threshold"],
        threshold,
    ):
        raise AssertionError("Threshold freeze failed")
    save_json(freeze, freeze_path)
    return freeze


def _verify_frozen_contract(
    freeze: dict,
    freeze_path: Path,
    root: Path,
) -> tuple[Path, Path, Path, float]:
    if freeze["status"] != "prepared":
        raise AssertionError("Pre-test freeze status is not prepared")
    checkpoint_path = resolve_recorded_path(
        freeze["selected_model"]["checkpoint"],
        root,
    )
    split_path = resolve_recorded_path(
        freeze["frozen_family_split"]["path"],
        root,
    )
    manifest_path = resolve_recorded_path(
        freeze["full_manifest"]["path"],
        root,
    )
    expected_hashes = (
        (
            checkpoint_path,
            freeze["selected_model"]["checkpoint_sha256"],
            "checkpoint",
        ),
        (
            split_path,
            freeze["frozen_family_split"]["sha256"],
            "family split",
        ),
        (
            manifest_path,
            freeze["full_manifest"]["sha256"],
            "manifest",
        ),
    )
    for path, expected_hash, label in expected_hashes:
        observed = sha256_file(path)
        if observed != expected_hash:
            raise AssertionError(
                f"Frozen {label} hash changed: expected {expected_hash}, "
                f"found {observed}"
            )
    threshold = float(
        freeze["uncertainty_threshold"]["selected_threshold"]
    )
    if freeze["uncertainty_threshold"]["source_split"] != "validation":
        raise AssertionError("Threshold was not selected on validation")
    if freeze["uncertainty_threshold"]["test_data_used"] is not False:
        raise AssertionError("Threshold record indicates test-data use")
    if freeze_path.parent.resolve() == root.resolve():
        raise AssertionError("Use a dedicated final-evaluation report folder")
    return checkpoint_path, split_path, manifest_path, threshold


def _load_and_validate_test_manifest(
    manifest_path: Path,
    family_split_path: Path,
    root: Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "family", "category", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    assert_no_family_leakage(manifest)

    frozen = pd.read_csv(
        family_split_path,
        usecols=["family", "category", "split"],
    )
    assert_no_family_leakage(frozen)
    observed_assignments = (
        manifest[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values(["family", "category", "split"])
        .reset_index(drop=True)
    )
    frozen_assignments = (
        frozen[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values(["family", "category", "split"])
        .reset_index(drop=True)
    )
    if not observed_assignments.equals(frozen_assignments):
        raise AssertionError(
            "Manifest family assignments differ from the frozen split"
        )

    test = manifest.loc[manifest["split"] == "test"].copy()
    if len(test) != EXPECTED_TEST_IMAGES:
        raise AssertionError(
            f"Expected {EXPECTED_TEST_IMAGES} test images; found {len(test)}"
        )
    category_counts = test["category"].value_counts().sort_index()
    expected_counts = pd.Series(
        EXPECTED_TEST_IMAGES_PER_CLASS,
        index=list(EXPECTED_CLASSES),
        dtype="int64",
    )
    if not category_counts.equals(expected_counts):
        raise AssertionError(
            f"Test category counts changed: {category_counts.to_dict()}"
        )
    family_counts = (
        test.groupby("category")["family"].nunique().sort_index()
    )
    expected_families = pd.Series(
        EXPECTED_TEST_FAMILIES_PER_CLASS,
        index=list(EXPECTED_CLASSES),
        dtype="int64",
    )
    if not family_counts.equals(expected_families):
        raise AssertionError(
            f"Test family counts changed: {family_counts.to_dict()}"
        )

    def resolve_image(value: str) -> str:
        path = Path(value)
        resolved = path if path.is_absolute() else root / path
        return str(resolved.resolve())

    test["resolved_image_path"] = test["image_path"].map(resolve_image)
    missing_images = [
        value
        for value in test["resolved_image_path"]
        if not Path(value).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"Missing final test image: {missing_images[0]}"
        )
    return test.reset_index(drop=True)


def _confidence_distribution(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    edges = np.linspace(0.0, 1.0, 11)
    rows: list[dict] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = predictions["confidence"].between(
                lower,
                upper,
                inclusive="both",
            )
        else:
            mask = (
                (predictions["confidence"] >= lower)
                & (predictions["confidence"] < upper)
            )
        subset = predictions.loc[mask]
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "predictions": len(subset),
                "correct_predictions": int(subset["correct"].sum()),
                "incorrect_predictions": int((~subset["correct"]).sum()),
                "accuracy": (
                    float(subset["correct"].mean())
                    if len(subset)
                    else None
                ),
            }
        )
    summary: dict[str, dict] = {}
    for name, subset in (
        ("all", predictions),
        ("correct", predictions.loc[predictions["correct"]]),
        ("incorrect", predictions.loc[~predictions["correct"]]),
    ):
        values = subset["confidence"]
        summary[name] = {
            "count": len(subset),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(values.median()) if len(values) else None,
            "minimum": float(values.min()) if len(values) else None,
            "q25": float(values.quantile(0.25)) if len(values) else None,
            "q75": float(values.quantile(0.75)) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
        }
    return pd.DataFrame(rows), summary


def _save_confusion_figure(
    matrix: np.ndarray,
    classes: list[str],
    path: Path,
) -> None:
    labels = [name.replace("_", " ").title() for name in classes]
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = int(matrix[row, column])
            color = (
                "white"
                if value > matrix.max() * 0.55
                else "#182230"
            )
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    ax.set_xticks(range(len(classes)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Final CNN test confusion matrix", fontsize=16, pad=28)
    ax.text(
        0.5,
        1.02,
        "600 images from 15 held-out font families",
        transform=ax.transAxes,
        ha="center",
        color="#526072",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Test images")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confidence_figure(
    predictions: pd.DataFrame,
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        predictions.loc[predictions["correct"], "confidence"],
        bins=bins,
        color="#2f6da4",
        alpha=0.8,
        label="Correct",
        edgecolor="white",
    )
    ax.hist(
        predictions.loc[~predictions["correct"], "confidence"],
        bins=bins,
        color="#d97757",
        alpha=0.8,
        label="Incorrect",
        edgecolor="white",
    )
    ax.axvline(
        threshold,
        color="#27313f",
        linestyle="--",
        linewidth=2,
        label=f"Validation-only threshold ({threshold:.2f})",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Maximum predicted probability")
    ax.set_ylabel("Test predictions")
    ax.set_title("Final CNN test confidence distribution", pad=28)
    ax.text(
        0.5,
        1.02,
        "Correct and incorrect predictions; 600 held-out images",
        transform=ax.transAxes,
        ha="center",
        color="#526072",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_uncertainty_figure(
    uncertainty: pd.DataFrame,
    path: Path,
) -> None:
    labels = (
        uncertainty["category"].str.replace("_", " ").str.title().tolist()
    )
    y = np.arange(len(uncertainty))
    accepted = uncertainty["accepted_predictions"].to_numpy()
    uncertain = uncertainty["uncertain_predictions"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        y,
        accepted,
        color="#2f6da4",
        edgecolor="#1f4f78",
        label="Accepted",
    )
    ax.barh(
        y,
        uncertain,
        left=accepted,
        color="#d9e4ef",
        edgecolor="#708399",
        label="Uncertain",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, EXPECTED_TEST_IMAGES_PER_CLASS)
    ax.set_xlabel("Test predictions")
    ax.set_title("Accepted and uncertain predictions by category", pad=28)
    ax.text(
        0.5,
        1.02,
        "Validation-only threshold; 120 held-out images per category",
        transform=ax.transAxes,
        ha="center",
        color="#526072",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#d9dee7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_final_test(
    freeze_path: str | Path,
    *,
    report_dir: str | Path | None = None,
) -> dict:
    """Run the selected checkpoint once on the frozen final test images."""
    root = project_root()
    freeze_path = Path(freeze_path)
    freeze = load_json(freeze_path)
    if report_dir is None:
        report_dir = freeze_path.parent
    report_dir = Path(report_dir)
    receipt_path = report_dir / "evaluation_receipt.json"
    metrics_path = report_dir / "final_test_metrics.json"
    if receipt_path.exists() or metrics_path.exists():
        raise RuntimeError(
            "Final evaluation has already started or completed; "
            "refusing to evaluate the test set again"
        )

    checkpoint_path, split_path, manifest_path, threshold = (
        _verify_frozen_contract(freeze, freeze_path, root)
    )
    model, checkpoint = load_cnn_checkpoint(checkpoint_path)
    classes = list(checkpoint["classes"])
    if classes != freeze["class_order"]:
        raise AssertionError("Checkpoint class order differs from freeze")
    preprocessing = checkpoint["preprocessing"]
    if preprocessing != freeze["preprocessing"]:
        raise AssertionError("Checkpoint preprocessing differs from freeze")
    if tuple(preprocessing["image_size"]) != EXPECTED_IMAGE_SIZE:
        raise AssertionError("Frozen image size changed")

    test = _load_and_validate_test_manifest(
        manifest_path,
        split_path,
        root,
    )
    set_seed(int(freeze["random_seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model.eval()
    transform = build_image_transform(
        tuple(preprocessing["image_size"]),
        training=False,
        augmentation={"enabled": False},
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "started",
        "started_at_utc": utc_now(),
        "purpose": "Single final evaluation on the frozen test families.",
        "freeze_record": display_path(freeze_path, root),
        "freeze_record_sha256": sha256_file(freeze_path),
        "test_images_expected": EXPECTED_TEST_IMAGES,
        "rerun_allowed": False,
    }
    save_json(receipt, receipt_path)

    rows: list[dict] = []
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for row in test.itertuples(index=False):
            inference_started = time.perf_counter()
            with Image.open(row.resolved_image_path) as image:
                prepared = ImageOps.exif_transpose(image).convert("RGB")
                tensor = transform(prepared).unsqueeze(0)
            probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()
            inference_ms = (
                time.perf_counter() - inference_started
            ) * 1000
            best_index = int(probabilities.argmax())
            prediction = classes[best_index]
            confidence = float(probabilities[best_index])
            result = {
                "image_path": row.image_path,
                "family": row.family,
                "category": row.category,
                "split": row.split,
                "predicted_category": prediction,
                "confidence": confidence,
                "uncertain": confidence < threshold,
                "correct": prediction == row.category,
                "inference_ms": inference_ms,
            }
            for class_index, class_name in enumerate(classes):
                result[f"probability_{class_name}"] = float(
                    probabilities[class_index]
                )
            rows.append(result)
    total_evaluation_seconds = time.perf_counter() - evaluation_started
    predictions = pd.DataFrame(rows)

    y_true = predictions["category"]
    y_pred = predictions["predicted_category"]
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=classes,
            average="macro",
            zero_division=0,
        )
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    classification_rows = [
        {
            "class": name,
            "precision": float(report_dict[name]["precision"]),
            "recall": float(report_dict[name]["recall"]),
            "f1": float(report_dict[name]["f1-score"]),
            "support": int(report_dict[name]["support"]),
        }
        for name in classes
    ]
    classification_frame = pd.DataFrame(classification_rows)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)

    accepted = ~predictions["uncertain"]
    accepted_count = int(accepted.sum())
    uncertain_count = int(predictions["uncertain"].sum())
    accepted_accuracy = (
        float(predictions.loc[accepted, "correct"].mean())
        if accepted_count
        else None
    )
    uncertainty_rows: list[dict] = []
    for category in classes:
        subset = predictions.loc[predictions["category"] == category]
        subset_accepted = ~subset["uncertain"]
        category_accepted = int(subset_accepted.sum())
        uncertainty_rows.append(
            {
                "category": category,
                "test_images": len(subset),
                "accepted_predictions": category_accepted,
                "uncertain_predictions": int(
                    subset["uncertain"].sum()
                ),
                "coverage": float(subset_accepted.mean()),
                "accepted_correct": int(
                    subset.loc[subset_accepted, "correct"].sum()
                ),
                "accepted_accuracy": (
                    float(
                        subset.loc[subset_accepted, "correct"].mean()
                    )
                    if category_accepted
                    else None
                ),
            }
        )
    uncertainty_frame = pd.DataFrame(uncertainty_rows)
    distribution_frame, confidence_summary = _confidence_distribution(
        predictions
    )

    errors = predictions.loc[~predictions["correct"]].copy()
    confident_mistakes = errors.loc[~errors["uncertain"]].sort_values(
        "confidence",
        ascending=False,
    )
    low_confidence_mistakes = errors.loc[errors["uncertain"]].sort_values(
        "confidence"
    )
    representative_errors = pd.concat(
        [
            confident_mistakes.head(5).assign(
                example_group="confident_mistake"
            ),
            low_confidence_mistakes.head(5).assign(
                example_group="low_confidence_mistake"
            ),
        ],
        ignore_index=True,
    )
    error_groups = (
        errors.groupby(["category", "predicted_category"])
        .size()
        .rename("errors")
        .reset_index()
        .sort_values(
            ["errors", "category", "predicted_category"],
            ascending=[False, True, True],
        )
    )

    predictions_path = report_dir / "final_test_predictions.csv"
    classification_path = report_dir / "final_classification_report.csv"
    confusion_path = report_dir / "final_confusion_matrix.csv"
    distribution_path = report_dir / "confidence_distribution.csv"
    uncertainty_path = report_dir / "uncertainty_by_class.csv"
    error_groups_path = report_dir / "errors_by_true_and_predicted.csv"
    confident_path = report_dir / "confident_mistakes.csv"
    low_confidence_path = report_dir / "low_confidence_mistakes.csv"
    representative_path = report_dir / "representative_error_examples.csv"
    predictions.to_csv(predictions_path, index=False)
    classification_frame.to_csv(classification_path, index=False)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        confusion_path
    )
    distribution_frame.to_csv(distribution_path, index=False)
    uncertainty_frame.to_csv(uncertainty_path, index=False)
    error_groups.to_csv(error_groups_path, index=False)
    confident_mistakes.to_csv(confident_path, index=False)
    low_confidence_mistakes.to_csv(low_confidence_path, index=False)
    representative_errors.to_csv(representative_path, index=False)

    confusion_figure = figures_dir / "final_test_confusion_matrix.png"
    confidence_figure = figures_dir / "final_test_confidence_distribution.png"
    uncertainty_figure = figures_dir / "final_test_uncertainty_by_class.png"
    _save_confusion_figure(matrix, classes, confusion_figure)
    _save_confidence_figure(predictions, threshold, confidence_figure)
    _save_uncertainty_figure(uncertainty_frame, uncertainty_figure)

    split_hash_after = sha256_file(split_path)
    manifest_hash_after = sha256_file(manifest_path)
    checkpoint_hash_after = sha256_file(checkpoint_path)
    metrics = {
        "status": "completed",
        "purpose": (
            "Single final CNN evaluation on previously untouched test "
            "families; no post-test tuning was performed."
        ),
        "completed_at_utc": utc_now(),
        "selected_model": freeze["selected_model"],
        "preprocessing": preprocessing,
        "class_order": classes,
        "random_seed": int(freeze["random_seed"]),
        "uncertainty_threshold": {
            "value": threshold,
            "selected_using": "validation only",
            "selection_rule": freeze["uncertainty_threshold"]["method"],
            "test_results_used_for_selection": False,
        },
        "test_results": {
            "images": len(predictions),
            "families": int(test["family"].nunique()),
            "images_per_category": (
                test["category"].value_counts().sort_index().to_dict()
            ),
            "families_per_category": (
                test.groupby("category")["family"]
                .nunique()
                .sort_index()
                .to_dict()
            ),
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "correct_predictions": int(predictions["correct"].sum()),
            "incorrect_predictions": int((~predictions["correct"]).sum()),
            "inference_seconds_total": total_evaluation_seconds,
            "inference_ms_per_image": float(
                predictions["inference_ms"].mean()
            ),
            "model_size_bytes": checkpoint_path.stat().st_size,
        },
        "uncertainty_results": {
            "accepted_predictions": accepted_count,
            "uncertain_predictions": uncertain_count,
            "coverage": float(accepted.mean()),
            "accepted_accuracy": accepted_accuracy,
            "confident_mistakes": len(confident_mistakes),
            "low_confidence_mistakes": len(low_confidence_mistakes),
            "confidence_distribution": confidence_summary,
        },
        "checks": {
            "exactly_600_test_images": len(predictions)
            == EXPECTED_TEST_IMAGES,
            "exactly_120_images_per_category": bool(
                (
                    test["category"].value_counts()
                    == EXPECTED_TEST_IMAGES_PER_CLASS
                ).all()
            ),
            "exactly_3_test_families_per_category": bool(
                (
                    test.groupby("category")["family"].nunique()
                    == EXPECTED_TEST_FAMILIES_PER_CLASS
                ).all()
            ),
            "family_overlap_count": 0,
            "training_splits": checkpoint["training_data"]["fit_splits"],
            "selection_split": checkpoint["training_data"][
                "selection_split"
            ],
            "test_used_for_training": False,
            "test_used_for_validation": False,
            "test_used_for_model_selection": False,
            "test_used_for_early_stopping": False,
            "test_used_for_threshold_selection": False,
            "checkpoint_loaded": True,
            "preprocessing_loaded": True,
            "checkpoint_hash_unchanged": (
                checkpoint_hash_after
                == freeze["selected_model"]["checkpoint_sha256"]
            ),
            "frozen_split_hash_unchanged": (
                split_hash_after
                == freeze["frozen_family_split"]["sha256"]
            ),
            "manifest_hash_unchanged": (
                manifest_hash_after
                == freeze["full_manifest"]["sha256"]
            ),
            "post_test_retraining_or_tuning": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "outputs": {
            "predictions": display_path(predictions_path, root),
            "classification_report": display_path(
                classification_path,
                root,
            ),
            "confusion_matrix_csv": display_path(confusion_path, root),
            "confidence_distribution": display_path(
                distribution_path,
                root,
            ),
            "uncertainty_by_class": display_path(
                uncertainty_path,
                root,
            ),
            "errors_by_true_and_predicted": display_path(
                error_groups_path,
                root,
            ),
            "confident_mistakes": display_path(confident_path, root),
            "low_confidence_mistakes": display_path(
                low_confidence_path,
                root,
            ),
            "representative_error_examples": display_path(
                representative_path,
                root,
            ),
            "figures": [
                display_path(confusion_figure, root),
                display_path(confidence_figure, root),
                display_path(uncertainty_figure, root),
            ],
        },
    }
    required_true = (
        "exactly_600_test_images",
        "exactly_120_images_per_category",
        "exactly_3_test_families_per_category",
        "checkpoint_loaded",
        "preprocessing_loaded",
        "checkpoint_hash_unchanged",
        "frozen_split_hash_unchanged",
        "manifest_hash_unchanged",
    )
    required_false = (
        "test_used_for_training",
        "test_used_for_validation",
        "test_used_for_model_selection",
        "test_used_for_early_stopping",
        "test_used_for_threshold_selection",
        "post_test_retraining_or_tuning",
    )
    if not all(metrics["checks"][key] is True for key in required_true):
        raise AssertionError("One or more final evaluation checks failed")
    if not all(metrics["checks"][key] is False for key in required_false):
        raise AssertionError("One or more final evaluation checks failed")
    save_json(metrics, metrics_path)
    receipt.update(
        {
            "status": "completed",
            "completed_at_utc": metrics["completed_at_utc"],
            "test_images_evaluated": len(predictions),
            "metrics": display_path(metrics_path, root),
            "metrics_sha256": sha256_file(metrics_path),
            "rerun_allowed": False,
        }
    )
    save_json(receipt, receipt_path)
    return metrics


def _defaults() -> dict[str, Path]:
    root = project_root()
    return {
        "checkpoint": root / "artifacts" / "cnn" / "cnn_model.pt",
        "metadata": root / "artifacts" / "cnn" / "cnn_metadata.json",
        "comparison": (
            root / "reports" / "cnn" / "cnn_experiment_comparison.csv"
        ),
        "validation_predictions": (
            root
            / "reports"
            / "cnn"
            / "best_cnn_validation_predictions.csv"
        ),
        "family_split": (
            root
            / "data"
            / "interim"
            / "google_fonts_final_family_split.csv"
        ),
        "manifest": (
            root / "reports" / "dataset" / "full_manifest.csv"
        ),
        "report_dir": root / "reports" / "final_evaluation",
    }


def main() -> None:
    defaults = _defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and run the one-time final FontSense CNN evaluation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Freeze checkpoint, preprocessing, hashes, and threshold.",
    )
    prepare_parser.add_argument(
        "--checkpoint",
        default=str(defaults["checkpoint"]),
    )
    prepare_parser.add_argument(
        "--metadata",
        default=str(defaults["metadata"]),
    )
    prepare_parser.add_argument(
        "--comparison",
        default=str(defaults["comparison"]),
    )
    prepare_parser.add_argument(
        "--validation-predictions",
        default=str(defaults["validation_predictions"]),
    )
    prepare_parser.add_argument(
        "--family-split",
        default=str(defaults["family_split"]),
    )
    prepare_parser.add_argument(
        "--manifest",
        default=str(defaults["manifest"]),
    )
    prepare_parser.add_argument(
        "--report-dir",
        default=str(defaults["report_dir"]),
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate once using the immutable pre-test freeze.",
    )
    evaluate_parser.add_argument(
        "--freeze",
        default=str(defaults["report_dir"] / "pre_test_freeze.json"),
    )
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_final_evaluation(
            checkpoint_path=args.checkpoint,
            metadata_path=args.metadata,
            comparison_path=args.comparison,
            validation_predictions_path=args.validation_predictions,
            family_split_path=args.family_split,
            manifest_path=args.manifest,
            report_dir=args.report_dir,
        )
    else:
        result = evaluate_final_test(args.freeze)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
