from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .features import HOGTransformer
from .mlflow_utils import optional_mlflow_run
from .split import assert_no_family_leakage
from .utils import load_json, project_root, save_json, set_seed

matplotlib.use("Agg")
from matplotlib import pyplot as plt


EXPECTED_CATEGORIES = (
    "display",
    "handwriting",
    "monospace",
    "sans_serif",
    "serif",
)
REQUIRED_MANIFEST_COLUMNS = {"image_path", "family", "category", "split"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classification_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for class_name in class_names:
        values = report[class_name]
        rows.append(
            {
                "class": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    labels = [name.replace("_", " ").title() for name in class_names]
    fig, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Validation images")
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="True category",
    )
    fig.suptitle(
        "Best HOG validation confusion matrix",
        x=0.5,
        y=0.985,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.947,
        "600 validation images from unseen families; no test images evaluated",
        ha="center",
        va="top",
        fontsize=9,
        color="#4b5563",
    )
    maximum = max(int(matrix.max()), 1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > maximum / 2 else "#172033",
                fontsize=10,
            )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _resolve_image_paths(frame: pd.DataFrame, root: Path) -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    for raw_path in frame["image_path"].astype(str):
        path = Path(raw_path)
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file():
            missing.append(raw_path)
        resolved.append(str(absolute.resolve()))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected train/validation images are missing; "
            f"first missing path: {missing[0]}"
        )
    return resolved


def _check_manifest_and_split(
    manifest: pd.DataFrame,
    family_split_path: Path | None,
    expected_split_hash: str | None,
) -> dict:
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError("Manifest contains zero rows")
    assert_no_family_leakage(manifest)

    observed_categories = tuple(sorted(manifest["category"].astype(str).unique()))
    if observed_categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"Expected categories {list(EXPECTED_CATEGORIES)}; "
            f"found {list(observed_categories)}"
        )
    required_splits = {"train", "validation", "test"}
    observed_splits = set(manifest["split"].astype(str).unique())
    if observed_splits != required_splits:
        raise ValueError(
            f"Expected manifest splits {sorted(required_splits)}; "
            f"found {sorted(observed_splits)}"
        )

    result = {
        "family_overlap_count": 0,
        "family_assignments_match_frozen_split": None,
        "frozen_split_sha256": None,
    }
    if family_split_path is None:
        return result
    if not family_split_path.is_file():
        raise FileNotFoundError(f"Frozen family split not found: {family_split_path}")

    split_hash = sha256_file(family_split_path)
    if expected_split_hash and split_hash != expected_split_hash:
        raise AssertionError(
            "Frozen family split hash changed: "
            f"expected {expected_split_hash}, found {split_hash}"
        )
    frozen = pd.read_csv(family_split_path)
    required_frozen = {"family", "category", "split"}
    missing_frozen = required_frozen - set(frozen.columns)
    if missing_frozen:
        raise ValueError(
            f"Frozen family split is missing columns: {sorted(missing_frozen)}"
        )
    manifest_assignments = (
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
    if not manifest_assignments.equals(frozen_assignments):
        raise AssertionError(
            "Full manifest family assignments do not match the frozen split"
        )
    result.update(
        {
            "family_assignments_match_frozen_split": True,
            "frozen_split_sha256": split_hash,
        }
    )
    return result


def _evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    prediction = probabilities.argmax(axis=1)
    metrics = {
        "validation_accuracy": float(accuracy_score(y_true, prediction)),
        "validation_macro_f1": float(
            f1_score(y_true, prediction, average="macro")
        ),
    }
    return prediction, metrics


def _log_model_artifacts(
    mlflow,
    model,
    encoder: LabelEncoder,
    per_class: pd.DataFrame,
    matrix: np.ndarray,
) -> int:
    with tempfile.TemporaryDirectory(prefix="fontsense_hog_") as temporary:
        temporary_dir = Path(temporary)
        model_path = temporary_dir / "hog_pipeline.joblib"
        encoder_path = temporary_dir / "label_encoder.joblib"
        per_class_path = temporary_dir / "classification_report.csv"
        matrix_path = temporary_dir / "confusion_matrix.csv"
        joblib.dump(model, model_path, compress=3)
        joblib.dump(encoder, encoder_path, compress=3)
        per_class.to_csv(per_class_path, index=False)
        pd.DataFrame(matrix).to_csv(matrix_path, index=False, header=False)
        model_size_bytes = model_path.stat().st_size
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(encoder_path), artifact_path="model")
        mlflow.log_artifact(str(per_class_path), artifact_path="validation")
        mlflow.log_artifact(str(matrix_path), artifact_path="validation")
    return model_size_bytes


def train_hog(
    manifest_path: str | Path,
    experiment_config_path: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    seed: int = 42,
    family_split_path: str | Path | None = None,
    tracking_dir: str | Path = "mlruns",
    enable_mlflow: bool = True,
    figure_dir: str | Path | None = None,
) -> dict:
    """Train and compare validation-only majority and HOG baselines."""
    set_seed(seed)
    root = project_root()
    manifest_path = Path(manifest_path)
    config = load_json(experiment_config_path)
    manifest_hash_before = sha256_file(manifest_path)
    manifest = pd.read_csv(manifest_path)

    if family_split_path is None and manifest_path.resolve() == (
        root / "reports" / "dataset" / "full_manifest.csv"
    ).resolve():
        family_split_path = (
            root / "data" / "interim" / "google_fonts_final_family_split.csv"
        )
    frozen_path = Path(family_split_path) if family_split_path is not None else None
    split_check = _check_manifest_and_split(
        manifest,
        frozen_path,
        config.get("expected_split_sha256") if frozen_path else None,
    )

    train = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    validation = manifest.loc[
        manifest["split"] == "validation"
    ].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Both train and validation rows are required")

    encoder = LabelEncoder().fit(train["category"].astype(str))
    if encoder.classes_.tolist() != list(EXPECTED_CATEGORIES):
        raise ValueError(
            "Training split must contain all five categories; "
            f"found {encoder.classes_.tolist()}"
        )
    unknown_validation = sorted(
        set(validation["category"].astype(str)) - set(encoder.classes_)
    )
    if unknown_validation:
        raise ValueError(
            f"Validation contains categories absent from training: {unknown_validation}"
        )
    y_train = encoder.transform(train["category"].astype(str))
    y_validation = encoder.transform(validation["category"].astype(str))
    train_paths = _resolve_image_paths(train, root)
    validation_paths = _resolve_image_paths(validation, root)

    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = (
        Path(figure_dir)
        if figure_dir is not None
        else root / "reports" / "figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = str(
        config.get("experiment_name", "FontSense HOG baselines")
    )
    tracking_dir = Path(tracking_dir)
    comparison_rows: list[dict] = []
    per_run_reports: dict[str, pd.DataFrame] = {}
    per_run_matrices: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Pipeline] = {}

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_x_train = np.zeros((len(train), 1), dtype=np.float32)
    majority_x_validation = np.zeros((len(validation), 1), dtype=np.float32)
    majority_fit_started = time.perf_counter()
    majority_model.fit(majority_x_train, y_train)
    majority_training_seconds = time.perf_counter() - majority_fit_started
    majority_inference_started = time.perf_counter()
    majority_probabilities = majority_model.predict_proba(
        majority_x_validation
    )
    majority_inference_seconds = (
        time.perf_counter() - majority_inference_started
    )
    majority_prediction, majority_metrics = _evaluate_predictions(
        y_validation, majority_probabilities
    )
    majority_report = _classification_frame(
        y_validation, majority_prediction, encoder.classes_.tolist()
    )
    majority_matrix = confusion_matrix(
        y_validation,
        majority_prediction,
        labels=np.arange(len(encoder.classes_)),
    )
    if not enable_mlflow:
        majority_run_id = ""
        majority_artifact_uri = ""
    else:
        with optional_mlflow_run(
            experiment_name,
            "Majority-class sanity check",
            tracking_dir=tracking_dir,
        ) as active_mlflow:
            if active_mlflow is None:
                raise RuntimeError(
                    "MLflow is required for baseline tracking but is not installed"
                )
            majority_size = _log_model_artifacts(
                active_mlflow,
                majority_model,
                encoder,
                majority_report,
                majority_matrix,
            )
            active_mlflow.log_params(
                {
                    "model_type": "majority_class",
                    "strategy": "most_frequent",
                    "seed": seed,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_images_loaded": 0,
                }
            )
            active_mlflow.log_metrics(
                {
                    **majority_metrics,
                    "training_seconds": majority_training_seconds,
                    "inference_seconds": majority_inference_seconds,
                    "inference_ms_per_image": (
                        majority_inference_seconds / len(validation) * 1000
                    ),
                    "model_size_bytes": majority_size,
                }
            )
            active_run = active_mlflow.active_run()
            majority_run_id = active_run.info.run_id
            majority_artifact_uri = active_run.info.artifact_uri
    if not enable_mlflow:
        with tempfile.TemporaryDirectory(prefix="fontsense_majority_") as temporary:
            path = Path(temporary) / "majority.joblib"
            joblib.dump(majority_model, path, compress=3)
            majority_size = path.stat().st_size
    comparison_rows.append(
        {
            "run_order": 0,
            "model_type": "majority_class",
            "run_name": "Majority-class sanity check",
            "reason": "Sanity check that always predicts the most common training category.",
            "image_width": None,
            "image_height": None,
            "orientations": None,
            "pixels_per_cell": None,
            "cells_per_block": None,
            "C": None,
            "solver": None,
            "train_rows": len(train),
            "validation_rows": len(validation),
            **majority_metrics,
            "feature_extraction_seconds": 0.0,
            "training_seconds": majority_training_seconds,
            "inference_seconds": majority_inference_seconds,
            "inference_ms_per_image": (
                majority_inference_seconds / len(validation) * 1000
            ),
            "model_size_bytes": majority_size,
            "preprocessing_fit_rows": 0,
            "mlflow_run_id": majority_run_id,
            "mlflow_artifact_uri": majority_artifact_uri,
        }
    )
    per_run_reports["Majority-class sanity check"] = majority_report
    per_run_matrices["Majority-class sanity check"] = majority_matrix

    feature_cache: dict[
        tuple, tuple[np.ndarray, np.ndarray, float]
    ] = {}
    logistic_config = config.get("logistic_regression", {})
    for index, experiment in enumerate(config["experiments"], start=1):
        run_name = str(experiment["name"])
        image_size = tuple(int(value) for value in experiment["image_size"])
        pixels_per_cell = tuple(
            int(value) for value in experiment["pixels_per_cell"]
        )
        cells_per_block = tuple(
            int(value) for value in config.get("cells_per_block", [2, 2])
        )
        orientations = int(config.get("orientations", 9))
        cache_key = (
            image_size,
            orientations,
            pixels_per_cell,
            cells_per_block,
        )
        transformer = HOGTransformer(
            image_size=image_size,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            show_progress=True,
        ).fit(train_paths, y_train)
        cache_reused = cache_key in feature_cache
        if cache_reused:
            x_train, x_validation, feature_seconds = feature_cache[cache_key]
        else:
            feature_started = time.perf_counter()
            x_train = transformer.transform(train_paths)
            x_validation = transformer.transform(validation_paths)
            feature_seconds = time.perf_counter() - feature_started
            feature_cache[cache_key] = (
                x_train,
                x_validation,
                feature_seconds,
            )

        classifier = LogisticRegression(
            C=float(experiment["C"]),
            solver=str(logistic_config.get("solver", "lbfgs")),
            max_iter=int(logistic_config.get("max_iter", 500)),
            tol=float(logistic_config.get("tolerance", 1e-4)),
            random_state=seed,
        )
        fit_started = time.perf_counter()
        classifier.fit(x_train, y_train)
        training_seconds = time.perf_counter() - fit_started
        pipeline = Pipeline(
            [
                ("hog", transformer),
                ("model", classifier),
            ]
        )
        inference_started = time.perf_counter()
        probabilities = pipeline.predict_proba(validation_paths)
        inference_seconds = time.perf_counter() - inference_started
        prediction, metrics = _evaluate_predictions(y_validation, probabilities)
        per_class = _classification_frame(
            y_validation, prediction, encoder.classes_.tolist()
        )
        matrix = confusion_matrix(
            y_validation,
            prediction,
            labels=np.arange(len(encoder.classes_)),
        )

        mlflow_run_id = ""
        mlflow_artifact_uri = ""
        if enable_mlflow:
            with optional_mlflow_run(
                experiment_name,
                run_name,
                tracking_dir=tracking_dir,
            ) as active_mlflow:
                if active_mlflow is None:
                    raise RuntimeError(
                        "MLflow is required for baseline tracking but is not installed"
                    )
                model_size_bytes = _log_model_artifacts(
                    active_mlflow,
                    pipeline,
                    encoder,
                    per_class,
                    matrix,
                )
                active_mlflow.log_params(
                    {
                        "model_type": "hog_multinomial_logistic_regression",
                        "reason": str(experiment["reason"]),
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "grayscale": True,
                        "orientations": orientations,
                        "pixels_per_cell": str(pixels_per_cell),
                        "cells_per_block": str(cells_per_block),
                        "block_norm": "L2-Hys",
                        "C": float(experiment["C"]),
                        "solver": classifier.solver,
                        "max_iter": classifier.max_iter,
                        "seed": seed,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "preprocessing_fit_rows": transformer.fit_sample_count_,
                        "test_images_loaded": 0,
                    }
                )
                active_mlflow.log_metrics(
                    {
                        **metrics,
                        "feature_extraction_seconds": feature_seconds,
                        "training_seconds": training_seconds,
                        "inference_seconds": inference_seconds,
                        "inference_ms_per_image": (
                            inference_seconds / len(validation) * 1000
                        ),
                        "model_size_bytes": model_size_bytes,
                    }
                )
                active_run = active_mlflow.active_run()
                mlflow_run_id = active_run.info.run_id
                mlflow_artifact_uri = active_run.info.artifact_uri
        else:
            with tempfile.TemporaryDirectory(
                prefix="fontsense_hog_size_"
            ) as temporary:
                path = Path(temporary) / "hog_pipeline.joblib"
                joblib.dump(pipeline, path, compress=3)
                model_size_bytes = path.stat().st_size

        comparison_rows.append(
            {
                "run_order": index,
                "model_type": "hog_logistic_regression",
                "run_name": run_name,
                "reason": str(experiment["reason"]),
                "image_width": image_size[0],
                "image_height": image_size[1],
                "orientations": orientations,
                "pixels_per_cell": f"{pixels_per_cell[0]}x{pixels_per_cell[1]}",
                "cells_per_block": f"{cells_per_block[0]}x{cells_per_block[1]}",
                "C": float(experiment["C"]),
                "solver": classifier.solver,
                "train_rows": len(train),
                "validation_rows": len(validation),
                **metrics,
                "feature_extraction_seconds": feature_seconds,
                "feature_cache_reused": cache_reused,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "inference_ms_per_image": (
                    inference_seconds / len(validation) * 1000
                ),
                "model_size_bytes": model_size_bytes,
                "preprocessing_fit_rows": transformer.fit_sample_count_,
                "classifier_iterations": int(classifier.n_iter_.max()),
                "mlflow_run_id": mlflow_run_id,
                "mlflow_artifact_uri": mlflow_artifact_uri,
            }
        )
        per_run_reports[run_name] = per_class
        per_run_matrices[run_name] = matrix
        fitted_models[run_name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values("run_order")
    hog_comparison = comparison.loc[
        comparison["model_type"] == "hog_logistic_regression"
    ]
    if hog_comparison.empty:
        raise RuntimeError("No HOG experiment was trained")
    best_row = hog_comparison.sort_values(
        ["validation_macro_f1", "validation_accuracy", "run_order"],
        ascending=[False, False, True],
    ).iloc[0]
    best_name = str(best_row["run_name"])
    best_model = fitted_models[best_name]

    pipeline_path = artifact_dir / "hog_pipeline.joblib"
    encoder_path = artifact_dir / "label_encoder.joblib"
    metadata_path = artifact_dir / "hog_metadata.json"
    joblib.dump(best_model, pipeline_path, compress=3)
    joblib.dump(encoder, encoder_path, compress=3)
    saved_model_size = pipeline_path.stat().st_size

    loaded_pipeline = joblib.load(pipeline_path)
    reload_probabilities = loaded_pipeline.predict_proba(
        [validation_paths[0]]
    )[0]
    if reload_probabilities.shape != (len(encoder.classes_),):
        raise AssertionError(
            "Reloaded HOG pipeline did not return all five class probabilities"
        )
    if not np.isclose(reload_probabilities.sum(), 1.0, atol=1e-6):
        raise AssertionError("Reloaded HOG probabilities do not sum to one")
    reload_prediction = encoder.inverse_transform(
        [int(reload_probabilities.argmax())]
    )[0]

    best_probabilities = best_model.predict_proba(validation_paths)
    best_prediction = best_probabilities.argmax(axis=1)
    prediction_frame = validation[
        ["image_path", "family", "category", "split"]
    ].copy()
    prediction_frame["predicted_category"] = encoder.inverse_transform(
        best_prediction
    )
    prediction_frame["correct"] = (
        prediction_frame["category"]
        == prediction_frame["predicted_category"]
    )
    for class_index, class_name in enumerate(encoder.classes_):
        prediction_frame[f"probability_{class_name}"] = best_probabilities[
            :, class_index
        ]

    comparison_path = report_dir / "validation_comparison.csv"
    mlflow_runs_path = report_dir / "mlflow_runs.csv"
    majority_report_path = report_dir / "majority_classification_report.csv"
    best_report_path = report_dir / "best_hog_classification_report.csv"
    prediction_path = report_dir / "best_hog_validation_predictions.csv"
    summary_path = report_dir / "baseline_validation_summary.json"
    confusion_path = figures_dir / "hog_validation_confusion_matrix.png"
    comparison.to_csv(comparison_path, index=False)
    comparison[
        [
            "run_order",
            "run_name",
            "model_type",
            "mlflow_run_id",
            "mlflow_artifact_uri",
            "validation_macro_f1",
            "validation_accuracy",
        ]
    ].to_csv(mlflow_runs_path, index=False)
    majority_report.to_csv(majority_report_path, index=False)
    per_run_reports[best_name].to_csv(best_report_path, index=False)
    prediction_frame.to_csv(prediction_path, index=False)
    _plot_confusion_matrix(
        per_run_matrices[best_name],
        encoder.classes_.tolist(),
        confusion_path,
    )

    manifest_hash_after = sha256_file(manifest_path)
    if manifest_hash_after != manifest_hash_before:
        raise AssertionError("Dataset manifest changed during baseline training")
    split_hash_after = (
        sha256_file(frozen_path) if frozen_path is not None else None
    )
    if (
        split_check["frozen_split_sha256"] is not None
        and split_hash_after != split_check["frozen_split_sha256"]
    ):
        raise AssertionError("Frozen family split changed during baseline training")

    best_config = config["experiments"][int(best_row["run_order"]) - 1]
    metadata = {
        "model_type": "hog_multinomial_logistic_regression",
        "selection_metric": "validation_macro_f1",
        "classes": encoder.classes_.tolist(),
        "hog": {
            "image_size": best_config["image_size"],
            "orientations": int(config.get("orientations", 9)),
            "pixels_per_cell": best_config["pixels_per_cell"],
            "cells_per_block": config.get("cells_per_block", [2, 2]),
            "grayscale": True,
            "block_norm": "L2-Hys",
        },
        "logistic_regression": {
            "C": float(best_config["C"]),
            "solver": str(logistic_config.get("solver", "lbfgs")),
            "max_iter": int(logistic_config.get("max_iter", 500)),
        },
        "best_validation_run": {
            "name": best_name,
            "validation_macro_f1": float(best_row["validation_macro_f1"]),
            "validation_accuracy": float(best_row["validation_accuracy"]),
        },
        "training_data": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "fit_splits": ["train"],
            "selection_split": "validation",
            "test_images_loaded": 0,
            "test_rows_evaluated": 0,
            "test_metrics_recorded": False,
        },
        "saved_model_size_bytes": saved_model_size,
        "seed": seed,
    }
    save_json(metadata, metadata_path)

    summary = {
        "status": "passed",
        "purpose": (
            "Validation-only majority and HOG baseline comparison; "
            "not final test performance."
        ),
        "experiment_name": experiment_name,
        "seed": seed,
        "train_rows_fitted": len(train),
        "validation_rows_compared": len(validation),
        "test_images_loaded": 0,
        "test_rows_evaluated": 0,
        "test_metrics_recorded": False,
        "family_overlap_count": split_check["family_overlap_count"],
        "family_assignments_match_frozen_split": split_check[
            "family_assignments_match_frozen_split"
        ],
        "frozen_split_sha256_before": split_check["frozen_split_sha256"],
        "frozen_split_sha256_after": split_hash_after,
        "manifest_sha256_before": manifest_hash_before,
        "manifest_sha256_after": manifest_hash_after,
        "preprocessing_fit_rows": int(
            best_model.named_steps["hog"].fit_sample_count_
        ),
        "preprocessing_fit_split": "train",
        "majority_baseline": {
            "predicted_training_class": str(
                encoder.inverse_transform(
                    [int(majority_model.class_prior_.argmax())]
                )[0]
            ),
            "validation_accuracy": float(
                comparison.iloc[0]["validation_accuracy"]
            ),
            "validation_macro_f1": float(
                comparison.iloc[0]["validation_macro_f1"]
            ),
            "role": "sanity-check baseline only",
        },
        "hog_experiments_completed": len(hog_comparison),
        "best_hog_run": {
            "name": best_name,
            "validation_accuracy": float(best_row["validation_accuracy"]),
            "validation_macro_f1": float(
                best_row["validation_macro_f1"]
            ),
            "inference_ms_per_image": float(
                best_row["inference_ms_per_image"]
            ),
            "saved_model_size_bytes": saved_model_size,
        },
        "saved_model_reload_check": {
            "passed": True,
            "prediction": str(reload_prediction),
            "probability_count": len(reload_probabilities),
            "probability_sum": float(reload_probabilities.sum()),
            "validation_example_path": validation.iloc[0]["image_path"],
        },
        "mlflow": {
            "tracking_database": str(
                (tracking_dir.resolve() / "mlflow.db")
            ),
            "runs_recorded": int(len(comparison)) if enable_mlflow else 0,
            "run_ids_exported": bool(
                enable_mlflow
                and comparison["mlflow_run_id"].astype(bool).all()
            ),
            "ui_command": (
                f"mlflow ui --backend-store-uri "
                f"sqlite:///{(tracking_dir.resolve() / 'mlflow.db').as_posix()}"
            ),
        },
        "outputs": {
            "pipeline": str(pipeline_path),
            "metadata": str(metadata_path),
            "comparison": str(comparison_path),
            "classification_report": str(best_report_path),
            "validation_predictions": str(prediction_path),
            "confusion_matrix": str(confusion_path),
        },
    }
    save_json(summary, summary_path)
    return {
        "artifact": str(pipeline_path),
        "best_validation": summary["best_hog_run"],
        "runs": comparison.to_dict("records"),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train FontSense validation-only majority and HOG + "
            "multinomial Logistic Regression baselines."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(
            project_root() / "reports" / "dataset" / "full_manifest.csv"
        ),
    )
    parser.add_argument(
        "--family-split",
        default=None,
        help=(
            "Frozen family split CSV. The final full manifest finds its "
            "standard frozen split automatically."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(project_root() / "config" / "hog_experiments.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(project_root() / "artifacts" / "baseline"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(project_root() / "reports" / "baseline"),
    )
    parser.add_argument(
        "--tracking-dir",
        default=str(project_root() / "mlruns"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_hog(
        args.manifest,
        args.config,
        args.artifact_dir,
        args.report_dir,
        args.seed,
        family_split_path=args.family_split,
        tracking_dir=args.tracking_dir,
    )
    print(json.dumps(result["best_validation"], indent=2))
    print(f"Saved model: {result['artifact']}")
    print(result["summary"]["mlflow"]["ui_command"])


if __name__ == "__main__":
    main()
