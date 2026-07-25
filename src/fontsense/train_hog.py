from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .features import extract_hog
from .mlflow_utils import optional_mlflow_run
from .split import assert_no_family_leakage
from .utils import load_json, project_root, save_json, set_seed


def matrix_from_frame(frame: pd.DataFrame, hog_config: dict) -> np.ndarray:
    features = [
        extract_hog(
            path,
            size=tuple(hog_config.get("image_size", [112, 48])),
            orientations=int(hog_config["orientations"]),
            pixels_per_cell=tuple(hog_config["pixels_per_cell"]),
            cells_per_block=tuple(hog_config["cells_per_block"]),
        )
        for path in tqdm(frame["image_path"], desc="Extracting HOG")
    ]
    return np.stack(features)


def evaluate_quick(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = model.predict(x)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro"))}


def train_hog(
    manifest_path: str | Path,
    experiment_config_path: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    manifest = pd.read_csv(manifest_path)
    assert_no_family_leakage(manifest)
    config = load_json(experiment_config_path)
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)

    encoder = LabelEncoder().fit(manifest["category"])
    x_train = matrix_from_frame(train, config)
    x_val = matrix_from_frame(val, config)
    y_train = encoder.transform(train["category"])
    y_val = encoder.transform(val["category"])

    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, artifact_dir / "label_encoder.joblib")

    rows: list[dict] = []
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    dummy_metrics = evaluate_quick(dummy, x_val, y_val)
    rows.append({"name": "majority_dummy", "C": None, "class_weight": None, **dummy_metrics})

    best_model = None
    best_row = None
    for exp in config["experiments"]:
        pipeline = Pipeline([
            ("model", OneVsRestClassifier(LogisticRegression(
                C=float(exp["C"]),
                class_weight=exp.get("class_weight"),
                max_iter=300,
                solver="liblinear",
                tol=1e-4,
                random_state=seed,
            ))),
        ])
        started = time.perf_counter()
        with optional_mlflow_run("FontSense-HOG", exp["name"]) as mlflow:
            pipeline.fit(x_train, y_train)
            metrics = evaluate_quick(pipeline, x_val, y_val)
            duration = time.perf_counter() - started
            result = {**exp, **metrics, "training_seconds": duration}
            rows.append(result)
            if mlflow is not None:
                mlflow.log_params({"C": exp["C"], "class_weight": str(exp.get("class_weight")), "seed": seed})
                mlflow.log_metrics(metrics | {"training_seconds": duration})
            if best_row is None or metrics["macro_f1"] > best_row["macro_f1"]:
                best_row = result
                best_model = pipeline

    if best_model is None or best_row is None:
        raise RuntimeError("No HOG model was trained")

    # Refit the selected configuration on train + validation. Test remains unseen.
    combined = pd.concat([train, val], ignore_index=True)
    x_combined = np.concatenate([x_train, x_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    best_model.fit(x_combined, y_combined)
    artifact_path = artifact_dir / "hog_pipeline.joblib"
    joblib.dump(best_model, artifact_path)
    save_json({
        "model_type": "hog_logistic_regression",
        "classes": encoder.classes_.tolist(),
        "hog": {
            "image_size": config.get("image_size", [112, 48]),
            "orientations": config["orientations"],
            "pixels_per_cell": config["pixels_per_cell"],
            "cells_per_block": config["cells_per_block"],
        },
        "selected_validation_run": best_row,
        "test_rows_reserved": len(test),
        "seed": seed,
    }, artifact_dir / "hog_metadata.json")
    runs = pd.DataFrame(rows)
    runs.to_csv(report_dir / "hog_validation_runs.csv", index=False)
    return {"artifact": str(artifact_path), "best_validation": best_row, "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FontSense HOG + Logistic Regression experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=str(project_root() / "config/hog_experiments.json"))
    parser.add_argument("--artifact-dir", default=str(project_root() / "artifacts"))
    parser.add_argument("--report-dir", default=str(project_root() / "reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_hog(args.manifest, args.config, args.artifact_dir, args.report_dir, args.seed)
    print(json.dumps(result["best_validation"], indent=2))
    print(f"Saved model: {result['artifact']}")


if __name__ == "__main__":
    main()
