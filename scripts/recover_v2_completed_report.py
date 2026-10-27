"""Recover a missing V2 summary after a reporting-path error, using saved evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch

from fontsense.train_cnn import load_cnn_checkpoint
from fontsense.utils import project_root, save_json
from fontsense.v2_data import sha256_file
from fontsense.v2_train import _update_stage


def recover(run_id: str) -> dict:
    root = project_root()
    config_path = root / "config/v2/cnn_refinement.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate = config["candidates"][0]
    report_dir = root / "reports/v2/refinement/candidate_d"
    checkpoint_path = root / "artifacts/v2/refinement/candidate_d/cnn_model.pt"
    summary_path = report_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError("The completed summary already exists; it will not be overwritten.")
    for kind, path in [("manifest", "reports/v2/data/full_manifest.csv"), ("split", "data/v2/frozen_family_split.csv")]:
        assert sha256_file(root / path) == config[f"expected_{kind}_sha256"]
    assert sha256_file(root / candidate["initial_checkpoint"]) == candidate["initial_checkpoint_sha256"]
    mlflow.set_tracking_uri("sqlite:///" + (root / "reports/v2/mlruns/mlflow.db").as_posix())
    run = MlflowClient().get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["candidate_id"] == "D"
    assert run.data.params["initial_checkpoint_sha256"] == candidate["initial_checkpoint_sha256"]
    model, checkpoint = load_cnn_checkpoint(checkpoint_path)
    assert checkpoint["training_data"]["fit_splits"] == ["train"]
    assert checkpoint["training_data"]["test_images_loaded"] == 0
    history = pd.read_csv(report_dir / "history.csv")
    predictions_path = report_dir / "validation_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == 3000 and set(predictions["split"]) == {"validation"}
    probability_columns = [f"probability_{name}" for name in checkpoint["classes"]]
    probabilities = predictions[probability_columns].to_numpy()
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1)
    accuracy = float(accuracy_score(predictions["category"], predictions["predicted_category"]))
    macro_f1 = float(f1_score(predictions["category"], predictions["predicted_category"], average="macro", zero_division=0))
    selected = checkpoint["selected_validation_run"]
    assert np.isclose(accuracy, selected["validation_accuracy"])
    assert np.isclose(macro_f1, selected["validation_macro_f1"])
    best_epoch = int(selected["best_epoch"])
    assert np.isclose(history.loc[history["epoch"] == best_epoch, "validation_macro_f1"].iloc[0], macro_f1)
    assert int(run.data.metrics["epochs_trained"]) == len(history)
    with torch.inference_mode():
        smoke_sum = float(torch.softmax(model(torch.zeros(1, 1, 48, 112)), dim=1).sum())
    assert np.isclose(smoke_sum, 1)
    summary = {
        "status": "passed", "candidate_id": "D", "name": candidate["name"], "reason": candidate["reason"],
        "learning_rate": candidate["learning_rate"], "width": candidate["width"], "dropout": candidate["dropout"],
        "initialization": {"checkpoint": candidate["initial_checkpoint"], "sha256": candidate["initial_checkpoint_sha256"]},
        "best_epoch": best_epoch, "epochs_trained": len(history), "stopped_early": len(history) < config["max_epochs"],
        "stop_reason": "early_stopping", "validation_macro_f1": macro_f1, "validation_accuracy": accuracy,
        "training_seconds": run.data.metrics["training_seconds"], "inference_ms_per_image": run.data.metrics["inference_ms_per_image"],
        "model_size_bytes": checkpoint_path.stat().st_size, "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": run.data.params["device"], "train_rows_fitted": 14000, "validation_rows_compared": len(predictions),
        "test_images_loaded": 0, "test_rows_evaluated": 0, "test_metrics_recorded": False,
        "augmentation_training_enabled": True, "augmentation_validation_enabled": False, "family_overlap_count": 0,
        "manifest_sha256": config["expected_manifest_sha256"], "split_sha256": config["expected_split_sha256"],
        "config_sha256": sha256_file(config_path), "checkpoint": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_reload_probability_sum": smoke_sum,
        "class_order": checkpoint["classes"], "mlflow_run_id": run_id, "validation_predictions": predictions_path.relative_to(root).as_posix(),
        "report_recovery": "Training and validation completed. The final summary failed while converting a relative output path. This summary was recovered from checkpoint metadata, validation predictions, epoch history, and the FINISHED MLflow run. No training or dataset-image inference was repeated.",
    }
    save_json(summary, summary_path)
    _update_stage("10b_candidate_D", summary_path, summary["checkpoint_sha256"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.run_id), indent=2))
