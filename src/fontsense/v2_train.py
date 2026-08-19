from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn

from .cnn_model import FontSenseCNN
from .mlflow_utils import optional_mlflow_run
from .train_cnn import (
    EXPECTED_CATEGORIES,
    _checkpoint_payload,
    _classification_frame,
    _make_loaders,
    _plot_confusion_matrix,
    _plot_learning_curves,
    _resolve_image_paths,
    _validate_frozen_split,
    build_image_transform,
    load_cnn_checkpoint,
    predict_loader,
    read_train_validation_manifest,
    run_epoch,
)
from .utils import load_json, project_root, save_json, set_seed
from .v2_data import sha256_file


def _candidate_slug(candidate_id: str) -> str:
    return f"candidate_{candidate_id.casefold()}"


def _save_progress_checkpoint(
    checkpoint_path: Path,
    state_dict: dict,
    classes: list[str],
    candidate: dict,
    image_size: tuple[int, int],
    augmentation: dict,
    best_epoch: int,
    validation_macro_f1: float,
    validation_accuracy: float,
    seed: int,
    train_rows: int,
    validation_rows: int,
) -> None:
    """Atomically persist a new best checkpoint before the next epoch starts."""
    payload = _checkpoint_payload(
        state_dict,
        classes,
        candidate,
        image_size,
        augmentation,
        best_epoch,
        {
            "validation_macro_f1": validation_macro_f1,
            "validation_accuracy": validation_accuracy,
        },
        seed,
        train_rows,
        validation_rows,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def _update_stage(stage: str, artifact: Path, checkpoint_hash: str | None = None) -> None:
    state_path = project_root() / "reports/v2/v2_pipeline_state.json"
    state = load_json(state_path)
    payload = {"status": "completed", "artifact": artifact.relative_to(project_root()).as_posix()}
    if checkpoint_hash is not None:
        payload["sha256"] = checkpoint_hash
    state["stages"][stage] = payload
    save_json(state, state_path)


def _load_training_frames(
    manifest_path: Path,
    split_path: Path,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, dict]:
    model_manifest = read_train_validation_manifest(manifest_path)
    split_check = _validate_frozen_split(
        model_manifest,
        split_path,
        str(config["expected_split_sha256"]),
        int(config["expected_family_count"]),
    )
    train_portable = model_manifest.loc[model_manifest["split"] == "train"].reset_index(drop=True)
    validation_portable = model_manifest.loc[
        model_manifest["split"] == "validation"
    ].reset_index(drop=True)
    encoder = LabelEncoder().fit(train_portable["category"])
    if encoder.classes_.tolist() != list(EXPECTED_CATEGORIES):
        raise AssertionError(f"Unexpected V2 class order: {encoder.classes_.tolist()}")
    train_frame = _resolve_image_paths(train_portable, project_root())
    validation_frame = _resolve_image_paths(validation_portable, project_root())
    return train_frame, validation_frame, validation_portable, encoder, split_check


def train_v2_candidate(
    candidate_id: str,
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    config_path: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    tracking_dir: str | Path,
    device_name: str | None = None,
) -> dict:
    root = project_root()
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    config_path = Path(config_path)
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    tracking_dir = Path(tracking_dir)
    config = load_json(config_path)
    candidates = {str(item["id"]).upper(): item for item in config["candidates"]}
    candidate_id = candidate_id.upper()
    if candidate_id not in candidates:
        raise ValueError(f"Unknown V2 candidate {candidate_id!r}")
    candidate = candidates[candidate_id]
    slug = _candidate_slug(candidate_id)
    candidate_report_dir = report_dir / slug
    candidate_artifact_dir = artifact_dir / slug
    summary_path = candidate_report_dir / "summary.json"
    checkpoint_path = candidate_artifact_dir / "cnn_model.pt"

    expected_hashes = {
        "manifest_sha256": str(config["expected_manifest_sha256"]),
        "split_sha256": str(config["expected_split_sha256"]),
        "config_sha256": sha256_file(config_path),
    }
    if summary_path.is_file() and checkpoint_path.is_file():
        existing = load_json(summary_path)
        if (
            existing.get("status") == "passed"
            and all(existing.get(key) == value for key, value in expected_hashes.items())
            and existing.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        ):
            return {"status": "reused", **existing}

    if sha256_file(manifest_path) != expected_hashes["manifest_sha256"]:
        raise AssertionError("V2 manifest hash changed before CNN training")
    if sha256_file(split_path) != expected_hashes["split_sha256"]:
        raise AssertionError("V2 family split hash changed before CNN training")
    train_frame, validation_frame, validation_portable, encoder, split_check = (
        _load_training_frames(manifest_path, split_path, config)
    )
    if len(train_frame) != 14_000 or len(validation_frame) != 3_000:
        raise AssertionError("V2 training must use 14,000 train and 3,000 validation images")

    seed = 42
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    image_size = tuple(int(value) for value in config["image_size"])
    augmentation = dict(config["augmentation"])
    train_loader, validation_loader, train_dataset, validation_dataset = _make_loaders(
        train_frame,
        validation_frame,
        encoder,
        image_size,
        augmentation,
        int(config["batch_size"]),
        seed,
    )
    if not train_dataset.augmentation_enabled or validation_dataset.augmentation_enabled:
        raise AssertionError("V2 augmentation boundary is incorrect")

    model = FontSenseCNN(
        len(encoder.classes_),
        width=int(candidate["width"]),
        dropout=float(candidate["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    maximum_epochs = int(config["max_epochs"])
    patience = int(config["early_stopping_patience"])
    min_delta = float(config["early_stopping_min_delta"])
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict] = []
    candidate_report_dir.mkdir(parents=True, exist_ok=True)
    candidate_artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with optional_mlflow_run(
        str(config["experiment_name"]),
        str(candidate["name"]),
        tracking_dir=tracking_dir,
    ) as active_mlflow:
        if active_mlflow is not None:
            active_mlflow.log_params(
                {
                    "candidate_id": candidate_id,
                    "model_type": "small_grayscale_cnn",
                    "reason": candidate["reason"],
                    "learning_rate": candidate["learning_rate"],
                    "width": candidate["width"],
                    "dropout": candidate["dropout"],
                    "batch_size": config["batch_size"],
                    "max_epochs": maximum_epochs,
                    "early_stopping_patience": patience,
                    "weight_decay": config["weight_decay"],
                    "seed": seed,
                    "device": str(device),
                    "train_rows": len(train_frame),
                    "validation_rows": len(validation_frame),
                    "test_images_loaded": 0,
                    "augmentation_train_only": True,
                }
            )
        for epoch in range(1, maximum_epochs + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, training=True
            )
            validation_metrics = run_epoch(
                model, validation_loader, criterion, None, device, training=False
            )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
            history.append(row)
            print(
                f"Candidate {candidate_id} epoch {epoch:02d}: "
                f"train_f1={row['train_macro_f1']:.4f} "
                f"val_f1={row['validation_macro_f1']:.4f}"
            )
            if active_mlflow is not None:
                active_mlflow.log_metrics(
                    {key: float(value) for key, value in row.items() if key != "epoch"},
                    step=epoch,
                )
            if validation_metrics["macro_f1"] > best_f1 + min_delta:
                best_f1 = float(validation_metrics["macro_f1"])
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                _save_progress_checkpoint(
                    checkpoint_path,
                    best_state,
                    encoder.classes_.tolist(),
                    candidate,
                    image_size,
                    augmentation,
                    best_epoch,
                    best_f1,
                    float(validation_metrics["accuracy"]),
                    seed,
                    len(train_frame),
                    len(validation_frame),
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break

        training_seconds = time.perf_counter() - started
        if best_state is None:
            raise RuntimeError(f"V2 candidate {candidate_id} produced no checkpoint")
        model.load_state_dict(best_state)
        targets, probabilities, inference_seconds = predict_loader(
            model, validation_loader, device
        )
        predictions = probabilities.argmax(axis=1)
        validation_accuracy = float(accuracy_score(targets, predictions))
        validation_macro_f1 = float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        )
        per_class = _classification_frame(targets, predictions, encoder.classes_.tolist())
        matrix = confusion_matrix(
            targets, predictions, labels=np.arange(len(encoder.classes_))
        )
        payload = _checkpoint_payload(
            best_state,
            encoder.classes_.tolist(),
            candidate,
            image_size,
            augmentation,
            best_epoch,
            {
                "validation_macro_f1": validation_macro_f1,
                "validation_accuracy": validation_accuracy,
            },
            seed,
            len(train_frame),
            len(validation_frame),
        )
        torch.save(payload, checkpoint_path)
        history_path = candidate_report_dir / "history.csv"
        per_class_path = candidate_report_dir / "classification_report.csv"
        predictions_path = candidate_report_dir / "validation_predictions.csv"
        curves_path = candidate_report_dir / "learning_curves.png"
        confusion_path = candidate_report_dir / "confusion_matrix.png"
        pd.DataFrame(history).to_csv(history_path, index=False)
        per_class.to_csv(per_class_path, index=False)
        _plot_learning_curves(pd.DataFrame(history), str(candidate["name"]), curves_path)
        _plot_confusion_matrix(
            matrix,
            encoder.classes_.tolist(),
            str(candidate["name"]),
            confusion_path,
            validation_images=len(validation_frame),
        )
        prediction_frame = validation_portable.copy()
        prediction_frame["predicted_category"] = encoder.inverse_transform(predictions)
        prediction_frame["correct"] = prediction_frame["category"] == prediction_frame["predicted_category"]
        for class_index, class_name in enumerate(encoder.classes_):
            prediction_frame[f"probability_{class_name}"] = probabilities[:, class_index]
        prediction_frame.to_csv(predictions_path, index=False)

        loaded_model, loaded = load_cnn_checkpoint(checkpoint_path)
        transform = build_image_transform(
            image_size, training=False, augmentation=augmentation
        )
        with Image.open(validation_frame.iloc[0]["image_path"]) as opened:
            tensor = transform(ImageOps.exif_transpose(opened).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            smoke_probabilities = torch.softmax(loaded_model(tensor), dim=1)[0].numpy()
        if not np.isclose(smoke_probabilities.sum(), 1.0, atol=1e-6):
            raise AssertionError("Reloaded V2 checkpoint probabilities do not sum to one")
        mlflow_run_id = ""
        if active_mlflow is not None:
            active_mlflow.log_metrics(
                {
                    "validation_macro_f1": validation_macro_f1,
                    "validation_accuracy": validation_accuracy,
                    "best_epoch": best_epoch,
                    "epochs_trained": len(history),
                    "training_seconds": training_seconds,
                    "inference_ms_per_image": inference_seconds / len(validation_frame) * 1000,
                    "model_size_bytes": checkpoint_path.stat().st_size,
                }
            )
            for artifact in (
                checkpoint_path, history_path, per_class_path, predictions_path,
                curves_path, confusion_path,
            ):
                active_mlflow.log_artifact(str(artifact), artifact_path=slug)
            mlflow_run_id = active_mlflow.active_run().info.run_id

    checkpoint_hash = sha256_file(checkpoint_path)
    if sha256_file(manifest_path) != expected_hashes["manifest_sha256"]:
        raise AssertionError("V2 manifest changed during training")
    if sha256_file(split_path) != expected_hashes["split_sha256"]:
        raise AssertionError("V2 split changed during training")
    summary = {
        "status": "passed",
        "candidate_id": candidate_id,
        "name": candidate["name"],
        "reason": candidate["reason"],
        "learning_rate": float(candidate["learning_rate"]),
        "width": int(candidate["width"]),
        "dropout": float(candidate["dropout"]),
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "stopped_early": len(history) < maximum_epochs,
        "validation_macro_f1": validation_macro_f1,
        "validation_accuracy": validation_accuracy,
        "training_seconds": training_seconds,
        "inference_ms_per_image": inference_seconds / len(validation_frame) * 1000,
        "model_size_bytes": checkpoint_path.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "train_rows_fitted": len(train_frame),
        "validation_rows_compared": len(validation_frame),
        "test_images_loaded": 0,
        "test_rows_evaluated": 0,
        "test_metrics_recorded": False,
        "augmentation_training_enabled": train_dataset.augmentation_enabled,
        "augmentation_validation_enabled": validation_dataset.augmentation_enabled,
        "family_overlap_count": split_check["family_overlap_count"],
        "manifest_sha256": expected_hashes["manifest_sha256"],
        "split_sha256": expected_hashes["split_sha256"],
        "config_sha256": expected_hashes["config_sha256"],
        "checkpoint": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_probability_sum": float(smoke_probabilities.sum()),
        "class_order": encoder.classes_.tolist(),
        "mlflow_run_id": mlflow_run_id,
        "validation_predictions": predictions_path.relative_to(root).as_posix(),
    }
    save_json(summary, summary_path)
    _update_stage(f"{7 + ord(candidate_id) - ord('A') + 1}_candidate_{candidate_id}", summary_path, checkpoint_hash)
    return summary


def select_v2_candidate(
    *,
    artifact_dir: str | Path,
    report_dir: str | Path,
) -> dict:
    root = project_root()
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    summaries = []
    for candidate_id in ("A", "B", "C"):
        path = report_dir / _candidate_slug(candidate_id) / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"V2 candidate {candidate_id} summary is missing")
        summaries.append(load_json(path))
    comparison = pd.DataFrame(summaries).sort_values("candidate_id")
    winner = comparison.sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    ).iloc[0]
    source_checkpoint = root / str(winner["checkpoint"])
    selected_dir = artifact_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = selected_dir / "cnn_model_v2.pt"
    shutil.copy2(source_checkpoint, selected_checkpoint)
    source_predictions = root / str(winner["validation_predictions"])
    selected_predictions = report_dir / "selected_validation_predictions.csv"
    shutil.copy2(source_predictions, selected_predictions)
    comparison_path = report_dir / "candidate_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    result = {
        "status": "selected_using_validation_only",
        "selection_metric": "validation_macro_f1",
        "selected_candidate": str(winner["candidate_id"]),
        "selected_name": str(winner["name"]),
        "validation_macro_f1": float(winner["validation_macro_f1"]),
        "validation_accuracy": float(winner["validation_accuracy"]),
        "checkpoint": selected_checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(selected_checkpoint),
        "validation_predictions": selected_predictions.relative_to(root).as_posix(),
        "test_images_loaded": 0,
        "test_results_used": False,
    }
    selected_path = report_dir / "selected_model.json"
    save_json(result, selected_path)
    _update_stage("11_model_selection", selected_path, result["checkpoint_sha256"])
    return result


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train resumable FontSense V2 CNN candidates")
    parser.add_argument("command", choices=["A", "B", "C", "all", "select"])
    parser.add_argument("--manifest", default=str(root / "reports/v2/data/full_manifest.csv"))
    parser.add_argument("--split", default=str(root / "data/v2/frozen_family_split.csv"))
    parser.add_argument("--config", default=str(root / "config/v2/cnn_experiments.json"))
    parser.add_argument("--artifact-dir", default=str(root / "artifacts/v2/cnn"))
    parser.add_argument("--report-dir", default=str(root / "reports/v2/cnn"))
    parser.add_argument("--tracking-dir", default=str(root / "reports/v2/mlruns"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.command == "select":
        result = select_v2_candidate(artifact_dir=args.artifact_dir, report_dir=args.report_dir)
        print(json.dumps(result, indent=2))
        return
    candidate_ids = ("A", "B", "C") if args.command == "all" else (args.command,)
    results = []
    for candidate_id in candidate_ids:
        results.append(
            train_v2_candidate(
                candidate_id,
                manifest_path=args.manifest,
                split_path=args.split,
                config_path=args.config,
                artifact_dir=args.artifact_dir,
                report_dir=args.report_dir,
                tracking_dir=args.tracking_dir,
                device_name=args.device,
            )
        )
    if args.command == "all":
        results.append(
            select_v2_candidate(artifact_dir=args.artifact_dir, report_dir=args.report_dir)
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
