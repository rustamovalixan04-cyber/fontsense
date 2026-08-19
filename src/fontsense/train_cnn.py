from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
import time
from contextlib import nullcontext
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
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .cnn_model import FontSenseCNN
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
ALLOWED_MODEL_SPLITS = {"train", "validation"}
REQUIRED_MANIFEST_COLUMNS = {"image_path", "family", "category", "split"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: str | Path, root: Path) -> str:
    """Prefer a portable repository-relative path for saved reports."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_train_validation_manifest(path: str | Path) -> pd.DataFrame:
    """Retain only train/validation rows; test rows never enter a DataFrame."""
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in reader:
            split = str(row["split"])
            if split not in ALLOWED_MODEL_SPLITS:
                continue
            rows.append(
                {
                    "image_path": str(row["image_path"]),
                    "family": str(row["family"]),
                    "category": str(row["category"]),
                    "split": split,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Manifest contains zero train/validation rows")
    if set(frame["split"]) != ALLOWED_MODEL_SPLITS:
        raise ValueError("Both train and validation rows are required")
    return frame


def build_image_transform(
    image_size: tuple[int, int],
    *,
    training: bool,
    augmentation: dict,
) -> transforms.Compose:
    width, height = image_size
    operations: list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (height, width),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
    ]
    if training and bool(augmentation.get("enabled", False)):
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=float(augmentation["rotation_degrees"]),
                    translate=tuple(
                        float(value)
                        for value in augmentation["translate_fraction"]
                    ),
                    scale=tuple(
                        float(value)
                        for value in augmentation["scale_range"]
                    ),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=127,
                ),
                transforms.RandomAdjustSharpness(
                    sharpness_factor=float(
                        augmentation["sharpness_factor"]
                    ),
                    p=float(augmentation["sharpness_probability"]),
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(operations)


class ManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        encoder: LabelEncoder,
        image_size: tuple[int, int],
        *,
        training: bool,
        augmentation: dict,
    ):
        unexpected = set(frame["split"].unique()) - ALLOWED_MODEL_SPLITS
        if unexpected:
            raise ValueError(
                f"Dataset received forbidden splits: {sorted(unexpected)}"
            )
        expected_split = "train" if training else "validation"
        observed = set(frame["split"].unique())
        if observed != {expected_split}:
            raise ValueError(
                f"{expected_split} dataset received splits: {sorted(observed)}"
            )
        self.frame = frame[["image_path", "category"]].reset_index(drop=True)
        self.encoder = encoder
        self.training = training
        self.augmentation_enabled = bool(
            training and augmentation.get("enabled", False)
        )
        self.transform = build_image_transform(
            image_size,
            training=training,
            augmentation=augmentation,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            prepared = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(prepared)
        label = int(self.encoder.transform([row["category"]])[0])
        return tensor, label


def _resolve_image_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    result = frame.copy()
    resolved: list[str] = []
    missing: list[str] = []
    for raw_path in result["image_path"].astype(str):
        path = Path(raw_path)
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file():
            missing.append(raw_path)
        resolved.append(str(absolute.resolve()))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} train/validation images are missing; "
            f"first missing path: {missing[0]}"
        )
    result["image_path"] = resolved
    return result


def _validate_frozen_split(
    model_manifest: pd.DataFrame,
    family_split_path: Path,
    expected_hash: str | None,
    expected_family_count: int | None,
) -> dict:
    if not family_split_path.is_file():
        raise FileNotFoundError(
            f"Frozen family split not found: {family_split_path}"
        )
    split_hash = sha256_file(family_split_path)
    if expected_hash is not None and split_hash != expected_hash:
        raise AssertionError(
            "Frozen family split hash changed: "
            f"expected {expected_hash}, found {split_hash}"
        )
    frozen = pd.read_csv(
        family_split_path,
        usecols=["family", "category", "split"],
    )
    assert_no_family_leakage(frozen)
    if (
        expected_family_count is not None
        and frozen["family"].nunique() != expected_family_count
    ):
        raise AssertionError(
            f"Expected {expected_family_count} frozen families; "
            f"found {frozen['family'].nunique()}"
        )
    allowed_frozen = (
        frozen.loc[frozen["split"].isin(ALLOWED_MODEL_SPLITS)]
        .drop_duplicates()
        .sort_values(["family", "category", "split"])
        .reset_index(drop=True)
    )
    observed = (
        model_manifest[["family", "category", "split"]]
        .drop_duplicates()
        .sort_values(["family", "category", "split"])
        .reset_index(drop=True)
    )
    if not observed.equals(allowed_frozen):
        raise AssertionError(
            "Train/validation manifest assignments do not match the frozen split"
        )
    return {
        "family_overlap_count": 0,
        "family_assignments_match_frozen_split": True,
        "frozen_split_sha256": split_hash,
    }


def _make_loaders(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    encoder: LabelEncoder,
    image_size: tuple[int, int],
    augmentation: dict,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, ManifestDataset, ManifestDataset]:
    train_dataset = ManifestDataset(
        train_frame,
        encoder,
        image_size,
        training=True,
        augmentation=augmentation,
    )
    validation_dataset = ManifestDataset(
        validation_frame,
        encoder,
        image_size,
        training=False,
        augmentation=augmentation,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    *,
    training: bool,
) -> dict[str, float]:
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_rows = int(labels.size(0))
        total_loss += float(loss.item()) * batch_rows
        total_rows += batch_rows
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if total_rows == 0:
        raise ValueError("DataLoader produced zero rows")
    return {
        "loss": total_loss / total_rows,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    targets: list[int] = []
    probabilities: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities.append(
                torch.softmax(logits, dim=1).detach().cpu().numpy()
            )
            targets.extend(labels.tolist())
    elapsed = time.perf_counter() - started
    return (
        np.asarray(targets, dtype=np.int64),
        np.concatenate(probabilities, axis=0),
        elapsed,
    )


def _classification_frame(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(
        [
            {
                "class": class_name,
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1": float(report[class_name]["f1-score"]),
                "support": int(report[class_name]["support"]),
            }
            for class_name in class_names
        ]
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _plot_learning_curves(
    history: pd.DataFrame,
    run_name: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        label="Train",
        color="#174a7e",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[0].plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation",
        color="#e07a5f",
        linewidth=2,
        marker="s",
        markersize=3,
    )
    axes[0].set(
        title="Cross-entropy loss",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axes[1].plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train accuracy",
        color="#174a7e",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[1].plot(
        history["epoch"],
        history["validation_accuracy"],
        label="Validation accuracy",
        color="#e07a5f",
        linewidth=2,
        marker="s",
        markersize=3,
    )
    axes[1].plot(
        history["epoch"],
        history["validation_macro_f1"],
        label="Validation macro F1",
        color="#5f6b3c",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set(
        title="Accuracy and selection metric",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, 1),
    )
    for axis in axes:
        axis.grid(axis="y", color="#d8dee8", linewidth=0.7)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{run_name}: training and validation curves", fontsize=15)
    figure.text(
        0.5,
        0.925,
        "Train uses mild augmentation; validation uses deterministic preprocessing",
        ha="center",
        color="#4b5563",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    run_name: str,
    output_path: Path,
    validation_images: int = 600,
) -> None:
    labels = [name.replace("_", " ").title() for name in class_names]
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Validation images",
    )
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="True category",
    )
    figure.suptitle(f"{run_name}: validation confusion matrix", y=0.985)
    figure.text(
        0.5,
        0.947,
        f"{validation_images:,} validation images from unseen families; no test images evaluated",
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
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def load_cnn_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[FontSenseCNN, dict]:
    target_device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=False,
    )
    architecture = checkpoint["architecture"]
    model = FontSenseCNN(
        num_classes=len(checkpoint["classes"]),
        width=int(architecture["width"]),
        dropout=float(architecture["dropout"]),
    ).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _checkpoint_payload(
    state_dict: dict,
    classes: list[str],
    experiment: dict,
    image_size: tuple[int, int],
    augmentation: dict,
    best_epoch: int,
    validation_metrics: dict,
    seed: int,
    train_rows: int,
    validation_rows: int,
) -> dict:
    return {
        "state_dict": state_dict,
        "classes": classes,
        "architecture": {
            "width": int(experiment["width"]),
            "dropout": float(experiment["dropout"]),
        },
        "preprocessing": {
            "image_size": list(image_size),
            "grayscale": True,
            "normalize_mean": [0.5],
            "normalize_std": [0.5],
        },
        "training_augmentation": augmentation,
        "selected_validation_run": {
            "name": str(experiment["name"]),
            "best_epoch": best_epoch,
            "validation_macro_f1": float(
                validation_metrics["validation_macro_f1"]
            ),
            "validation_accuracy": float(
                validation_metrics["validation_accuracy"]
            ),
        },
        "training_data": {
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "fit_splits": ["train"],
            "selection_split": "validation",
            "test_images_loaded": 0,
            "test_rows_evaluated": 0,
            "test_metrics_recorded": False,
        },
        "seed": seed,
    }


def train_cnn(
    manifest_path: str | Path,
    experiment_config_path: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    *,
    family_split_path: str | Path | None,
    baseline_summary_path: str | Path,
    tracking_dir: str | Path = "mlruns",
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int = 42,
    enable_mlflow: bool = True,
    device_name: str | None = None,
) -> dict:
    """Train validation-only CNN experiments without loading a test image."""
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = project_root()
    manifest_path = Path(manifest_path)
    final_manifest_path = (
        root / "reports" / "dataset" / "full_manifest.csv"
    ).resolve()
    is_final_manifest = manifest_path.resolve() == final_manifest_path
    family_split_was_explicit = family_split_path is not None
    if family_split_path is None:
        family_split_path = (
            root
            / "data"
            / "interim"
            / "google_fonts_final_family_split.csv"
            if is_final_manifest
            else manifest_path.parent / "families.csv"
        )
    family_split_path = Path(family_split_path)
    baseline_summary_path = Path(baseline_summary_path)
    config = load_json(experiment_config_path)
    manifest_hash_before = sha256_file(manifest_path)
    model_manifest = read_train_validation_manifest(manifest_path)
    split_check = _validate_frozen_split(
        model_manifest,
        family_split_path,
        str(config["expected_split_sha256"])
        if is_final_manifest or family_split_was_explicit
        else None,
        int(config.get("expected_family_count", 90))
        if is_final_manifest or family_split_was_explicit
        else None,
    )

    train_frame = model_manifest.loc[
        model_manifest["split"] == "train"
    ].reset_index(drop=True)
    validation_frame = model_manifest.loc[
        model_manifest["split"] == "validation"
    ].reset_index(drop=True)
    encoder = LabelEncoder().fit(train_frame["category"])
    if encoder.classes_.tolist() != list(EXPECTED_CATEGORIES):
        raise ValueError(
            "Training split must contain all five categories; "
            f"found {encoder.classes_.tolist()}"
        )
    unknown_validation = sorted(
        set(validation_frame["category"]) - set(encoder.classes_)
    )
    if unknown_validation:
        raise ValueError(
            f"Validation contains categories absent from training: "
            f"{unknown_validation}"
        )
    train_frame = _resolve_image_paths(train_frame, root)
    validation_frame = _resolve_image_paths(validation_frame, root)

    image_size = tuple(int(value) for value in config["image_size"])
    augmentation = dict(config["augmentation"])
    maximum_epochs = int(
        epochs if epochs is not None else config["max_epochs"]
    )
    selected_batch_size = int(
        batch_size if batch_size is not None else config["batch_size"]
    )
    patience = int(config["early_stopping_patience"])
    min_delta = float(config["early_stopping_min_delta"])
    weight_decay = float(config["weight_decay"])
    if maximum_epochs < 1 or selected_batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")

    if device_name is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    figures_dir = report_dir / "figures"
    run_details_dir = report_dir / "run_details"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    run_details_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir = Path(tracking_dir)
    experiment_name = str(config["experiment_name"])

    all_history: list[dict] = []
    comparison_rows: list[dict] = []
    run_outputs: dict[str, dict] = {}
    best_global_name: str | None = None
    best_global_f1 = -1.0
    best_global_payload: dict | None = None

    for run_order, experiment in enumerate(
        config["experiments"],
        start=1,
    ):
        run_name = str(experiment["name"])
        slug = _slug(run_name)
        set_seed(seed)
        (
            train_loader,
            validation_loader,
            train_dataset,
            validation_dataset,
        ) = _make_loaders(
            train_frame,
            validation_frame,
            encoder,
            image_size,
            augmentation,
            selected_batch_size,
            seed,
        )
        if not train_dataset.augmentation_enabled:
            raise AssertionError("Training augmentation was not enabled")
        if validation_dataset.augmentation_enabled:
            raise AssertionError(
                "Validation augmentation must always be disabled"
            )

        model = FontSenseCNN(
            len(encoder.classes_),
            width=int(experiment["width"]),
            dropout=float(experiment["dropout"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(experiment["learning_rate"]),
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        best_run_f1 = -1.0
        best_run_epoch = 0
        best_run_state: dict | None = None
        stale_epochs = 0
        history_rows: list[dict] = []
        training_started = time.perf_counter()

        context = (
            optional_mlflow_run(
                experiment_name,
                run_name,
                tracking_dir=tracking_dir,
            )
            if enable_mlflow
            else nullcontext(None)
        )
        with context as active_mlflow:
            if enable_mlflow and active_mlflow is None:
                raise RuntimeError(
                    "MLflow is required for CNN tracking but is not installed"
                )
            if active_mlflow is not None:
                active_mlflow.log_params(
                    {
                        "model_type": "small_grayscale_cnn",
                        "reason": str(experiment["reason"]),
                        "learning_rate": float(
                            experiment["learning_rate"]
                        ),
                        "width": int(experiment["width"]),
                        "dropout": float(experiment["dropout"]),
                        "image_width": image_size[0],
                        "image_height": image_size[1],
                        "grayscale": True,
                        "batch_size": selected_batch_size,
                        "max_epochs": maximum_epochs,
                        "early_stopping_patience": patience,
                        "early_stopping_min_delta": min_delta,
                        "weight_decay": weight_decay,
                        "augmentation": json.dumps(
                            augmentation,
                            sort_keys=True,
                        ),
                        "augmentation_train_only": True,
                        "seed": seed,
                        "device": str(device),
                        "train_rows": len(train_frame),
                        "validation_rows": len(validation_frame),
                        "test_images_loaded": 0,
                    }
                )

            for epoch in range(1, maximum_epochs + 1):
                train_metrics = run_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    training=True,
                )
                validation_metrics = run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    optimizer=None,
                    device=device,
                    training=False,
                )
                row = {
                    "run_order": run_order,
                    "run": run_name,
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                }
                all_history.append(row)
                history_rows.append(row)
                if active_mlflow is not None:
                    active_mlflow.log_metrics(
                        {
                            f"epoch_{key}": float(value)
                            for key, value in row.items()
                            if key
                            not in {"run_order", "run", "epoch"}
                        },
                        step=epoch,
                    )

                if (
                    validation_metrics["macro_f1"]
                    > best_run_f1 + min_delta
                ):
                    best_run_f1 = validation_metrics["macro_f1"]
                    best_run_epoch = epoch
                    best_run_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= patience:
                    break

            training_seconds = time.perf_counter() - training_started
            if best_run_state is None:
                raise RuntimeError(f"{run_name} did not produce a checkpoint")
            model.load_state_dict(best_run_state)
            (
                validation_targets,
                validation_probabilities,
                inference_seconds,
            ) = predict_loader(model, validation_loader, device)
            validation_predictions = validation_probabilities.argmax(axis=1)
            validation_accuracy = float(
                accuracy_score(
                    validation_targets,
                    validation_predictions,
                )
            )
            validation_macro_f1 = float(
                f1_score(
                    validation_targets,
                    validation_predictions,
                    average="macro",
                    zero_division=0,
                )
            )
            per_class = _classification_frame(
                validation_targets,
                validation_predictions,
                encoder.classes_.tolist(),
            )
            matrix = confusion_matrix(
                validation_targets,
                validation_predictions,
                labels=np.arange(len(encoder.classes_)),
            )
            history_frame = pd.DataFrame(history_rows)
            history_path = run_details_dir / f"{slug}_history.csv"
            per_class_path = (
                run_details_dir / f"{slug}_classification_report.csv"
            )
            curves_path = figures_dir / f"{slug}_learning_curves.png"
            confusion_path = (
                figures_dir / f"{slug}_confusion_matrix.png"
            )
            history_frame.to_csv(history_path, index=False)
            per_class.to_csv(per_class_path, index=False)
            _plot_learning_curves(
                history_frame,
                run_name,
                curves_path,
            )
            _plot_confusion_matrix(
                matrix,
                encoder.classes_.tolist(),
                run_name,
                confusion_path,
            )

            validation_result = {
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
            }
            payload = _checkpoint_payload(
                best_run_state,
                encoder.classes_.tolist(),
                experiment,
                image_size,
                augmentation,
                best_run_epoch,
                validation_result,
                seed,
                len(train_frame),
                len(validation_frame),
            )
            with tempfile.TemporaryDirectory(
                prefix="fontsense_cnn_"
            ) as temporary:
                checkpoint_path = Path(temporary) / "cnn_model.pt"
                torch.save(payload, checkpoint_path)
                model_size_bytes = checkpoint_path.stat().st_size
                if active_mlflow is not None:
                    active_mlflow.log_artifact(
                        str(checkpoint_path),
                        artifact_path="model",
                    )
            if active_mlflow is not None:
                active_mlflow.log_artifact(
                    str(history_path),
                    artifact_path="training",
                )
                active_mlflow.log_artifact(
                    str(per_class_path),
                    artifact_path="validation",
                )
                active_mlflow.log_artifact(
                    str(curves_path),
                    artifact_path="figures",
                )
                active_mlflow.log_artifact(
                    str(confusion_path),
                    artifact_path="figures",
                )
                active_mlflow.log_metrics(
                    {
                        "validation_macro_f1": validation_macro_f1,
                        "validation_accuracy": validation_accuracy,
                        "best_validation_macro_f1": validation_macro_f1,
                        "best_validation_accuracy": validation_accuracy,
                        "best_epoch": best_run_epoch,
                        "epochs_trained": len(history_frame),
                        "training_seconds": training_seconds,
                        "inference_seconds": inference_seconds,
                        "inference_ms_per_image": (
                            inference_seconds
                            / len(validation_frame)
                            * 1000
                        ),
                        "model_size_bytes": model_size_bytes,
                        "parameter_count": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                        ),
                        **{
                            f"validation_{row['class']}_{metric}": float(
                                row[metric]
                            )
                            for _, row in per_class.iterrows()
                            for metric in ("precision", "recall", "f1")
                        },
                    }
                )
                active_run = active_mlflow.active_run()
                mlflow_run_id = active_run.info.run_id
                mlflow_artifact_uri = (
                    f"{display_path(tracking_dir, root)}/"
                    f"{active_run.info.experiment_id}/"
                    f"{mlflow_run_id}/artifacts"
                )
            else:
                mlflow_run_id = ""
                mlflow_artifact_uri = ""

        comparison_row = {
            "run_order": run_order,
            "run_name": run_name,
            "reason": str(experiment["reason"]),
            "learning_rate": float(experiment["learning_rate"]),
            "width": int(experiment["width"]),
            "dropout": float(experiment["dropout"]),
            "image_width": image_size[0],
            "image_height": image_size[1],
            "batch_size": selected_batch_size,
            "augmentation_train_only": True,
            "epochs_trained": len(history_rows),
            "best_epoch": best_run_epoch,
            "stopped_early": len(history_rows) < maximum_epochs,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_macro_f1,
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "inference_ms_per_image": (
                inference_seconds / len(validation_frame) * 1000
            ),
            "model_size_bytes": model_size_bytes,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "mlflow_run_id": mlflow_run_id,
            "mlflow_artifact_uri": mlflow_artifact_uri,
        }
        comparison_rows.append(comparison_row)
        run_outputs[run_name] = {
            "payload": payload,
            "per_class": per_class,
            "matrix": matrix,
            "probabilities": validation_probabilities,
            "predictions": validation_predictions,
            "history_path": history_path,
            "per_class_path": per_class_path,
            "curves_path": curves_path,
            "confusion_path": confusion_path,
        }
        if validation_macro_f1 > best_global_f1:
            best_global_f1 = validation_macro_f1
            best_global_name = run_name
            best_global_payload = payload

    if best_global_name is None or best_global_payload is None:
        raise RuntimeError("No CNN experiment completed")
    comparison = pd.DataFrame(comparison_rows).sort_values("run_order")
    best_row = comparison.loc[
        comparison["run_name"] == best_global_name
    ].iloc[0]
    best_output = run_outputs[best_global_name]

    model_path = artifact_dir / "cnn_model.pt"
    metadata_path = artifact_dir / "cnn_metadata.json"
    torch.save(best_global_payload, model_path)
    saved_model_size = model_path.stat().st_size

    loaded_model, loaded_checkpoint = load_cnn_checkpoint(model_path)
    inference_transform = build_image_transform(
        image_size,
        training=False,
        augmentation=augmentation,
    )
    with Image.open(validation_frame.iloc[0]["image_path"]) as image:
        tensor = inference_transform(
            ImageOps.exif_transpose(image).convert("RGB")
        ).unsqueeze(0)
    with torch.inference_mode():
        reload_probabilities = torch.softmax(
            loaded_model(tensor),
            dim=1,
        )[0].numpy()
    if reload_probabilities.shape != (len(encoder.classes_),):
        raise AssertionError(
            "Reloaded CNN did not return all five probabilities"
        )
    if not np.isclose(reload_probabilities.sum(), 1.0, atol=1e-6):
        raise AssertionError("Reloaded CNN probabilities do not sum to one")
    reload_prediction = loaded_checkpoint["classes"][
        int(reload_probabilities.argmax())
    ]

    comparison_path = report_dir / "cnn_experiment_comparison.csv"
    history_path = report_dir / "cnn_training_history.csv"
    mlflow_runs_path = report_dir / "mlflow_runs.csv"
    per_class_path = report_dir / "best_cnn_classification_report.csv"
    predictions_path = report_dir / "best_cnn_validation_predictions.csv"
    model_comparison_path = report_dir / "model_comparison.csv"
    summary_path = report_dir / "cnn_validation_summary.json"
    comparison.to_csv(comparison_path, index=False)
    pd.DataFrame(all_history).to_csv(history_path, index=False)
    comparison[
        [
            "run_order",
            "run_name",
            "mlflow_run_id",
            "mlflow_artifact_uri",
            "validation_macro_f1",
            "validation_accuracy",
        ]
    ].to_csv(mlflow_runs_path, index=False)
    best_output["per_class"].to_csv(per_class_path, index=False)

    best_probabilities = best_output["probabilities"]
    prediction_frame = validation_frame[
        ["image_path", "family", "category", "split"]
    ].copy()
    prediction_frame["image_path"] = model_manifest.loc[
        model_manifest["split"] == "validation",
        "image_path",
    ].reset_index(drop=True)
    prediction_frame["predicted_category"] = encoder.inverse_transform(
        best_output["predictions"]
    )
    prediction_frame["correct"] = (
        prediction_frame["category"]
        == prediction_frame["predicted_category"]
    )
    for class_index, class_name in enumerate(encoder.classes_):
        prediction_frame[f"probability_{class_name}"] = best_probabilities[
            :, class_index
        ]
    prediction_frame.to_csv(predictions_path, index=False)

    baseline = load_json(baseline_summary_path)
    baseline_comparison_path = baseline_summary_path.parent / "validation_comparison.csv"
    majority_inference_ms = None
    majority_model_size = None
    if baseline_comparison_path.exists():
        baseline_comparison = pd.read_csv(baseline_comparison_path)
        majority_rows = baseline_comparison.loc[
            baseline_comparison["model_type"] == "majority_class"
        ]
        if len(majority_rows) == 1:
            majority_inference_ms = float(
                majority_rows.iloc[0]["inference_ms_per_image"]
            )
            majority_model_size = int(
                majority_rows.iloc[0]["model_size_bytes"]
            )
    model_comparison = pd.DataFrame(
        [
            {
                "model": "Majority-class baseline",
                "model_family": "majority",
                "validation_macro_f1": baseline["majority_baseline"][
                    "validation_macro_f1"
                ],
                "validation_accuracy": baseline["majority_baseline"][
                    "validation_accuracy"
                ],
                "inference_ms_per_image": majority_inference_ms,
                "model_size_bytes": majority_model_size,
                "selection_basis": "sanity check only",
            },
            {
                "model": "Best HOG + Logistic Regression",
                "model_family": "hog",
                "validation_macro_f1": baseline["best_hog_run"][
                    "validation_macro_f1"
                ],
                "validation_accuracy": baseline["best_hog_run"][
                    "validation_accuracy"
                ],
                "inference_ms_per_image": baseline["best_hog_run"][
                    "inference_ms_per_image"
                ],
                "model_size_bytes": baseline["best_hog_run"][
                    "saved_model_size_bytes"
                ],
                "selection_basis": "best HOG validation macro F1",
            },
            {
                "model": "Best small CNN",
                "model_family": "cnn",
                "validation_macro_f1": float(
                    best_row["validation_macro_f1"]
                ),
                "validation_accuracy": float(
                    best_row["validation_accuracy"]
                ),
                "inference_ms_per_image": float(
                    best_row["inference_ms_per_image"]
                ),
                "model_size_bytes": saved_model_size,
                "selection_basis": "best CNN validation macro F1",
            },
        ]
    )
    model_comparison.to_csv(model_comparison_path, index=False)
    selected_model_row = model_comparison.sort_values(
        ["validation_macro_f1", "validation_accuracy"],
        ascending=False,
    ).iloc[0]

    manifest_hash_after = sha256_file(manifest_path)
    split_hash_after = sha256_file(family_split_path)
    if manifest_hash_after != manifest_hash_before:
        raise AssertionError("Dataset manifest changed during CNN training")
    if split_hash_after != split_check["frozen_split_sha256"]:
        raise AssertionError("Frozen family split changed during CNN training")

    metadata = {
        "model_type": "small_grayscale_cnn",
        "selection_metric": "validation_macro_f1",
        "classes": encoder.classes_.tolist(),
        "architecture": best_global_payload["architecture"],
        "preprocessing": best_global_payload["preprocessing"],
        "training_augmentation": augmentation,
        "best_validation_run": {
            "name": best_global_name,
            "best_epoch": int(best_row["best_epoch"]),
            "validation_macro_f1": float(
                best_row["validation_macro_f1"]
            ),
            "validation_accuracy": float(
                best_row["validation_accuracy"]
            ),
        },
        "training_data": best_global_payload["training_data"],
        "saved_model_size_bytes": saved_model_size,
        "seed": seed,
    }
    save_json(metadata, metadata_path)

    summary = {
        "status": "passed",
        "purpose": (
            "Validation-only small CNN comparison; not final test performance."
        ),
        "experiment_name": experiment_name,
        "seed": seed,
        "device": str(device),
        "train_rows_fitted": len(train_frame),
        "validation_rows_compared": len(validation_frame),
        "test_images_loaded": 0,
        "test_rows_evaluated": 0,
        "test_metrics_recorded": False,
        "family_overlap_count": split_check["family_overlap_count"],
        "family_assignments_match_frozen_split": split_check[
            "family_assignments_match_frozen_split"
        ],
        "frozen_split_sha256_before": split_check[
            "frozen_split_sha256"
        ],
        "frozen_split_sha256_after": split_hash_after,
        "manifest_sha256_before": manifest_hash_before,
        "manifest_sha256_after": manifest_hash_after,
        "augmentation": {
            "training_enabled": True,
            "validation_enabled": False,
            "settings": augmentation,
        },
        "cnn_experiments_completed": len(comparison),
        "best_cnn_run": {
            "name": best_global_name,
            "best_epoch": int(best_row["best_epoch"]),
            "epochs_trained": int(best_row["epochs_trained"]),
            "validation_accuracy": float(
                best_row["validation_accuracy"]
            ),
            "validation_macro_f1": float(
                best_row["validation_macro_f1"]
            ),
            "training_seconds": float(best_row["training_seconds"]),
            "inference_ms_per_image": float(
                best_row["inference_ms_per_image"]
            ),
            "saved_model_size_bytes": saved_model_size,
            "parameter_count": int(best_row["parameter_count"]),
        },
        "saved_model_reload_check": {
            "passed": True,
            "prediction": reload_prediction,
            "probability_count": len(reload_probabilities),
            "probability_sum": float(reload_probabilities.sum()),
            "validation_example_path": prediction_frame.iloc[0][
                "image_path"
            ],
        },
        "model_selected_for_final_test": {
            "model": str(selected_model_row["model"]),
            "model_family": str(selected_model_row["model_family"]),
            "reason": (
                "Highest validation macro F1; the untouched test split "
                "has not been evaluated."
            ),
            "validation_macro_f1": float(
                selected_model_row["validation_macro_f1"]
            ),
        },
        "mlflow": {
            "tracking_database": display_path(
                tracking_dir / "mlflow.db",
                root,
            ),
            "runs_recorded": int(len(comparison)) if enable_mlflow else 0,
            "run_ids_exported": bool(
                enable_mlflow
                and comparison["mlflow_run_id"].astype(bool).all()
            ),
            "ui_command": (
                "mlflow ui --backend-store-uri "
                f"sqlite:///{display_path(tracking_dir / 'mlflow.db', root)}"
            ),
        },
        "outputs": {
            "checkpoint": display_path(model_path, root),
            "metadata": display_path(metadata_path, root),
            "experiment_comparison": display_path(
                comparison_path,
                root,
            ),
            "training_history": display_path(history_path, root),
            "classification_report": display_path(per_class_path, root),
            "validation_predictions": display_path(
                predictions_path,
                root,
            ),
            "model_comparison": display_path(
                model_comparison_path,
                root,
            ),
            "figures": [
                display_path(output["curves_path"], root)
                for output in run_outputs.values()
            ]
            + [
                display_path(output["confusion_path"], root)
                for output in run_outputs.values()
            ],
        },
    }
    save_json(summary, summary_path)
    return {
        "artifact": str(model_path),
        "best_validation": summary["best_cnn_run"],
        "selected_for_final_test": summary[
            "model_selected_for_final_test"
        ],
        "runs": comparison.to_dict("records"),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train FontSense validation-only small CNN experiments."
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
            "Family split CSV. The full final manifest and proof manifests "
            "find their standard split files automatically."
        ),
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(
            project_root()
            / "reports"
            / "baseline"
            / "baseline_validation_summary.json"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(project_root() / "config" / "cnn_experiments.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(project_root() / "artifacts" / "cnn"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(project_root() / "reports" / "cnn"),
    )
    parser.add_argument(
        "--tracking-dir",
        default=str(project_root() / "mlruns"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_cnn(
        args.manifest,
        args.config,
        args.artifact_dir,
        args.report_dir,
        family_split_path=args.family_split,
        baseline_summary_path=args.baseline_summary,
        tracking_dir=args.tracking_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result["best_validation"], indent=2))
    print(
        json.dumps(result["selected_for_final_test"], indent=2)
    )
    print(f"Saved model: {result['artifact']}")
    print(result["summary"]["mlflow"]["ui_command"])


if __name__ == "__main__":
    main()
