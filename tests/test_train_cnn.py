from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, ImageDraw
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

import fontsense.train_cnn as train_cnn_module
from fontsense.cnn_model import FontSenseCNN
from fontsense.train_cnn import (
    ManifestDataset,
    build_image_transform,
    load_cnn_checkpoint,
    train_cnn,
)


CATEGORIES = ("display", "handwriting", "monospace", "sans_serif", "serif")
ROOT = Path(__file__).parents[1]


def _write_image(path: Path, category_index: int, image_index: int) -> None:
    image = Image.new("RGB", (64, 32), "white")
    draw = ImageDraw.Draw(image)
    x = 4 + category_index * 4
    draw.rectangle(
        (x, 4 + image_index, x + 7 + category_index, 26),
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _tiny_training_files(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, set[str]]:
    rows = []
    split_rows = []
    test_paths: set[str] = set()
    for category_index, category in enumerate(CATEGORIES):
        for split, image_count in (
            ("train", 2),
            ("validation", 1),
            ("test", 1),
        ):
            family = f"{category}_{split}_family"
            split_rows.append(
                {
                    "family": family,
                    "category": category,
                    "split": split,
                }
            )
            for image_index in range(image_count):
                image_path = (
                    tmp_path
                    / "images"
                    / split
                    / f"{category}_{image_index}.png"
                )
                _write_image(image_path, category_index, image_index)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "family": family,
                        "category": category,
                        "split": split,
                    }
                )
                if split == "test":
                    test_paths.add(str(image_path.resolve()))

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    split_path = tmp_path / "family_split.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    config = {
        "experiment_name": "FontSense CNN unit tests",
        "expected_split_sha256": split_hash,
        "expected_family_count": len(split_rows),
        "image_size": [32, 16],
        "grayscale": True,
        "batch_size": 5,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "early_stopping_min_delta": 0.0001,
        "weight_decay": 0.0001,
        "augmentation": {
            "enabled": True,
            "rotation_degrees": 1.0,
            "translate_fraction": [0.01, 0.01],
            "scale_range": [0.99, 1.01],
            "sharpness_probability": 0.1,
            "sharpness_factor": 1.1,
        },
        "experiments": [
            {
                "name": "Reference",
                "reason": "Reference",
                "learning_rate": 0.001,
                "dropout": 0.2,
                "width": 2,
            },
            {
                "name": "Lower learning rate",
                "reason": "Learning-rate check",
                "learning_rate": 0.0003,
                "dropout": 0.2,
                "width": 2,
            },
            {
                "name": "Wider filters",
                "reason": "Width check",
                "learning_rate": 0.001,
                "dropout": 0.2,
                "width": 3,
            },
        ],
    }
    config_path = tmp_path / "cnn_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    baseline_path = tmp_path / "baseline_summary.json"
    baseline_path.write_text(
        json.dumps(
            {
                "majority_baseline": {
                    "validation_macro_f1": 0.0666667,
                    "validation_accuracy": 0.2,
                },
                "best_hog_run": {
                    "validation_macro_f1": 0.69,
                    "validation_accuracy": 0.695,
                    "inference_ms_per_image": 2.2,
                    "saved_model_size_bytes": 35700,
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        manifest_path,
        split_path,
        config_path,
        baseline_path,
        test_paths,
    )


def test_cnn_returns_five_probabilities():
    model = FontSenseCNN(num_classes=5, width=4, dropout=0.2)
    logits = model(torch.zeros(2, 1, 48, 112))
    probabilities = torch.softmax(logits, dim=1)

    assert logits.shape == (2, 5)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_augmentation_is_training_only():
    augmentation = {
        "enabled": True,
        "rotation_degrees": 2.0,
        "translate_fraction": [0.02, 0.03],
        "scale_range": [0.97, 1.03],
        "sharpness_probability": 0.2,
        "sharpness_factor": 1.25,
    }
    train_transform = build_image_transform(
        (112, 48),
        training=True,
        augmentation=augmentation,
    )
    validation_transform = build_image_transform(
        (112, 48),
        training=False,
        augmentation=augmentation,
    )

    assert any(
        isinstance(operation, transforms.RandomAffine)
        for operation in train_transform.transforms
    )
    assert any(
        isinstance(operation, transforms.RandomAdjustSharpness)
        for operation in train_transform.transforms
    )
    assert not any(
        isinstance(
            operation,
            (transforms.RandomAffine, transforms.RandomAdjustSharpness),
        )
        for operation in validation_transform.transforms
    )


def test_manifest_dataset_rejects_test_rows(tmp_path: Path):
    image_path = tmp_path / "test.png"
    _write_image(image_path, 0, 0)
    frame = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "category": "display",
                "split": "test",
            }
        ]
    )
    encoder = LabelEncoder().fit(CATEGORIES)

    with pytest.raises(ValueError, match="forbidden splits"):
        ManifestDataset(
            frame,
            encoder,
            (32, 16),
            training=False,
            augmentation={"enabled": False},
        )


def test_config_has_three_controlled_nonduplicate_experiments():
    config = json.loads(
        (ROOT / "config" / "cnn_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    experiments = config["experiments"]
    assert len(experiments) >= 3
    signatures = {
        (
            float(experiment["learning_rate"]),
            int(experiment["width"]),
            float(experiment["dropout"]),
        )
        for experiment in experiments
    }
    assert len(signatures) == len(experiments)
    reference = experiments[0]
    for experiment in experiments[1:]:
        changes = sum(
            [
                experiment["learning_rate"]
                != reference["learning_rate"],
                experiment["width"] != reference["width"],
                experiment["dropout"] != reference["dropout"],
            ]
        )
        assert changes == 1


def test_training_never_opens_test_images_and_checkpoint_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        manifest_path,
        split_path,
        config_path,
        baseline_path,
        test_paths,
    ) = _tiny_training_files(tmp_path)
    opened_paths: list[str] = []
    original_open = train_cnn_module.Image.open

    def recording_open(path, *args, **kwargs):
        opened_paths.append(str(Path(path).resolve()))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(train_cnn_module.Image, "open", recording_open)
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    result = train_cnn(
        manifest_path,
        config_path,
        artifact_dir,
        report_dir,
        family_split_path=split_path,
        baseline_summary_path=baseline_path,
        tracking_dir=tmp_path / "mlruns",
        epochs=1,
        seed=42,
        enable_mlflow=False,
        device_name="cpu",
    )

    assert test_paths.isdisjoint(opened_paths)
    assert result["summary"]["test_images_loaded"] == 0
    assert result["summary"]["test_rows_evaluated"] == 0
    assert result["summary"]["augmentation"][
        "validation_enabled"
    ] is False
    comparison = pd.read_csv(
        report_dir / "cnn_experiment_comparison.csv"
    )
    assert len(comparison) == 3
    assert set(comparison["validation_accuracy"].between(0, 1)) == {True}
    model, checkpoint = load_cnn_checkpoint(
        artifact_dir / "cnn_model.pt"
    )
    with torch.inference_mode():
        probabilities = torch.softmax(
            model(torch.zeros(1, 1, 16, 32)),
            dim=1,
        ).numpy()
    assert probabilities.shape == (1, 5)
    assert np.isclose(probabilities.sum(), 1.0)
    assert checkpoint["training_data"]["fit_splits"] == ["train"]
    assert len(list((report_dir / "figures").glob("*.png"))) == 6


def test_training_stops_if_frozen_split_hash_changed(tmp_path: Path):
    (
        manifest_path,
        split_path,
        config_path,
        baseline_path,
        _,
    ) = _tiny_training_files(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["expected_split_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AssertionError, match="Frozen family split hash changed"):
        train_cnn(
            manifest_path,
            config_path,
            tmp_path / "artifacts",
            tmp_path / "reports",
            family_split_path=split_path,
            baseline_summary_path=baseline_path,
            enable_mlflow=False,
            device_name="cpu",
        )
