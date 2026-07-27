from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

import fontsense.features as features_module
from fontsense.features import HOGTransformer
from fontsense.train_hog import train_hog


CATEGORIES = ("display", "handwriting", "monospace", "sans_serif", "serif")
ROOT = Path(__file__).parents[1]


def _write_image(path: Path, category_index: int, image_index: int) -> None:
    image = Image.new("RGB", (64, 32), "white")
    draw = ImageDraw.Draw(image)
    x = 4 + category_index * 3
    draw.rectangle(
        (x, 5 + image_index, x + 8 + category_index, 25),
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _tiny_training_files(tmp_path: Path) -> tuple[Path, Path, Path, set[str]]:
    rows = []
    split_rows = []
    test_paths: set[str] = set()
    for category_index, category in enumerate(CATEGORIES):
        for split, images in (("train", 2), ("validation", 1), ("test", 1)):
            family = f"{category}_{split}_family"
            split_rows.append(
                {
                    "family": family,
                    "category": category,
                    "split": split,
                }
            )
            for image_index in range(images):
                image_path = tmp_path / "images" / split / (
                    f"{category}_{image_index}.png"
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
    family_split_path = tmp_path / "family_split.csv"
    pd.DataFrame(split_rows).to_csv(family_split_path, index=False)
    split_hash = hashlib.sha256(family_split_path.read_bytes()).hexdigest()
    config = {
        "experiment_name": "FontSense HOG unit tests",
        "expected_split_sha256": split_hash,
        "orientations": 9,
        "cells_per_block": [2, 2],
        "experiments": [
            {
                "name": "Reference",
                "reason": "Reference",
                "image_size": [32, 16],
                "pixels_per_cell": [4, 4],
                "C": 1.0,
            },
            {
                "name": "Compact",
                "reason": "Resolution check",
                "image_size": [24, 16],
                "pixels_per_cell": [4, 4],
                "C": 1.0,
            },
            {
                "name": "Regularized",
                "reason": "Regularization check",
                "image_size": [32, 16],
                "pixels_per_cell": [4, 4],
                "C": 0.2,
            },
        ],
        "logistic_regression": {
            "solver": "lbfgs",
            "max_iter": 100,
            "tolerance": 0.0001,
        },
    }
    config_path = tmp_path / "hog_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return manifest_path, family_split_path, config_path, test_paths


def test_hog_transformer_is_deterministic_and_accepts_saved_feature_vectors(
    tmp_path: Path,
):
    image_path = tmp_path / "sample.png"
    _write_image(image_path, category_index=2, image_index=0)
    transformer = HOGTransformer(
        image_size=(32, 16),
        pixels_per_cell=(4, 4),
    ).fit([str(image_path)])

    first = transformer.transform([str(image_path)])
    second = transformer.transform([str(image_path)])
    assert transformer.fit_sample_count_ == 1
    assert first.shape[0] == 1
    assert np.array_equal(first, second)
    assert np.array_equal(transformer.transform(first), first)


def test_config_has_meaningful_nonduplicate_hog_experiments():
    config_path = ROOT / "config" / "hog_experiments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments = config["experiments"]
    assert len(experiments) >= 3
    signatures = {
        (
            tuple(experiment["image_size"]),
            tuple(experiment["pixels_per_cell"]),
            float(experiment["C"]),
        )
        for experiment in experiments
    }
    assert len(signatures) == len(experiments)

    reference = experiments[0]
    for experiment in experiments[1:]:
        changed = sum(
            [
                experiment["image_size"] != reference["image_size"],
                experiment["pixels_per_cell"]
                != reference["pixels_per_cell"],
                float(experiment["C"]) != float(reference["C"]),
            ]
        )
        assert changed == 1


def test_training_uses_train_and_validation_images_but_never_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, split_path, config_path, test_paths = _tiny_training_files(
        tmp_path
    )
    opened_paths: list[str] = []
    original_extract_hog = features_module.extract_hog

    def recording_extract_hog(image, **kwargs):
        opened_paths.append(str(Path(image).resolve()))
        return original_extract_hog(image, **kwargs)

    monkeypatch.setattr(features_module, "extract_hog", recording_extract_hog)
    artifact_dir = tmp_path / "artifacts"
    report_dir = tmp_path / "reports"
    result = train_hog(
        manifest_path,
        config_path,
        artifact_dir,
        report_dir,
        seed=42,
        family_split_path=split_path,
        tracking_dir=tmp_path / "mlruns",
        enable_mlflow=False,
        figure_dir=tmp_path / "figures",
    )

    assert test_paths.isdisjoint(opened_paths)
    assert result["summary"]["test_images_loaded"] == 0
    assert result["summary"]["test_rows_evaluated"] == 0
    assert result["summary"]["preprocessing_fit_rows"] == 10
    comparison = pd.read_csv(report_dir / "validation_comparison.csv")
    assert len(comparison) == 4
    assert set(comparison["validation_rows"]) == {5}
    assert set(comparison.loc[comparison["model_type"] != "majority_class", "preprocessing_fit_rows"]) == {10}

    pipeline = joblib.load(artifact_dir / "hog_pipeline.joblib")
    probabilities = pipeline.predict_proba(
        [str(tmp_path / "images" / "validation" / "display_0.png")]
    )
    assert probabilities.shape == (1, 5)
    assert np.isclose(probabilities.sum(), 1.0)


def test_training_stops_if_frozen_split_hash_changed(tmp_path: Path):
    manifest_path, split_path, config_path, _ = _tiny_training_files(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["expected_split_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AssertionError, match="Frozen family split hash changed"):
        train_hog(
            manifest_path,
            config_path,
            tmp_path / "artifacts",
            tmp_path / "reports",
            family_split_path=split_path,
            enable_mlflow=False,
            figure_dir=tmp_path / "figures",
        )


def test_saved_full_baseline_is_validation_only_and_reloadable():
    summary = json.loads(
        (
            ROOT
            / "reports"
            / "baseline"
            / "baseline_validation_summary.json"
        ).read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(
        ROOT / "reports" / "baseline" / "validation_comparison.csv"
    )
    pipeline = joblib.load(
        ROOT / "artifacts" / "baseline" / "hog_pipeline.joblib"
    )

    assert summary["status"] == "passed"
    assert summary["train_rows_fitted"] == 2400
    assert summary["validation_rows_compared"] == 600
    assert summary["test_images_loaded"] == 0
    assert summary["test_rows_evaluated"] == 0
    assert summary["test_metrics_recorded"] is False
    assert summary["family_overlap_count"] == 0
    assert (
        summary["frozen_split_sha256_before"]
        == summary["frozen_split_sha256_after"]
        == "f6cdd858449a4143993b051e9de83578cd423544135dbf6cc17f004359d2e17b"
    )
    assert len(comparison) == 5
    assert comparison["mlflow_run_id"].astype(str).str.len().eq(32).all()
    assert pipeline.named_steps["hog"].fit_sample_count_ == 2400
