from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from fontsense.cnn_model import FontSenseCNN
from fontsense.train_cnn import read_train_validation_manifest
from fontsense.v2_train import _save_progress_checkpoint, train_v2_candidate, select_v2_candidate, _initialize_from_checkpoint


def test_v2_candidate_config_has_three_meaningful_small_models():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/v2/cnn_experiments.json").read_text(encoding="utf-8"))
    candidates = config["candidates"]
    assert [item["id"] for item in candidates] == ["A", "B", "C"]
    assert [item["width"] for item in candidates] == [16, 24, 32]
    assert len({(item["width"], item["dropout"], item["learning_rate"]) for item in candidates}) == 3
    assert config["image_size"] == [112, 48]
    assert config["max_epochs"] >= 30


def test_v2_training_reader_never_retains_test_rows(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,family,category,split\n"
        "train.png,Train Family,serif,train\n"
        "validation.png,Validation Family,serif,validation\n"
        "secret-test.png,Test Family,serif,test\n",
        encoding="utf-8",
    )
    frame = read_train_validation_manifest(manifest)
    assert set(frame["split"]) == {"train", "validation"}
    assert "secret-test.png" not in frame["image_path"].tolist()


def test_v2_best_checkpoint_is_persisted_before_next_epoch(tmp_path: Path):
    model = FontSenseCNN(num_classes=5, width=16, dropout=0.25)
    checkpoint = tmp_path / "candidate" / "cnn_model.pt"
    candidate = {
        "name": "Test candidate",
        "width": 16,
        "dropout": 0.25,
    }

    _save_progress_checkpoint(
        checkpoint,
        model.state_dict(),
        ["display", "handwriting", "monospace", "sans_serif", "serif"],
        candidate,
        (112, 48),
        {"enabled": True},
        13,
        0.7652,
        0.7647,
        42,
        14_000,
        3_000,
    )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["selected_validation_run"]["best_epoch"] == 13
    assert saved["selected_validation_run"]["validation_macro_f1"] == 0.7652
    assert saved["training_data"]["test_images_loaded"] == 0
    assert not checkpoint.with_suffix(".pt.tmp").exists()


def test_stop_signal_prevents_starting_training(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    signal = tmp_path / "STOP"
    signal.touch()
    with pytest.raises(RuntimeError, match="Training stop signal exists"):
        train_v2_candidate(
            "C", manifest_path=tmp_path / "missing.csv", split_path=tmp_path / "missing_split.csv",
            config_path=root / "config/v2/cnn_experiments.json", artifact_dir=tmp_path,
            report_dir=tmp_path, tracking_dir=tmp_path, stop_file=signal,
        )


def test_selection_rejects_test_contaminated_evidence(tmp_path: Path):
    directory = tmp_path / "candidate_a"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({
        "status": "passed", "candidate_id": "A", "test_images_loaded": 1,
        "test_rows_evaluated": 0, "test_metrics_recorded": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="validation-only evidence"):
        select_v2_candidate(artifact_dir=tmp_path / "artifacts", report_dir=tmp_path)


def test_refinement_requires_matching_initial_checkpoint_hash():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/v2/cnn_refinement.json").read_text(encoding="utf-8"))
    candidate = dict(config["candidates"][0])
    model = FontSenseCNN(num_classes=5, width=24, dropout=0.28)
    candidate["initial_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Initialization checkpoint hash mismatch"):
        _initialize_from_checkpoint(model, candidate, ["display", "handwriting", "monospace", "sans_serif", "serif"], (112, 48))


def test_completed_candidate_reuses_relative_output_paths_without_training(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    result = train_v2_candidate(
        "B", manifest_path="reports/v2/data/full_manifest.csv",
        split_path="data/v2/frozen_family_split.csv", config_path="config/v2/cnn_experiments.json",
        artifact_dir="artifacts/v2/cnn", report_dir="reports/v2/cnn", tracking_dir="reports/v2/mlruns",
    )
    assert result["status"] == "reused"
    assert result["validation_macro_f1"] == pytest.approx(0.7958228907353944)
