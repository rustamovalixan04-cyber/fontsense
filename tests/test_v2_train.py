from __future__ import annotations

import json
from pathlib import Path

import torch

from fontsense.cnn_model import FontSenseCNN
from fontsense.train_cnn import read_train_validation_manifest
from fontsense.v2_train import _save_progress_checkpoint


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
