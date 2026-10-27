"""Frozen inference runtime shared by the web API and Gradio interface."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .inference import FontSensePredictor

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = (
    Path(sys._MEIPASS).resolve()
    if getattr(sys, "frozen", False) and getattr(sys, "_MEIPASS", None)
    else SOURCE_ROOT
)
FINAL_MODEL_LABEL = "Final CNN"
FROZEN_THRESHOLD = 0.60
FINAL_CLASS_ORDER = ("display", "handwriting", "monospace", "sans_serif", "serif")
FINAL_PREPROCESSING = {
    "image_size": [112, 48],
    "grayscale": True,
    "normalize_mean": [0.5],
    "normalize_std": [0.5],
}
FREEZE_PATH = ROOT / "reports/final_evaluation/pre_test_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_contract() -> dict:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    checkpoint = (ROOT / freeze["selected_model"]["checkpoint"]).resolve()
    if ROOT.resolve() not in checkpoint.parents:
        raise RuntimeError("The frozen checkpoint path leaves the project folder.")
    actual_hash = sha256_file(checkpoint)
    if actual_hash != freeze["selected_model"]["checkpoint_sha256"].lower():
        raise RuntimeError("The final CNN checkpoint hash does not match the frozen evaluation.")
    if float(freeze["uncertainty_threshold"]["selected_threshold"]) != FROZEN_THRESHOLD:
        raise RuntimeError("The frozen confidence threshold changed.")
    if tuple(freeze["class_order"]) != FINAL_CLASS_ORDER:
        raise RuntimeError("The frozen CNN class order changed.")
    if freeze["preprocessing"] != FINAL_PREPROCESSING:
        raise RuntimeError("The frozen CNN preprocessing changed.")
    return {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": actual_hash,
        "threshold": FROZEN_THRESHOLD,
        "class_order": list(FINAL_CLASS_ORDER),
        "preprocessing": dict(FINAL_PREPROCESSING),
    }


def load_final_cnn_predictor(contract: dict) -> FontSensePredictor:
    predictor = FontSensePredictor(
        artifact_dir=contract["checkpoint_path"].parent,
        model="cnn",
        threshold=contract["threshold"],
    )
    if predictor.classes != contract["class_order"]:
        raise RuntimeError("The loaded CNN class order differs from the freeze.")
    if predictor.preprocessing != contract["preprocessing"]:
        raise RuntimeError("The loaded CNN preprocessing differs from the freeze.")
    return predictor


FROZEN_CONTRACT = load_frozen_contract()
FINAL_CHECKPOINT_PATH = FROZEN_CONTRACT["checkpoint_path"]
FINAL_CHECKPOINT_SHA256 = FROZEN_CONTRACT["checkpoint_sha256"]
FINAL_CNN_PREDICTOR = load_final_cnn_predictor(FROZEN_CONTRACT)


def get_predictor(model_name: str = FINAL_MODEL_LABEL) -> FontSensePredictor:
    if model_name != FINAL_MODEL_LABEL:
        raise ValueError("Choose an available FontSense model.")
    return FINAL_CNN_PREDICTOR
