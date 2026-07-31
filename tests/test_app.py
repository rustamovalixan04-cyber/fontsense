from __future__ import annotations

import inspect
import json
from pathlib import Path

from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont
import pytest
import torch

import app
from fontsense.train_cnn import build_image_transform


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASSES = [
    "display",
    "handwriting",
    "monospace",
    "sans_serif",
    "serif",
]


@pytest.fixture
def readable_image() -> Image.Image:
    image = Image.new("RGB", (224, 96), "white")
    font = ImageFont.truetype(
        str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"),
        36,
    )
    ImageDraw.Draw(image).text((14, 24), "FontSense", font=font, fill="black")
    return image


def stub_prediction(confidence: float) -> dict:
    return {
        "predicted_category": "sans_serif",
        "confidence": confidence,
        "probabilities": {
            "display": 0.05,
            "handwriting": 0.05,
            "monospace": 0.05,
            "sans_serif": confidence,
            "serif": 0.85 - confidence,
        },
    }


def test_final_checkpoint_loads_and_matches_frozen_record():
    freeze = json.loads(
        (
            ROOT / "reports" / "final_evaluation" / "pre_test_freeze.json"
        ).read_text(encoding="utf-8")
    )

    assert app.FINAL_CHECKPOINT_PATH == (
        ROOT / "artifacts" / "cnn" / "cnn_model.pt"
    )
    assert app.FINAL_CHECKPOINT_SHA256 == (
        freeze["selected_model"]["checkpoint_sha256"]
    )
    assert app.FINAL_CNN_PREDICTOR.model_type == "cnn"
    assert app.FINAL_CNN_PREDICTOR.pipeline.training is False
    assert app.FROZEN_THRESHOLD == pytest.approx(0.60)


def test_final_class_order_is_exact():
    assert list(app.FINAL_CLASS_ORDER) == EXPECTED_CLASSES
    assert app.FINAL_CNN_PREDICTOR.classes == EXPECTED_CLASSES


def test_final_preprocessing_output_shape(readable_image):
    tensor = app.FINAL_CNN_PREDICTOR.preprocess(readable_image)
    evaluation_transform = build_image_transform(
        (112, 48),
        training=False,
        augmentation={"enabled": False},
    )
    expected_tensor = evaluation_transform(readable_image)

    assert tuple(tensor.shape) == (1, 48, 112)
    assert torch.equal(tensor, expected_tensor)
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0
    assert app.FINAL_CNN_PREDICTOR.preprocessing == {
        "image_size": [112, 48],
        "grayscale": True,
        "normalize_mean": [0.5],
        "normalize_std": [0.5],
    }


def test_final_cnn_returns_valid_five_class_prediction(readable_image):
    prediction = app.FINAL_CNN_PREDICTOR.predict(readable_image)

    assert prediction["predicted_category"] in EXPECTED_CLASSES
    assert list(prediction["probabilities"]) == EXPECTED_CLASSES
    assert sum(prediction["probabilities"].values()) == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["accepted"] is (
        prediction["confidence"] >= app.FROZEN_THRESHOLD
    )
    assert prediction["uncertain"] is not prediction["accepted"]


def test_accepted_prediction_wording():
    result, probabilities, status = app.format_prediction(
        stub_prediction(0.80)
    )

    assert "Predicted category: Sans serif" in result
    assert "Confidence: 80.0%" in result
    assert "Accepted" in status
    assert "Uncertain" not in status
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_uncertain_prediction_wording():
    result, probabilities, status = app.format_prediction(
        stub_prediction(0.55)
    )

    assert "Possible category: Sans serif" in result
    assert "Confidence: 55.0%" in result
    assert "Low confidence" in result
    assert "Uncertain" in status
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_threshold_changes_only_status_and_not_probabilities():
    prediction = stub_prediction(0.80)

    accepted_result, accepted_probabilities, accepted_status = (
        app.format_prediction(prediction, threshold=0.60)
    )
    uncertain_result, uncertain_probabilities, uncertain_status = (
        app.format_prediction(prediction, threshold=0.90)
    )

    assert accepted_probabilities == uncertain_probabilities
    assert "Predicted category" in accepted_result
    assert "Accepted" in accepted_status
    assert "Possible category" in uncertain_result
    assert "Uncertain" in uncertain_status


@pytest.mark.parametrize(
    ("invalid_input", "expected_message"),
    [
        (None, "No image uploaded"),
        ("not-an-image", "Unsupported input"),
        (Image.new("RGB", (80, 80), "white"), "blank or nearly blank"),
    ],
)
def test_invalid_input_is_handled(invalid_input, expected_message):
    result, probabilities, status = app.classify(invalid_input)

    assert expected_message in f"{result} {status}"
    assert probabilities == {}


def test_corrupted_image_is_handled():
    class CorruptedImage(Image.Image):
        def load(self):
            raise OSError("broken image data")

    result, probabilities, status = app.classify(CorruptedImage())

    assert "Could not classify image" in result
    assert "valid PNG or JPEG" in status
    assert probabilities == {}


def test_inference_error_is_handled_without_exposing_details(
    monkeypatch,
    readable_image,
):
    class BrokenPredictor:
        def predict(self, image):
            raise RuntimeError("private implementation detail")

    monkeypatch.setattr(app, "FINAL_CNN_PREDICTOR", BrokenPredictor())

    result, probabilities, status = app.classify(readable_image)

    assert "Inference error" in result
    assert "private implementation detail" not in status
    assert probabilities == {}


def test_final_model_is_not_reloaded_for_each_prediction(
    monkeypatch,
    readable_image,
):
    class CountingPredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, image):
            self.calls += 1
            return stub_prediction(0.80)

    predictor = CountingPredictor()
    monkeypatch.setattr(app, "FINAL_CNN_PREDICTOR", predictor)

    def fail_if_reloaded(contract):
        pytest.fail("The final CNN was reloaded during prediction.")

    monkeypatch.setattr(app, "load_final_cnn_predictor", fail_if_reloaded)

    app.classify(readable_image)
    app.classify(readable_image)

    assert predictor.calls == 2


def test_interface_defaults_to_final_cnn_and_has_predict_and_reset():
    assert app.build_demo() is app.demo
    assert app.model.value == app.FINAL_MODEL_LABEL
    assert app.FINAL_MODEL_LABEL in app.MODEL_CHOICES

    predict_inputs = [app.image._id, app.model._id]
    predict_outputs = [
        app.result_text._id,
        app.probabilities._id,
        app.status_text._id,
    ]
    assert any(
        dependency["api_name"] == "predict"
        and dependency["inputs"] == predict_inputs
        and dependency["outputs"] == predict_outputs
        for dependency in app.demo.config["dependencies"]
    )
    assert any(
        dependency["api_name"] == "reset"
        for dependency in app.demo.config["dependencies"]
    )


def test_app_does_not_load_manifests_or_rerun_final_evaluation():
    source = inspect.getsource(app)

    assert "full_manifest.csv" not in source
    assert "google_fonts_final_family_split.csv" not in source
    assert "fontsense.final_evaluation" not in source
    assert "evaluate_final_cnn" not in source
