from PIL import Image
import pytest

import app


@pytest.mark.parametrize("threshold", [0.70, 0.90])
def test_classify_warns_when_confidence_is_below_threshold(monkeypatch, threshold):
    class StubPredictor:
        def __init__(self, artifact_dir, model_name, selected_threshold):
            self.threshold = float(selected_threshold)

        def predict(self, image):
            confidence = 0.633
            return {
                "predicted_category": "serif",
                "confidence": confidence,
                "probabilities": {"serif": confidence},
                "warning": "Low confidence" if confidence < self.threshold else "",
                "inference_ms": 1.0,
            }

    monkeypatch.setattr(app, "AVAILABLE", ["hog"])
    monkeypatch.setattr(app, "FontSensePredictor", StubPredictor)
    app._cache.clear()

    _, _, warning = app.classify(Image.new("RGB", (40, 40), "white"), "hog", threshold)

    assert warning == "Low confidence"


def test_threshold_change_reclassifies_the_current_image():
    expected_inputs = [app.image._id, app.model._id, app.threshold._id]
    expected_outputs = [app.result_text._id, app.probabilities._id, app.warning._id]

    assert any(
        (app.threshold._id, "change") in dependency["targets"]
        and dependency["inputs"] == expected_inputs
        and dependency["outputs"] == expected_outputs
        for dependency in app.demo.config["dependencies"]
    )
