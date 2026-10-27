from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from api import index as api
from fontsense.inference import FontSensePredictor, validate_probabilities
from fontsense.mlflow_utils import optional_mlflow_run


def test_transparent_text_is_composited_on_white():
    image = Image.new("RGBA", (200, 80), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((20, 20, 150, 45), fill=(0, 0, 0, 255))
    prepared = FontSensePredictor.prepare_image(image)
    assert prepared.getpixel((0, 0)) == (255, 255, 255)
    assert prepared.getpixel((30, 30)) == (0, 0, 0)
    payload = BytesIO()
    image.save(payload, format="PNG")
    assert np.array_equal(api.decode_image(payload.getvalue()), prepared)


def test_exif_orientation_is_applied_once():
    image = Image.new("RGB", (100, 40), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 30, 39), fill="black")
    image.getexif()[274] = 6
    prepared = FontSensePredictor.prepare_image(image)
    assert prepared.size == (40, 100)
    assert prepared.getpixel((20, 10)) == (0, 0, 0)
    assert prepared.getpixel((20, 80)) == (255, 255, 255)


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.1])
def test_invalid_threshold_is_rejected_before_loading_model(threshold):
    with pytest.raises(ValueError, match="threshold must be between"):
        FontSensePredictor(artifact_dir="missing", model="cnn", threshold=threshold)


@pytest.mark.parametrize("values", [
    [0.2, 0.2, 0.2, 0.2, float("nan")],
    [-0.1, 0.3, 0.2, 0.2, 0.4],
    [0.2, 0.2],
    [0.2] * 4 + [0.9],
])
def test_invalid_probabilities_never_become_a_prediction(values):
    with pytest.raises(RuntimeError, match="invalid class probabilities"):
        validate_probabilities(values, 5)


def test_actual_file_format_is_checked_even_with_png_mime():
    payload = BytesIO()
    Image.new("RGB", (80, 80)).save(payload, format="GIF")
    with TestClient(api.app) as client:
        result = client.post("/predict", content=payload.getvalue(), headers={"content-type": "image/png"})
    assert result.status_code == 415


def test_streamed_upload_limit_does_not_depend_on_content_length(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 64)
    with TestClient(api.app) as client:
        result = client.post("/predict", content=iter([b"x" * 40, b"y" * 40]), headers={"content-type": "image/png"})
    assert result.status_code == 413


def test_oversized_dimensions_are_rejected_before_decoding(monkeypatch):
    class OversizedImage:
        format = "PNG"
        size = (6001, 20)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def load(self):
            pytest.fail("Oversized pixels were decoded")

    monkeypatch.setattr(api.Image, "open", lambda *args: OversizedImage())
    with pytest.raises(api.HTTPException) as error:
        api.decode_image(b"placeholder")
    assert error.value.status_code == 413


def test_pillow_decompression_bomb_becomes_clear_error(monkeypatch):
    def bomb(*args):
        raise Image.DecompressionBombError("oversized image header")
    monkeypatch.setattr(api.Image, "open", bomb)
    with pytest.raises(api.HTTPException) as error:
        api.decode_image(b"image")
    assert error.value.status_code == 413


def test_checkpoint_load_error_is_safe_http_response(monkeypatch):
    def broken_runtime():
        raise RuntimeError("private path to a missing checkpoint")

    monkeypatch.setattr(api, "get_runtime", broken_runtime)
    payload = BytesIO()
    Image.new("RGB", (80, 80), "white").save(payload, format="PNG")
    with TestClient(api.app) as client:
        result = client.post("/predict", content=payload.getvalue(), headers={"content-type": "image/png"})
    assert result.status_code == 500
    assert "private path" not in result.text
    assert "could not process" in result.json()["detail"]


def test_web_runtime_avoids_loading_gradio():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "from fontsense import runtime; import sys; assert 'gradio' not in sys.modules; assert runtime.FINAL_CNN_PREDICTOR.classes == list(runtime.FINAL_CLASS_ORDER)"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_mlflow_does_not_hide_import_errors_inside_training(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        set_tracking_uri=lambda *args: None,
        set_experiment=lambda *args: None,
        start_run=lambda **kwargs: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with pytest.raises(ImportError, match="missing training dependency"):
        with optional_mlflow_run("test", "test", tracking_dir=tmp_path):
            raise ImportError("missing training dependency")
