from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tomllib

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_configuration_uses_slim_cpu_runtime() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    install_command = config["installCommand"]
    function = config["functions"]["api/index.py"]

    assert "requirements-vercel.txt" in install_command
    assert config["framework"] == "fastapi"
    assert project["tool"]["vercel"]["entrypoint"] == "api.index:app"
    assert config["fluid"] is True
    assert function["maxDuration"] == 300
    assert "artifacts/cnn/**" in function["includeFiles"]
    assert "api/static/**" in function["includeFiles"]

    requirements = (ROOT / "requirements-vercel.txt").read_text(
        encoding="utf-8"
    )
    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.13.0+cpu" in requirements
    assert "torchvision==0.28.0+cpu" in requirements
    assert "mlflow" not in requirements


def test_vercel_asgi_app_and_health_check() -> None:
    from api.index import app, get_runtime
    from app import FINAL_CHECKPOINT_SHA256

    with TestClient(app) as client:
        page = client.get("/")
        health = client.get("/healthz")

    assert page.status_code == 200
    assert "FontSense" in page.text
    assert "/gradio_api/upload" not in page.text
    assert "/gradio_api/queue" not in page.text
    assert 'fetch("/predict"' in page.text
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
    }
    assert get_runtime() is get_runtime()


def test_vercel_prediction_receives_image_bytes_in_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import api.index as vercel_api

    class StubPredictor:
        def predict(self, image: Image.Image) -> dict[str, object]:
            assert image.mode == "RGB"
            return {
                "predicted_category": "serif",
                "confidence": 0.70,
                "probabilities": {
                    "display": 0.05,
                    "handwriting": 0.05,
                    "monospace": 0.05,
                    "sans_serif": 0.15,
                    "serif": 0.70,
                },
            }

    class StubRuntime:
        FINAL_MODEL_LABEL = "Final CNN"
        FROZEN_THRESHOLD = 0.60
        FINAL_CLASS_ORDER = (
            "display",
            "handwriting",
            "monospace",
            "sans_serif",
            "serif",
        )

        @staticmethod
        def get_predictor(model_name: str) -> StubPredictor:
            assert model_name == "Final CNN"
            return StubPredictor()

    monkeypatch.setattr(vercel_api, "get_runtime", lambda: StubRuntime())
    image = Image.new("RGB", (224, 96), "white")
    ImageDraw.Draw(image).text((20, 30), "FontSense", fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    with TestClient(vercel_api.app) as client:
        response = client.post(
            "/predict",
            content=buffer.getvalue(),
            headers={"content-type": "image/png"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_category"] == "serif"
    assert result["confidence"] == pytest.approx(0.70)
    assert result["accepted"] is True
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("content", "content_type", "expected_status"),
    [
        (b"not an image", "image/png", 400),
        (b"plain text", "text/plain", 415),
        (b"", "image/jpeg", 400),
    ],
)
def test_vercel_prediction_rejects_invalid_uploads(
    content: bytes,
    content_type: str,
    expected_status: int,
) -> None:
    from api.index import app

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=content,
            headers={"content-type": content_type},
        )

    assert response.status_code == expected_status
    assert response.json()["detail"]
