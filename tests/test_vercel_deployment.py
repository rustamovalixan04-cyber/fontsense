from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_configuration_uses_slim_cpu_runtime() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    install_command = config["installCommand"]
    function = config["functions"]["api/index.py"]

    assert "requirements-vercel.txt" in install_command
    assert config["framework"] == "fastapi"
    assert config["fluid"] is True
    assert function["maxDuration"] == 300
    assert "artifacts/cnn/**" in function["includeFiles"]

    requirements = (ROOT / "requirements-vercel.txt").read_text(
        encoding="utf-8"
    )
    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.13.0+cpu" in requirements
    assert "torchvision==0.28.0+cpu" in requirements
    assert "mlflow" not in requirements


def test_vercel_asgi_app_and_health_check() -> None:
    from api.index import app
    from app import FINAL_CHECKPOINT_SHA256

    with TestClient(app) as client:
        page = client.get("/")
        health = client.get("/healthz")

    assert page.status_code == 200
    assert "FontSense" in page.text
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
    }
