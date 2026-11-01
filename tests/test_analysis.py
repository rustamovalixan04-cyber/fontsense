"""Exercise the real API adapter and SDK using local, unbilled HTTP responses."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json

from fastapi.testclient import TestClient
import httpx
import httpx2
from PIL import Image
import pytest

from api import analysis
from api.index import app


ENV_KEYS = (
    "OPENAI_API_KEY", "OPENAI_VISION_MODEL", "FONTSENSE_ANALYSIS_ACCESS_CODE",
    "UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN",
    "FONTSENSE_ANALYSIS_DAILY_LIMIT",
)
CODE = "private-test-code-" + "x" * 32
HEADERS = {"authorization": f"Bearer {CODE}", "content-type": "image/png"}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def configured(monkeypatch):
    for key, value in {
        "OPENAI_API_KEY": "sk-test-never-a-real-key",
        "FONTSENSE_ANALYSIS_ACCESS_CODE": CODE,
        "UPSTASH_REDIS_REST_URL": "https://quota-test.upstash.io",
        "UPSTASH_REDIS_REST_TOKEN": "redis-test-secret",
    }.items():
        monkeypatch.setenv(key, value)
    return analysis.configuration()


@pytest.fixture
def crop():
    output = BytesIO()
    Image.new("RGB", (300, 80), "white").save(output, format="PNG")
    return output.getvalue()


def sample_analysis():
    return {
        "readable_text": True, "observed_text": "FontSense",
        "summary": "A geometric-looking text crop.", "style": "sans_serif",
        "weight": "Medium", "spacing": "Open", "proportions": "Wide",
        "contrast": "Low stroke contrast",
        "suggested_fonts": [{"family": "Montserrat", "reason": "Similar geometric shapes."}],
        "pairing_suggestions": [{"family": "Lora", "reason": "A contrasting serif."}],
        "improvements": ["Check spacing around the capital F."],
        "limitations": ["The font family cannot be verified from this crop."],
    }


def install_provider(monkeypatch, *, output=None, status="completed", error=None, refusal=False):
    requests = []
    real_sdk = analysis.AsyncOpenAI

    def handler(request):
        requests.append(request)
        if error:
            return httpx2.Response(error, json={"error": {"message": "PRIVATE upstream diagnostic"}})
        content = (
            [{"type": "refusal", "refusal": "Cannot analyze this crop."}] if refusal
            else [{"type": "output_text", "text": json.dumps(output or sample_analysis()), "annotations": []}]
        )
        return httpx2.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": analysis.DEFAULT_MODEL, "status": status,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed", "content": content}],
        })

    def sdk(**kwargs):
        assert kwargs["max_retries"] == 0
        return real_sdk(**kwargs, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))

    async def reserve(_settings):
        pass

    monkeypatch.setattr(analysis, "AsyncOpenAI", sdk)
    monkeypatch.setattr(analysis, "reserve_usage", reserve)
    return requests


def test_disabled_without_credentials_does_not_load_classifier_or_call_provider(monkeypatch, crop):
    import api.index as api
    monkeypatch.setattr(api, "get_runtime", lambda: pytest.fail("Analysis must not load the classifier"))
    with TestClient(app) as client:
        assert client.get("/analysis/status").json() == {"enabled": False}
        assert client.post("/analyze", content=crop, headers=HEADERS).status_code == 503


@pytest.mark.parametrize("missing", ENV_KEYS[:1] + ENV_KEYS[2:5])
def test_missing_required_setting_fails_closed(configured, monkeypatch, missing):
    monkeypatch.delenv(missing)
    assert analysis.configuration() is None


@pytest.mark.parametrize("value", ["0", "501", "invalid"])
def test_invalid_daily_limit_disables_analysis(configured, monkeypatch, value):
    monkeypatch.setenv("FONTSENSE_ANALYSIS_DAILY_LIMIT", value)
    assert analysis.configuration() is None


def test_private_access_code_is_required_before_image_or_paid_processing(configured, monkeypatch):
    monkeypatch.setattr(analysis, "analyze_image", lambda *_: pytest.fail("Unauthorized provider call"))
    with TestClient(app) as client:
        assert client.get("/analysis/status").json() == {"enabled": True}
        for authorization in ("", "Bearer wrong", "Basic " + CODE):
            response = client.post("/analyze", content=b"bad", headers={"authorization": authorization})
            assert response.status_code == 401
            assert "sk-test" not in response.text and CODE not in response.text


def test_vision_analysis_uses_real_sdk_and_keeps_secrets_server_side(configured, monkeypatch, crop):
    requests = install_provider(monkeypatch)
    with TestClient(app) as client:
        response = client.post("/analyze", content=crop, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["analysis"]["style"] == "sans_serif"
    assert "not verified" in response.json()["notice"]
    assert "confidence" not in response.json()["analysis"]
    assert "sk-test" not in response.text and CODE not in response.text
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    body = json.loads(request.content)
    assert body["model"] == "gpt-6.1-sol"
    assert body["store"] is False
    assert body["text"]["format"]["strict"] is True
    assert body["max_output_tokens"] == analysis.MAX_OUTPUT_TOKENS
    assert body["reasoning"]["effort"] == "low"
    image = body["input"][0]["content"][1]
    assert image["type"] == "input_image" and image["detail"] == "high"
    assert image["image_url"].startswith("data:image/jpeg;base64,")


def test_unreadable_crop_never_returns_font_suggestions(configured, monkeypatch, crop):
    output = sample_analysis()
    output["readable_text"] = False
    install_provider(monkeypatch, output=output)
    with TestClient(app) as client:
        response = client.post("/analyze", content=crop, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["analysis"]["suggested_fonts"] == []
    assert response.json()["analysis"]["pairing_suggestions"] == []


@pytest.mark.parametrize("error", [400, 401, 429, 500])
def test_provider_errors_are_sanitized_and_not_retried(configured, monkeypatch, crop, error):
    requests = install_provider(monkeypatch, error=error)
    with TestClient(app) as client:
        response = client.post("/analyze", content=crop, headers=HEADERS)
    assert response.status_code == 502
    assert "PRIVATE" not in response.text and "sk-test" not in response.text
    assert len(requests) == 1


@pytest.mark.parametrize("case", ["refusal", "incomplete", "malformed"])
def test_incomplete_or_invalid_analysis_is_not_presented_as_success(configured, monkeypatch, crop, case):
    install_provider(
        monkeypatch, refusal=case == "refusal",
        status="incomplete" if case == "incomplete" else "completed",
        output={"unexpected": "shape"} if case == "malformed" else None,
    )
    with TestClient(app) as client:
        assert client.post("/analyze", content=crop, headers=HEADERS).status_code == 502


@pytest.mark.parametrize("failure, expected", [
    (analysis.AnalysisLimitReached, 429), (analysis.AnalysisUnavailable, 503), (TimeoutError, 502),
])
def test_dependency_failure_does_not_touch_classifier(configured, monkeypatch, crop, failure, expected):
    async def fail(*args):
        raise failure("PRIVATE failure")
    monkeypatch.setattr(analysis, "analyze_image", fail)
    with TestClient(app) as client:
        response = client.post("/analyze", content=crop, headers=HEADERS)
    assert response.status_code == expected
    assert "PRIVATE" not in response.text


@pytest.mark.parametrize("body, mime, status", [
    (b"corrupt", "image/png", 400), (b"", "image/jpeg", 400),
    (b"text", "text/plain", 415), (b"x" * (4 * 1024 * 1024 + 1), "image/png", 413),
])
def test_invalid_upload_never_uses_paid_request(configured, monkeypatch, body, mime, status):
    monkeypatch.setattr(analysis, "analyze_image", lambda *_: pytest.fail("Invalid image billed"))
    with TestClient(app) as client:
        response = client.post("/analyze", content=body, headers={**HEADERS, "content-type": mime})
    assert response.status_code == status


def test_streamed_upload_cannot_bypass_size_limit(configured, monkeypatch):
    monkeypatch.setattr(analysis, "analyze_image", lambda *_: pytest.fail("Oversized stream billed"))
    chunks = (b"x" * (1024 * 1024) for _ in range(5))
    with TestClient(app) as client:
        response = client.post("/analyze", content=chunks, headers=HEADERS)
    assert response.status_code == 413


def test_disguised_non_png_jpeg_is_rejected(configured):
    output = BytesIO()
    Image.new("RGB", (10, 10), "white").save(output, format="GIF")
    with TestClient(app) as client:
        response = client.post("/analyze", content=output.getvalue(), headers=HEADERS)
    assert response.status_code == 415


def test_deployment_bundle_includes_analysis_module_and_dependency_file():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "vercel.json").read_text())
    assert "api/analysis.py" in config["functions"]["api/index.py"]["includeFiles"]
    ignore = (root / ".vercelignore").read_text()
    assert "!/api/analysis.py" in ignore
    assert "!/requirements-analysis.txt" in ignore
    assert "-r requirements-analysis.txt" in (root / "requirements-vercel.txt").read_text()


@pytest.mark.parametrize("result, expected", [
    ({"result": 1}, None), ({"result": 0}, analysis.AnalysisLimitReached),
    ({"error": "Redis failed"}, analysis.AnalysisUnavailable),
    ({"result": True}, analysis.AnalysisUnavailable),
])
def test_shared_quota_fails_closed(configured, monkeypatch, result, expected):
    requests = []
    real_client = httpx.AsyncClient
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=result)
    monkeypatch.setattr(analysis.httpx, "AsyncClient", lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(handler)))
    if expected:
        with pytest.raises(expected):
            asyncio.run(analysis.reserve_usage(configured))
    else:
        asyncio.run(analysis.reserve_usage(configured))
    assert len(requests) == 1
    command = json.loads(requests[0].content)
    assert command[:3] == ["EVAL", analysis.QUOTA_SCRIPT, "2"]
    assert command[-1] == "25"


def test_analysis_encoding_removes_metadata_and_bounds_size():
    source = Image.new("RGB", (4000, 1000), "white")
    source.info["comment"] = "private metadata"
    encoded = analysis.image_data_url(source)
    with Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[1]))) as sent:
        assert sent.size == (1600, 400)
        assert "comment" not in sent.info and not sent.getexif()
    assert source.size == (4000, 1000)
