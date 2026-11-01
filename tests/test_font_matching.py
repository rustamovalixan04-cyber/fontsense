"""Product checks are separate from the untouched capstone held-out dataset."""
from io import BytesIO
import sys
import subprocess

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pytest

from api.compact import CHECKPOINT_SHA256, get_model, text_crop
from api.font_match import catalog, identify, render_reference
from api.index import app


def png(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def specimen(font_id, text="Visual craft"):
    rendered = render_reference(font_id, text)
    return ImageOps.expand(rendered, (80, 60), "white")


def test_font_matching_returns_real_catalog_entries_and_text():
    result = identify(specimen("tinos"))
    assert result["category"] == "serif"
    assert result["text"].replace(" ", "").lower() == "visualcraft"
    assert result["matches"][0]["family"] == "Tinos"
    assert result["needs_text"] is False
    known = {record["id"] for record in catalog()}
    assert all(match["id"] in known for match in result["matches"])
    assert "confidence" not in result["matches"][0]


def test_dark_and_transparent_inputs_keep_the_same_match():
    image = specimen("arimo", "Clean Lines")
    dark = ImageOps.invert(image)
    white = identify(image, "Clean Lines")
    inverted = identify(dark, "Clean Lines")
    assert white["category"] == inverted["category"] == "sans_serif"
    assert white["matches"][0]["id"] == inverted["matches"][0]["id"] == "arimo"
    transparent = image.convert("RGBA")
    transparent.putalpha(ImageOps.invert(image.convert("L")))
    # Preserve black foreground pixels and make the white background transparent.
    transparent.paste((0, 0, 0), (0, 0, transparent.width, transparent.height), transparent.getchannel("A"))
    assert text_crop(transparent).getbbox()


def test_manual_text_correction_skips_ocr(monkeypatch):
    import api.font_match as matching
    monkeypatch.setattr(matching, "read_text", lambda _: pytest.fail("Manual text should skip OCR"))
    result = identify(specimen("cousine", "Hello 42"), "Hello 42")
    assert result["text"] == "Hello 42"
    assert result["category"] == "monospace"


def test_blank_image_is_rejected_instead_of_returning_a_font():
    with TestClient(app) as client:
        result = client.post("/identify", content=png(Image.new("RGB", (300, 90), "white")), headers={"content-type": "image/png"})
    assert result.status_code == 400


def test_ocr_uncertainty_asks_for_text_instead_of_inventing_fonts(monkeypatch):
    import api.font_match as matching
    monkeypatch.setattr(matching, "read_text", lambda _: ("?", 0.2))
    result = identify(specimen("tinos"))
    assert result["needs_text"] is True and result["matches"] == []


def test_image_to_api_to_actual_preview_flow():
    with TestClient(app) as client:
        response = client.post("/identify?text=Visual%20craft", content=png(specimen("tinos")), headers={"content-type": "image/png"})
        assert response.status_code == 200
        result = response.json()
        preview = client.get(result["matches"][0]["preview"])
        invalid = client.get("/font-preview/unknown")
    assert result["category"] == "serif"
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    with Image.open(BytesIO(preview.content)) as rendered:
        assert rendered.size == (940, 160)
    assert invalid.status_code == 404


def test_health_uses_small_runtime_without_importing_training_frameworks():
    # Other historical tests import PyTorch. Check isolation in a fresh process.
    result = subprocess.run([sys.executable, "-c", "from api.index import health_check; import sys; assert health_check()['status'] == 'ok'; assert 'torch' not in sys.modules and 'gradio' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("text", ["x" * 81, "non-Latin: 字"])
def test_invalid_manual_text_is_rejected(text):
    with pytest.raises(ValueError):
        identify(specimen("tinos"), text)
