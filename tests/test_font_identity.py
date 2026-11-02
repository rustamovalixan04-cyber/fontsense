from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageOps
import pytest

from api.font_identity import recognize
from api.font_match import identify, render_reference
from api.index import app
from api.name_encoder import encoder, forward_features, ROOT

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("filename", ["impact_typography.png", "impact_screenshot_crop.png"])
def test_impact_is_named_separately_from_anton_alternatives(filename):
    with Image.open(FIXTURES / filename) as image:
        result = identify(image.convert("RGB"))
    assert result["identified_font"]["family"] == "Impact"
    assert result["category"] == "display"
    assert "Anton" in {font["family"] for font in result["matches"]}
    assert all(font["family"] != "Impact" for font in result["matches"])


def test_impact_name_survives_dark_background_and_jpeg():
    image = ImageOps.invert(Image.open(FIXTURES / "impact_typography.png").convert("RGB"))
    output = BytesIO(); image.save(output, format="JPEG", quality=60)
    with TestClient(app) as client:
        response = client.post("/identify", content=output.getvalue(), headers={"content-type": "image/jpeg"})
    assert response.status_code == 200
    assert response.json()["identified_font"]["family"] == "Impact"


@pytest.mark.parametrize("font_id", ["anton", "bebasneue", "alfaslabone", "arimo"])
def test_bold_lookalikes_are_not_automatically_called_impact(font_id):
    image = render_reference(font_id, "Typography")
    result = recognize(image, "Typography")
    assert result is None or result["family"] != "Impact"


def test_short_text_does_not_force_a_font_name():
    assert recognize(render_reference("anton", "II"), "II") is None


def test_name_encoder_matches_pytorch_export_vectors():
    with np.load(ROOT / "font_name_encoder.npz", allow_pickle=False) as data:
        pixels, expected = data["parity_pixels"], data["parity_vectors"]
    runtime = encoder()
    for patch, target in zip(pixels, expected):
        actual = runtime.projection @ forward_features(patch, runtime.blocks)
        actual /= np.linalg.norm(actual)
        np.testing.assert_allclose(actual, target, atol=3e-5, rtol=3e-5)
