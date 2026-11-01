"""Numerical parity with original frozen weights, using fresh synthetic inputs."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import pytest

from api.compact import CLASSES, get_model
from api.font_match import render_reference


def test_compact_probabilities_match_original_pytorch_model():
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from fontsense.cnn_model import FontSenseCNN
    checkpoint = torch.load(root / "artifacts/cnn/cnn_model.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    original = FontSenseCNN(width=state["features.0.weight"].shape[0])
    original.load_state_dict(state)
    original.eval()
    torch.set_num_threads(1)
    rng = np.random.default_rng(1729)
    images = [Image.fromarray(rng.integers(0, 256, (150, 350), dtype=np.uint8)).convert("RGB")]
    images.extend(ImageOps.expand(render_reference(font, "Fresh words 42"), 60, "white") for font in ("tinos", "arimo", "cousine", "anton", "caveat"))
    for image in images:
        gray = image.convert("L").resize((112, 48), Image.Resampling.BILINEAR)
        pixels = (np.asarray(gray, dtype=np.float32) / 255 - .5) / .5
        with torch.inference_mode():
            expected = original(torch.from_numpy(pixels[None, None])).softmax(1).numpy()[0]
        actual = get_model().predict(image)["probabilities"]
        np.testing.assert_allclose([actual[category] for category in CLASSES], expected, atol=2e-6, rtol=1e-5)
