from pathlib import Path

from matplotlib import get_data_path

from fontsense.generate_dataset import render_text_image


def test_render_has_expected_size():
    path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    image = render_text_image("FontSense", path, seed=1)
    assert image.size == (224, 96)
