from PIL import Image

from fontsense.features import extract_hog, prepare_grayscale


def test_preprocessing_shape_and_hog_stability():
    image = Image.new("RGB", (400, 150), "white")
    array = prepare_grayscale(image)
    assert array.shape == (96, 224)
    first = extract_hog(image)
    second = extract_hog(image)
    assert first.shape == second.shape
    assert (first == second).all()
