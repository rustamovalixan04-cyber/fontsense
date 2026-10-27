from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent


def get_runtime_root() -> Path:
    """Return the source root or PyInstaller's bundled resource directory."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and bundle_root:
        return Path(bundle_root).resolve()
    return SOURCE_ROOT.resolve()


ROOT = get_runtime_root()
sys.path.insert(0, str(SOURCE_ROOT / "src"))

import gradio as gr
from PIL import Image

from fontsense.inference import FontSensePredictor, validate_probabilities
from fontsense.runtime import (
    FINAL_MODEL_LABEL,
    FROZEN_THRESHOLD,
    FINAL_CLASS_ORDER,
    FINAL_PREPROCESSING,
    FREEZE_PATH,
    sha256_file,
    load_frozen_contract,
    load_final_cnn_predictor,
    FROZEN_CONTRACT,
    FINAL_CHECKPOINT_PATH,
    FINAL_CHECKPOINT_SHA256,
    FINAL_CNN_PREDICTOR,
)


HOG_MODEL_LABEL = "HOG comparison"
HOG_ARTIFACT_DIR = ROOT / "artifacts" / "baseline"

HOG_AVAILABLE = (
    all(
        importlib.util.find_spec(package) is not None
        for package in ("joblib", "sklearn", "skimage", "tqdm")
    )
    and all(
        (HOG_ARTIFACT_DIR / filename).exists()
        for filename in (
            "hog_pipeline.joblib",
            "label_encoder.joblib",
            "hog_metadata.json",
        )
    )
)
MODEL_CHOICES = [FINAL_MODEL_LABEL]
if HOG_AVAILABLE:
    MODEL_CHOICES.append(HOG_MODEL_LABEL)


@lru_cache(maxsize=1)
def load_optional_hog_predictor() -> FontSensePredictor:
    if not HOG_AVAILABLE:
        raise ValueError("The optional HOG comparison model is unavailable.")
    return FontSensePredictor(
        artifact_dir=HOG_ARTIFACT_DIR,
        model="hog",
        threshold=FROZEN_THRESHOLD,
    )


def get_predictor(model_name: str) -> FontSensePredictor:
    if model_name == FINAL_MODEL_LABEL:
        return FINAL_CNN_PREDICTOR
    if model_name == HOG_MODEL_LABEL:
        return load_optional_hog_predictor()
    raise ValueError("Choose an available FontSense model.")


def display_category(category: str) -> str:
    return category.replace("_", " ").capitalize()


def format_prediction(
    prediction: dict,
    threshold: float = FROZEN_THRESHOLD,
) -> tuple[str, dict[str, float], str]:
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("The uncertainty threshold must be between 0 and 1.")
    values = validate_probabilities(
        [prediction["probabilities"][category] for category in FINAL_CLASS_ORDER],
        len(FINAL_CLASS_ORDER),
    )
    probabilities = {
        display_category(category): float(values[index])
        for index, category in enumerate(FINAL_CLASS_ORDER)
    }
    total_probability = sum(probabilities.values())
    if not all(math.isfinite(value) for value in probabilities.values()):
        raise RuntimeError("The model returned an invalid probability.")
    if not math.isclose(total_probability, 1.0, abs_tol=1e-4):
        raise RuntimeError("The model probabilities do not sum to one.")

    confidence = float(prediction["confidence"])
    category = display_category(prediction["predicted_category"])
    accepted = confidence >= float(threshold)
    if accepted:
        result = (
            f"### Predicted category: {category}\n\n"
            f"**Confidence: {confidence:.1%}**"
        )
        status = (
            "✅ **Accepted** — confidence meets the frozen "
            f"{threshold:.0%} threshold."
        )
    else:
        result = (
            f"### Possible category: {category}\n\n"
            f"**Confidence: {confidence:.1%}**\n\n"
            "**Low confidence — treat this as an uncertain first guess.**"
        )
        status = (
            "⚠️ **Uncertain** — confidence is below the frozen "
            f"{threshold:.0%} threshold."
        )
    return result, probabilities, status


def classify(
    image: Image.Image | None,
    model_name: str = FINAL_MODEL_LABEL,
) -> tuple[str, dict[str, float], str]:
    if image is None:
        return (
            "### No image uploaded",
            {},
            "Upload a PNG or JPEG text crop, then select **Predict**.",
        )
    try:
        prediction = get_predictor(model_name).predict(image)
        return format_prediction(prediction)
    except ValueError as exc:
        return "### Could not classify image", {}, f"⚠️ {exc}"
    except Exception:
        return (
            "### Inference error",
            {},
            "⚠️ FontSense could not process this image. "
            "Try another valid PNG or JPEG text crop.",
        )


def reset_app():
    return (
        None,
        "### No prediction yet",
        {},
        "Upload a readable text crop to begin.",
    )


APP_CSS = """
.gradio-container { max-width: 980px !important; }
#fontsense-result { min-height: 108px; padding: 14px 16px; border-radius: 10px; }
#fontsense-status { min-height: 54px; }
"""


with gr.Blocks(title="FontSense") as demo:
    gr.Markdown(
        """
# FontSense

Upload a cropped Latin-script text image to estimate its broad font category.

For best results, crop one clear line of text. Avoid full pages and mixed fonts.

> **FontSense predicts broad font categories and does not identify the exact
> font family.**
"""
    )
    with gr.Row():
        image = gr.Image(
            type="pil",
            image_mode="RGBA",
            sources=["upload"],
            label="Upload a PNG or JPEG text crop",
            placeholder="Choose a clear image containing readable text",
            height=330,
        )
        with gr.Column():
            model = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=FINAL_MODEL_LABEL,
                label="Prediction model",
                visible=len(MODEL_CHOICES) > 1,
                info=(
                    "The frozen final CNN is the default. "
                    "HOG is an optional baseline comparison."
                ),
            )
            gr.Markdown(
                """
**Confidence rule:** predictions at or above **60%** are accepted.
Lower-confidence predictions are marked uncertain.
"""
            )
            with gr.Row():
                predict_button = gr.Button("Predict", variant="primary")
                reset_button = gr.Button("Reset")

    gr.Markdown("## Result")
    result_text = gr.Markdown(
        "### No prediction yet",
        elem_id="fontsense-result",
    )
    probabilities = gr.Label(
        num_top_classes=5,
        label="Probabilities for all five categories",
    )
    status_text = gr.Markdown(
        "Upload a readable text crop to begin.",
        elem_id="fontsense-status",
    )
    gr.Markdown(
        """
### Understanding uncertainty

An **uncertain** result is still the model's first guess, but its confidence is
below the frozen 60% validation-selected threshold. Do not treat it as a
reliable exact identification.
Even a confident prediction can be wrong on unfamiliar fonts and photographs.
"""
    )

    predict_button.click(
        classify,
        inputs=[image, model],
        outputs=[result_text, probabilities, status_text],
        api_name="predict",
    )
    reset_button.click(
        reset_app,
        outputs=[image, result_text, probabilities, status_text],
        api_name="reset",
    )


def build_demo() -> gr.Blocks:
    """Return the single initialized interface for scripts and notebooks."""
    return demo


if __name__ == "__main__":
    build_demo().launch(css=APP_CSS)
