from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from PIL import Image

from fontsense.inference import FontSensePredictor


FINAL_MODEL_LABEL = "Final CNN"
HOG_MODEL_LABEL = "HOG comparison"
FROZEN_THRESHOLD = 0.60
FINAL_CLASS_ORDER = (
    "display",
    "handwriting",
    "monospace",
    "sans_serif",
    "serif",
)
FINAL_PREPROCESSING = {
    "image_size": [112, 48],
    "grayscale": True,
    "normalize_mean": [0.5],
    "normalize_std": [0.5],
}
FREEZE_PATH = ROOT / "reports" / "final_evaluation" / "pre_test_freeze.json"
HOG_ARTIFACT_DIR = ROOT / "artifacts" / "baseline"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_contract() -> dict:
    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        freeze = json.load(handle)

    checkpoint_path = (
        ROOT / freeze["selected_model"]["checkpoint"]
    ).resolve()
    if ROOT.resolve() not in checkpoint_path.parents:
        raise RuntimeError("The frozen checkpoint path leaves the project folder.")
    expected_hash = freeze["selected_model"]["checkpoint_sha256"].lower()
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "The final CNN checkpoint hash does not match the frozen evaluation."
        )
    if float(
        freeze["uncertainty_threshold"]["selected_threshold"]
    ) != FROZEN_THRESHOLD:
        raise RuntimeError("The frozen confidence threshold changed.")
    if tuple(freeze["class_order"]) != FINAL_CLASS_ORDER:
        raise RuntimeError("The frozen CNN class order changed.")
    if freeze["preprocessing"] != FINAL_PREPROCESSING:
        raise RuntimeError("The frozen CNN preprocessing changed.")

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": actual_hash,
        "threshold": FROZEN_THRESHOLD,
        "class_order": list(FINAL_CLASS_ORDER),
        "preprocessing": dict(FINAL_PREPROCESSING),
    }


def load_final_cnn_predictor(contract: dict) -> FontSensePredictor:
    predictor = FontSensePredictor(
        artifact_dir=contract["checkpoint_path"].parent,
        model="cnn",
        threshold=contract["threshold"],
    )
    if predictor.classes != contract["class_order"]:
        raise RuntimeError("The loaded CNN class order differs from the freeze.")
    if predictor.preprocessing != contract["preprocessing"]:
        raise RuntimeError("The loaded CNN preprocessing differs from the freeze.")
    return predictor


FROZEN_CONTRACT = load_frozen_contract()
FINAL_CHECKPOINT_PATH = FROZEN_CONTRACT["checkpoint_path"]
FINAL_CHECKPOINT_SHA256 = FROZEN_CONTRACT["checkpoint_sha256"]
FINAL_CNN_PREDICTOR = load_final_cnn_predictor(FROZEN_CONTRACT)

HOG_AVAILABLE = all(
    (HOG_ARTIFACT_DIR / filename).exists()
    for filename in (
        "hog_pipeline.joblib",
        "label_encoder.joblib",
        "hog_metadata.json",
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
    probabilities = {
        display_category(category): float(
            prediction["probabilities"][category]
        )
        for category in FINAL_CLASS_ORDER
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

> **FontSense predicts broad font categories and does not identify the exact
> font family.**
"""
    )
    with gr.Row():
        image = gr.Image(
            type="pil",
            image_mode="RGB",
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


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
