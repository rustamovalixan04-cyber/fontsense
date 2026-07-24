from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from PIL import Image

from fontsense.inference import FontSensePredictor

FINAL_ARTIFACT_DIR = ROOT / "artifacts"
PROOF_ARTIFACT_DIR = FINAL_ARTIFACT_DIR / "proof"
ARTIFACT_DIR = FINAL_ARTIFACT_DIR if (FINAL_ARTIFACT_DIR / "hog_pipeline.joblib").exists() or (FINAL_ARTIFACT_DIR / "cnn_model.pt").exists() else PROOF_ARTIFACT_DIR
AVAILABLE = []
if (ARTIFACT_DIR / "hog_pipeline.joblib").exists():
    AVAILABLE.append("hog")
if (ARTIFACT_DIR / "cnn_model.pt").exists():
    AVAILABLE.append("cnn")

_cache = {}


def classify(image: Image.Image, model_name: str, threshold: float):
    if image is None:
        return "No image provided", {}, "Upload a PNG or JPEG text crop."
    if model_name not in AVAILABLE:
        return "Model unavailable", {}, f"Run the training pipeline to create the {model_name} artifact."
    try:
        key = (model_name, round(float(threshold), 2))
        predictor = _cache.setdefault(key, FontSensePredictor(ARTIFACT_DIR, model_name, threshold))
        result = predictor.predict(image)
        title = result["predicted_category"].replace("_", " ").title()
        status = f"{title} — {result['confidence']:.1%} confidence"
        note = result["warning"] or f"Inference time: {result['inference_ms']:.1f} ms"
        return status, result["probabilities"], note
    except Exception as exc:
        return "Could not classify image", {}, str(exc)


with gr.Blocks(title="FontSense") as demo:
    gr.Markdown("# FontSense\nUpload a cropped Latin-script text image. The result is a broad typeface category, not an exact font match.")
    with gr.Row():
        image = gr.Image(type="pil", label="Text image")
        with gr.Column():
            model = gr.Dropdown(choices=AVAILABLE or ["hog"], value=(AVAILABLE[0] if AVAILABLE else "hog"), label="Model")
            threshold = gr.Slider(0.30, 0.90, value=0.55, step=0.05, label="Uncertainty threshold")
            button = gr.Button("Classify", variant="primary")
    result_text = gr.Textbox(label="Prediction")
    probabilities = gr.Label(num_top_classes=5, label="Class probabilities")
    warning = gr.Textbox(label="Status / warning")
    button.click(classify, inputs=[image, model, threshold], outputs=[result_text, probabilities, warning])

if __name__ == "__main__":
    demo.launch()
