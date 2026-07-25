from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps

from .cnn_model import FontSenseCNN
from .features import extract_hog
from .utils import load_json


class FontSensePredictor:
    def __init__(self, artifact_dir: str | Path = "artifacts", model: str = "hog", threshold: float = 0.55):
        self.artifact_dir = Path(artifact_dir)
        self.model_type = model
        self.threshold = float(threshold)
        if model == "hog":
            self.pipeline = joblib.load(self.artifact_dir / "hog_pipeline.joblib")
            self.encoder = joblib.load(self.artifact_dir / "label_encoder.joblib")
            self.metadata = load_json(self.artifact_dir / "hog_metadata.json")
        elif model == "cnn":
            import torch
            checkpoint = torch.load(self.artifact_dir / "cnn_model.pt", map_location="cpu", weights_only=False)
            architecture = checkpoint["architecture"]
            self.classes = checkpoint["classes"]
            self.pipeline = FontSenseCNN(len(self.classes), width=architecture["width"], dropout=architecture["dropout"])
            self.pipeline.load_state_dict(checkpoint["state_dict"])
            self.pipeline.eval()
        else:
            raise ValueError("model must be 'hog' or 'cnn'")

    @staticmethod
    def validate_image(image: Image.Image) -> None:
        if image is None:
            raise ValueError("No image was provided.")
        width, height = image.size
        if width < 20 or height < 20:
            raise ValueError("The image is too small. Use a crop of at least 20×20 pixels.")
        if width > 6000 or height > 6000:
            raise ValueError("The image is too large. Resize it before prediction.")

    def predict(self, image: Image.Image) -> dict:
        self.validate_image(image)
        started = time.perf_counter()
        if self.model_type == "hog":
            config = self.metadata["hog"]
            vector = extract_hog(
                image,
                size=tuple(config.get("image_size", [112, 48])),
                orientations=int(config["orientations"]),
                pixels_per_cell=tuple(config["pixels_per_cell"]),
                cells_per_block=tuple(config["cells_per_block"]),
            )[None, :]
            probabilities = self.pipeline.predict_proba(vector)[0]
            classes = self.encoder.classes_.tolist()
        else:
            import torch
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((96, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
            tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(self.pipeline(tensor), dim=1)[0].numpy()
            classes = self.classes

        order = np.argsort(probabilities)[::-1]
        best_index = int(order[0])
        confidence = float(probabilities[best_index])
        return {
            "predicted_category": classes[best_index],
            "confidence": confidence,
            "uncertain": confidence < self.threshold,
            "warning": "Low confidence — treat this as an uncertain first guess." if confidence < self.threshold else "",
            "probabilities": {classes[i]: float(probabilities[i]) for i in order},
            "inference_ms": (time.perf_counter() - started) * 1000,
            "model": self.model_type,
        }
