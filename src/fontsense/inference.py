from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .utils import load_json


def build_cnn_inference_transform(image_size: tuple[int, int]):
    """Build the frozen evaluation transform without importing training code."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    width, height = image_size
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                (height, width),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_cnn_for_inference(checkpoint_path: str | Path):
    """Load the saved CNN on CPU using only inference dependencies."""
    import torch

    from .cnn_model import FontSenseCNN

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    architecture = checkpoint["architecture"]
    model = FontSenseCNN(
        num_classes=len(checkpoint["classes"]),
        width=int(architecture["width"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


class FontSensePredictor:
    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        model: str = "hog",
        threshold: float = 0.55,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.model_type = model
        self.threshold = float(threshold)
        self.transform = None
        self.preprocessing = None

        if model == "hog":
            import joblib

            self.pipeline = joblib.load(
                self.artifact_dir / "hog_pipeline.joblib"
            )
            self.encoder = joblib.load(
                self.artifact_dir / "label_encoder.joblib"
            )
            self.metadata = load_json(
                self.artifact_dir / "hog_metadata.json"
            )
            self.classes = self.encoder.classes_.tolist()
        elif model == "cnn":
            self.pipeline, self.checkpoint = load_cnn_for_inference(
                self.artifact_dir / "cnn_model.pt"
            )
            self.classes = list(self.checkpoint["classes"])
            self.preprocessing = dict(self.checkpoint["preprocessing"])
            if self.preprocessing.get("grayscale") is not True:
                raise ValueError("The CNN checkpoint must use grayscale input.")
            if self.preprocessing.get("normalize_mean") != [0.5]:
                raise ValueError(
                    "The CNN checkpoint normalization mean changed."
                )
            if self.preprocessing.get("normalize_std") != [0.5]:
                raise ValueError(
                    "The CNN checkpoint normalization standard deviation changed."
                )
            image_size = tuple(
                int(value) for value in self.preprocessing["image_size"]
            )
            self.transform = build_cnn_inference_transform(
                image_size
            )
        else:
            raise ValueError("model must be 'hog' or 'cnn'")

    @staticmethod
    def prepare_image(image: Image.Image) -> Image.Image:
        if image is None:
            raise ValueError("No image was provided.")
        if not isinstance(image, Image.Image):
            raise ValueError("Unsupported input. Upload a PNG or JPEG image.")
        try:
            image.load()
            prepared = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, SyntaxError, ValueError) as exc:
            raise ValueError(
                "The image could not be read. Upload a valid PNG or JPEG file."
            ) from exc

        width, height = prepared.size
        if width < 20 or height < 20:
            raise ValueError(
                "The image is too small. Use a crop of at least 20×20 pixels."
            )
        if width > 6000 or height > 6000:
            raise ValueError("The image is too large. Resize it before prediction.")

        grayscale = np.asarray(prepared.convert("L"), dtype=np.float32)
        standard_deviation = float(grayscale.std())
        lower, upper = np.percentile(grayscale, [1, 99])
        if standard_deviation < 2.0 or (
            float(upper - lower) < 5.0 and standard_deviation < 5.0
        ):
            raise ValueError(
                "The image appears blank or nearly blank. "
                "Upload a readable text crop."
            )
        return prepared

    def preprocess(self, image: Image.Image):
        if self.model_type != "cnn" or self.transform is None:
            raise ValueError(
                "Preprocessing tensors are available only for the CNN."
            )
        return self.transform(self.prepare_image(image))

    def predict(self, image: Image.Image) -> dict:
        prepared = self.prepare_image(image)
        started = time.perf_counter()
        if self.model_type == "hog":
            from .features import extract_hog

            config = self.metadata["hog"]
            vector = extract_hog(
                prepared,
                size=tuple(config.get("image_size", [112, 48])),
                orientations=int(config["orientations"]),
                pixels_per_cell=tuple(config["pixels_per_cell"]),
                cells_per_block=tuple(config["cells_per_block"]),
            )[None, :]
            probabilities = self.pipeline.predict_proba(vector)[0]
        else:
            import torch

            tensor = self.transform(prepared).unsqueeze(0)
            with torch.no_grad():
                probabilities = (
                    torch.softmax(self.pipeline(tensor), dim=1)[0]
                    .cpu()
                    .numpy()
                )

        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        uncertain = confidence < self.threshold
        return {
            "predicted_category": self.classes[best_index],
            "confidence": confidence,
            "accepted": not uncertain,
            "uncertain": uncertain,
            "warning": (
                "Low confidence — treat this as an uncertain first guess."
                if uncertain
                else ""
            ),
            "probabilities": {
                self.classes[index]: float(probabilities[index])
                for index in range(len(self.classes))
            },
            "inference_ms": (time.perf_counter() - started) * 1000,
            "model": self.model_type,
        }
