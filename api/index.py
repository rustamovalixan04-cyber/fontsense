"""Stateless Vercel entry point for the frozen FontSense CNN."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import logging
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError


LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent / "static"
INDEX_HTML = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(title="FontSense", docs_url=None, redoc_url=None)


@lru_cache(maxsize=1)
def get_runtime() -> ModuleType:
    """Load the frozen model once, when an inference process first needs it."""
    import app as runtime

    return runtime


def decode_image(payload: bytes) -> Image.Image:
    """Validate uploaded bytes and return a safely oriented RGB image."""
    if not payload:
        raise HTTPException(status_code=400, detail="Choose a PNG or JPEG image.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The image is larger than the 4 MB upload limit.",
        )

    try:
        with Image.open(BytesIO(payload)) as uploaded:
            width, height = uploaded.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="The image dimensions are too large.",
                )
            uploaded.load()
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The file is not a readable PNG or JPEG image.",
        ) from exc

    return image


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage() -> HTMLResponse:
    """Serve the lightweight browser interface without loading PyTorch."""
    return HTMLResponse(INDEX_HTML)


@app.get("/healthz", include_in_schema=False)
def health_check() -> dict[str, str]:
    """Confirm that the web process and frozen checkpoint load correctly."""
    runtime = get_runtime()
    return {
        "status": "ok",
        "checkpoint_sha256": runtime.FINAL_CHECKPOINT_SHA256,
    }


@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, object]:
    """Classify image bytes in the same request that receives them."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported.",
        )

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="The image is larger than the 4 MB upload limit.",
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="The upload size is invalid.",
            ) from None

    image = decode_image(await request.body())
    runtime = get_runtime()

    try:
        prediction = runtime.get_predictor(
            runtime.FINAL_MODEL_LABEL
        ).predict(image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("FontSense inference failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "FontSense could not process this image. "
                "Try another clear PNG or JPEG text crop."
            ),
        ) from exc

    confidence = float(prediction["confidence"])
    probabilities = {
        category: float(prediction["probabilities"][category])
        for category in runtime.FINAL_CLASS_ORDER
    }
    return {
        "predicted_category": prediction["predicted_category"],
        "confidence": confidence,
        "probabilities": probabilities,
        "accepted": confidence >= runtime.FROZEN_THRESHOLD,
        "threshold": runtime.FROZEN_THRESHOLD,
        "class_order": list(runtime.FINAL_CLASS_ORDER),
    }
