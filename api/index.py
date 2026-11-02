"""Stateless Vercel entry point for the frozen FontSense CNN."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from io import BytesIO
import logging
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool
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
    """Load the compact frozen-weight runtime without Gradio or PyTorch."""
    from api import compact as runtime

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
            if uploaded.format not in {"PNG", "JPEG"}:
                raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported.")
            width, height = uploaded.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="The image dimensions are too large.",
                )
            uploaded.load()
            oriented = ImageOps.exif_transpose(uploaded).convert("RGBA")
            background = Image.new("RGBA", oriented.size, "white")
            background.alpha_composite(oriented)
            image = background.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The file is not a readable PNG or JPEG image.",
        ) from exc

    return image


async def read_image(request: Request) -> Image.Image:
    """Bound streamed bodies as well as uploads with a Content-Length header."""
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported.")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            raise HTTPException(status_code=400, detail="The upload size is invalid.") from None
        if size < 0:
            raise HTTPException(status_code=400, detail="The upload size is invalid.")
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The image is larger than the 4 MB upload limit.")
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The image is larger than the 4 MB upload limit.")
        payload.extend(chunk)
    return decode_image(bytes(payload))


@app.post("/identify", include_in_schema=False)
async def identify_fonts(request: Request, text: str = "") -> dict:
    """Return image-specific font alternatives and a broad category."""
    if len(text) > 80:
        raise HTTPException(status_code=400, detail="Use up to 80 visible characters.")
    image = await read_image(request)
    from api import font_match
    try:
        return await run_in_threadpool(font_match.identify, image, text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        LOGGER.exception("Font matching failed")
        raise HTTPException(status_code=503, detail="Font matching is unavailable. Please try again.") from None


@app.get("/font-preview/{font_id}", include_in_schema=False)
def font_preview(font_id: str, text: str = "Typography", weight: float | None = None, width: float | None = None, optical_size: float | None = None) -> Response:
    from api.font_match import preview_png
    try:
        return Response(preview_png(font_id, text, weight, width, optical_size), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except ValueError:
        raise HTTPException(status_code=404, detail="Font preview not found.") from None


@app.get("/analysis/status", include_in_schema=False)
def analysis_status() -> dict[str, bool]:
    """Expose feature availability only, never credentials or configuration."""
    from api import analysis as typography
    return {"enabled": typography.configuration() is not None}


@app.post("/analyze", include_in_schema=False)
async def analyze(request: Request) -> dict:
    """Analyze typography on demand, independently of classifier results."""
    from api import analysis as typography
    settings = typography.configuration()
    if settings is None:
        raise HTTPException(status_code=503, detail="Typography analysis is not enabled. Category prediction is still available.")
    if not typography.authorize(request.headers.get("authorization", ""), settings):
        raise HTTPException(status_code=401, detail="Enter a valid private access code.", headers={"WWW-Authenticate": "Bearer"})
    image = await read_image(request)
    try:
        return await asyncio.wait_for(typography.analyze_image(image, settings), timeout=45)
    except typography.AnalysisLimitReached:
        raise HTTPException(status_code=429, detail="The analysis usage limit has been reached. Try again later.", headers={"Retry-After": "60"}) from None
    except typography.AnalysisUnavailable:
        raise HTTPException(status_code=503, detail="Typography analysis is temporarily unavailable. Category prediction is still available.") from None
    except (typography.AnalysisFailed, TimeoutError):
        raise HTTPException(status_code=502, detail="Typography analysis could not finish. Category prediction is still available.") from None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage() -> HTMLResponse:
    """Serve the lightweight browser interface without loading PyTorch."""
    return HTMLResponse(INDEX_HTML)


@app.get("/healthz", include_in_schema=False)
def health_check() -> dict[str, str]:
    """Confirm that the web process and frozen checkpoint load correctly."""
    runtime = get_runtime()
    runtime.get_predictor(runtime.FINAL_MODEL_LABEL)
    return {
        "status": "ok",
        "checkpoint_sha256": runtime.FINAL_CHECKPOINT_SHA256,
    }


@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, object]:
    """Classify image bytes in the same request that receives them."""
    image = await read_image(request)
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
