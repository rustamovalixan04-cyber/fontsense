"""Vercel ASGI entry point for the existing FontSense Gradio app."""

from fastapi import FastAPI
import gradio as gr

from app import APP_CSS, FINAL_CHECKPOINT_SHA256, build_demo


base_app = FastAPI(
    title="FontSense",
    docs_url=None,
    redoc_url=None,
)


@base_app.get("/healthz", include_in_schema=False)
def health_check() -> dict[str, str]:
    """Confirm that the web process and frozen checkpoint loaded."""
    return {
        "status": "ok",
        "checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
    }


app = gr.mount_gradio_app(
    base_app,
    build_demo(),
    path="/",
    css=APP_CSS,
    show_error=False,
    max_file_size="4mb",
)
