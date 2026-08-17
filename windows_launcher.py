from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import socket
import sys

from PIL import Image


LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PORT_SEARCH_LIMIT = 100


def load_application():
    """Import the application only after CLI arguments have been parsed."""
    import app

    return app


def frozen_runtime_report() -> dict:
    """Verify the exact assessed CNN contract used by the application."""
    app = load_application()
    contract = app.load_frozen_contract()
    predictor = app.FINAL_CNN_PREDICTOR

    checks = {
        "checkpoint_sha256": (
            contract["checkpoint_sha256"] == app.FINAL_CHECKPOINT_SHA256
        ),
        "class_order": predictor.classes == contract["class_order"],
        "preprocessing": predictor.preprocessing == contract["preprocessing"],
        "threshold": predictor.threshold == contract["threshold"] == 0.60,
        "cnn_loaded": (
            predictor.model_type == "cnn"
            and predictor.pipeline.training is False
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checkpoint_path": str(contract["checkpoint_path"]),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "class_order": contract["class_order"],
        "preprocessing": contract["preprocessing"],
        "threshold": contract["threshold"],
    }


def predict_json(image_path: str | Path) -> dict:
    """Run one image through the frozen CNN and return stable JSON fields."""
    app = load_application()
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        prediction = app.FINAL_CNN_PREDICTOR.predict(image)

    probabilities = {
        category: float(prediction["probabilities"][category])
        for category in app.FINAL_CLASS_ORDER
    }
    return {
        "predicted_category": prediction["predicted_category"],
        "confidence": float(prediction["confidence"]),
        "accepted": bool(prediction["accepted"]),
        "uncertain": bool(prediction["uncertain"]),
        "probabilities": probabilities,
        "class_order": list(app.FINAL_CLASS_ORDER),
        "threshold": app.FROZEN_THRESHOLD,
    }


def find_available_port() -> int:
    """Choose the first free local Gradio port for a normal launch."""
    for port in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SEARCH_LIMIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind((LOCAL_HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free local port was found for FontSense.")


def launch_application(*, inbrowser: bool, port: int | None = None) -> None:
    """Start the local-only Gradio application."""
    print("FontSense is starting...", flush=True)
    print("Loading the final CNN...", flush=True)
    app = load_application()
    selected_port = port if port is not None else find_available_port()
    local_url = f"http://{LOCAL_HOST}:{selected_port}"
    if inbrowser:
        print("Opening FontSense in your browser...", flush=True)
    print("If the browser does not open, keep this window open.", flush=True)
    print("Open this address in your browser:", flush=True)
    print(local_url, flush=True)
    app.build_demo().launch(
        css=app.APP_CSS,
        inbrowser=inbrowser,
        server_name=LOCAL_HOST,
        server_port=selected_port,
        share=False,
    )


def show_fatal_startup_error(message: str) -> None:
    """Show a lightweight Windows dialog when a frozen launch cannot start."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return
    try:
        ctypes.windll.user32.MessageBoxW(
            None,
            message,
            "FontSense startup error",
            0x10,
        )
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch or verify the frozen FontSense CNN application."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="verify the frozen runtime and exit",
    )
    mode.add_argument(
        "--predict-json",
        metavar="IMAGE_PATH",
        help="predict one image and print machine-readable JSON",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the local server without opening a browser",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="optional local server port used for smoke testing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            report = frozen_runtime_report()
            print(json.dumps(report, sort_keys=True))
            return 0 if report["status"] == "PASS" else 1
        if args.predict_json:
            print(json.dumps(predict_json(args.predict_json), sort_keys=True))
            return 0
        launch_application(inbrowser=not args.no_browser, port=args.port)
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if args.self_test or args.predict_json:
            print(
                json.dumps({"status": "FAIL", "error": error}, sort_keys=True),
                file=sys.stderr,
            )
        else:
            message = (
                "FontSense could not start.\n\n"
                f"{error}\n\n"
                "Please keep the complete FontSense folder together and try again."
            )
            print(message, file=sys.stderr, flush=True)
            show_fatal_startup_error(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
