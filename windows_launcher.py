from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


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


def launch_application(*, inbrowser: bool, port: int | None = None) -> None:
    """Start the local-only Gradio application."""
    app = load_application()
    app.build_demo().launch(
        css=app.APP_CSS,
        inbrowser=inbrowser,
        server_name="127.0.0.1",
        server_port=port,
        share=False,
    )


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
        failure = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
