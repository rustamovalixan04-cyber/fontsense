from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay

from .inference import FontSensePredictor
from .split import assert_no_family_leakage
from .utils import project_root, save_json


def evaluate(manifest_path: str | Path, artifact_dir: str | Path, report_dir: str | Path, model: str, threshold: float = 0.55) -> dict:
    manifest = pd.read_csv(manifest_path)
    assert_no_family_leakage(manifest)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    predictor = FontSensePredictor(artifact_dir, model=model, threshold=threshold)
    rows: list[dict] = []
    for _, row in test.iterrows():
        with Image.open(row["image_path"]) as image:
            result = predictor.predict(image.convert("RGB"))
        rows.append({
            **row.to_dict(),
            "predicted_category": result["predicted_category"],
            "confidence": result["confidence"],
            "uncertain": result["uncertain"],
            "inference_ms": result["inference_ms"],
        })
    predictions = pd.DataFrame(rows)
    labels = sorted(manifest["category"].unique().tolist())
    y_true = predictions["category"]
    y_pred = predictions["predicted_category"]
    metrics = {
        "model": model,
        "test_images": int(len(test)),
        "test_families": int(test["family"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_inference_ms": float(predictions["inference_ms"].mean()),
        "uncertain_rate": float(predictions["uncertain"].mean()),
    }
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)).T
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(report_dir / f"{model}_test_predictions.csv", index=False)
    predictions[predictions["category"] != predictions["predicted_category"]].sort_values("confidence", ascending=False).to_csv(report_dir / f"{model}_error_examples.csv", index=False)
    report.to_csv(report_dir / f"{model}_classification_report.csv")
    save_json(metrics, report_dir / f"{model}_test_metrics.json")

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title(f"FontSense {model.upper()} — held-out family test set")
    fig.tight_layout()
    fig.savefig(report_dir / "figures" / f"{model}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FontSense model on untouched test families.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-dir", default=str(project_root() / "artifacts"))
    parser.add_argument("--report-dir", default=str(project_root() / "reports"))
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()
    metrics = evaluate(args.manifest, args.artifact_dir, args.report_dir, args.model, args.threshold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
