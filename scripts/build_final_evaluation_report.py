"""Build the canonical portable report for the final CNN evaluation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "final_evaluation"
ARTIFACT_PATH = REPORT_DIR / "final_evaluation_report_artifact.json"


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    generated_at: str,
    tables_used: list[str],
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": generated_at,
            "tables_used": tables_used,
        },
    }


def build_report_artifact() -> None:
    metrics = json.loads(
        (REPORT_DIR / "final_test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if metrics["status"] != "completed":
        raise RuntimeError("Final evaluation must be complete first")
    classification = pd.read_csv(
        REPORT_DIR / "final_classification_report.csv"
    )
    confusion = pd.read_csv(
        REPORT_DIR / "final_confusion_matrix.csv",
        index_col=0,
    )
    confidence = pd.read_csv(
        REPORT_DIR / "confidence_distribution.csv"
    )
    uncertainty = pd.read_csv(
        REPORT_DIR / "uncertainty_by_class.csv"
    )
    errors = pd.read_csv(
        REPORT_DIR / "errors_by_true_and_predicted.csv"
    )

    confusion_long = (
        confusion.rename_axis("true_category")
        .reset_index()
        .melt(
            id_vars="true_category",
            var_name="predicted_category",
            value_name="test_images",
        )
    )
    confidence_long = pd.concat(
        [
            confidence[
                ["bin_lower", "bin_upper", "correct_predictions"]
            ]
            .rename(columns={"correct_predictions": "predictions"})
            .assign(outcome="Correct"),
            confidence[
                ["bin_lower", "bin_upper", "incorrect_predictions"]
            ]
            .rename(columns={"incorrect_predictions": "predictions"})
            .assign(outcome="Incorrect"),
        ],
        ignore_index=True,
    )
    confidence_long["confidence_bin"] = confidence_long.apply(
        lambda row: f"{row['bin_lower']:.1f}–{row['bin_upper']:.1f}",
        axis=1,
    )

    headline = pd.DataFrame(
        [
            {
                "test_macro_f1": metrics["test_results"]["macro_f1"],
                "test_accuracy": metrics["test_results"]["accuracy"],
                "correct_predictions": metrics["test_results"][
                    "correct_predictions"
                ],
                "incorrect_predictions": metrics["test_results"][
                    "incorrect_predictions"
                ],
                "accepted_accuracy": metrics["uncertainty_results"][
                    "accepted_accuracy"
                ],
                "accepted_predictions": metrics["uncertainty_results"][
                    "accepted_predictions"
                ],
                "uncertain_predictions": metrics["uncertainty_results"][
                    "uncertain_predictions"
                ],
                "confidence_threshold": metrics[
                    "uncertainty_threshold"
                ]["value"],
                "inference_ms_per_image": metrics["test_results"][
                    "inference_ms_per_image"
                ],
                "model_size_bytes": metrics["test_results"][
                    "model_size_bytes"
                ],
            }
        ]
    )

    connection = sqlite3.connect(":memory:")
    headline.to_sql("final_headline", connection, index=False)
    classification.to_sql(
        "final_classification",
        connection,
        index=False,
    )
    confusion_long.to_sql(
        "final_confusion",
        connection,
        index=False,
    )
    confidence_long.to_sql(
        "final_confidence",
        connection,
        index=False,
    )
    uncertainty.to_sql(
        "final_uncertainty",
        connection,
        index=False,
    )
    errors.to_sql("final_errors", connection, index=False)

    headline_sql = """
        SELECT test_macro_f1, test_accuracy, correct_predictions,
               incorrect_predictions, accepted_accuracy,
               accepted_predictions, uncertain_predictions,
               confidence_threshold, inference_ms_per_image,
               model_size_bytes
        FROM final_headline
    """.strip()
    classification_sql = """
        SELECT class, precision, recall, f1, support
        FROM final_classification
        ORDER BY f1 DESC, class
    """.strip()
    confusion_sql = """
        SELECT true_category, predicted_category, test_images
        FROM final_confusion
        ORDER BY true_category, predicted_category
    """.strip()
    confidence_sql = """
        SELECT confidence_bin, bin_lower, bin_upper, outcome, predictions
        FROM final_confidence
        ORDER BY bin_lower, outcome
    """.strip()
    uncertainty_sql = """
        SELECT category, test_images, accepted_predictions,
               uncertain_predictions, coverage, accepted_correct,
               accepted_accuracy
        FROM final_uncertainty
        ORDER BY category
    """.strip()
    errors_sql = """
        SELECT category, predicted_category, errors
        FROM final_errors
        ORDER BY errors DESC, category, predicted_category
    """.strip()

    headline_rows = _records(
        pd.read_sql_query(headline_sql, connection)
    )
    classification_rows = _records(
        pd.read_sql_query(classification_sql, connection)
    )
    confusion_rows = _records(
        pd.read_sql_query(confusion_sql, connection)
    )
    confidence_rows = _records(
        pd.read_sql_query(confidence_sql, connection)
    )
    uncertainty_rows = _records(
        pd.read_sql_query(uncertainty_sql, connection)
    )
    error_rows = _records(
        pd.read_sql_query(errors_sql, connection)
    )
    connection.close()

    strongest = max(classification_rows, key=lambda row: row["f1"])
    weakest = min(classification_rows, key=lambda row: row["f1"])
    top_error = error_rows[0]
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    sources = [
        _source(
            "headline",
            "Final test headline metrics",
            "reports/final_evaluation/final_test_metrics.json",
            headline_sql,
            "Reads the immutable final test metrics and uncertainty summary.",
            generated_at,
            ["final_headline"],
        ),
        _source(
            "classification",
            "Final per-class classification metrics",
            "reports/final_evaluation/final_classification_report.csv",
            classification_sql,
            "Reads test precision, recall, F1, and support by category.",
            generated_at,
            ["final_classification"],
        ),
        _source(
            "confusion",
            "Final test confusion matrix",
            "reports/final_evaluation/final_confusion_matrix.csv",
            confusion_sql,
            "Reads every true and predicted category cell.",
            generated_at,
            ["final_confusion"],
        ),
        _source(
            "confidence",
            "Final confidence distribution",
            "reports/final_evaluation/confidence_distribution.csv",
            confidence_sql,
            "Reads binned correct and incorrect prediction confidence.",
            generated_at,
            ["final_confidence"],
        ),
        _source(
            "uncertainty",
            "Final uncertainty analysis by class",
            "reports/final_evaluation/uncertainty_by_class.csv",
            uncertainty_sql,
            "Reads accepted coverage and accepted accuracy by category.",
            generated_at,
            ["final_uncertainty"],
        ),
        _source(
            "errors",
            "Final errors by true and predicted category",
            "reports/final_evaluation/errors_by_true_and_predicted.csv",
            errors_sql,
            "Reads the count of each observed error direction.",
            generated_at,
            ["final_errors"],
        ),
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "FontSense final CNN evaluation",
        "description": (
            "Single held-out-family test evaluation of the selected CNN."
        ),
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "macro_f1_card",
                "description": (
                    "Primary metric across all five held-out categories."
                ),
                "dataset": "headline",
                "sourceId": "headline",
                "metrics": [
                    {
                        "label": "Test macro F1",
                        "field": "test_macro_f1",
                        "format": "percent",
                    }
                ],
            },
            {
                "id": "accuracy_card",
                "description": "Correct predictions among 600 test images.",
                "dataset": "headline",
                "sourceId": "headline",
                "metrics": [
                    {
                        "label": "Test accuracy",
                        "field": "test_accuracy",
                        "format": "percent",
                    }
                ],
            },
            {
                "id": "accepted_accuracy_card",
                "description": (
                    "Accuracy after applying the validation-only threshold."
                ),
                "dataset": "headline",
                "sourceId": "headline",
                "metrics": [
                    {
                        "label": "Accepted accuracy",
                        "field": "accepted_accuracy",
                        "format": "percent",
                    }
                ],
            },
            {
                "id": "uncertain_card",
                "description": (
                    "Predictions below the frozen 0.60 confidence threshold."
                ),
                "dataset": "headline",
                "sourceId": "headline",
                "metrics": [
                    {
                        "label": "Uncertain predictions",
                        "field": "uncertain_predictions",
                        "format": "number",
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "class_f1_chart",
                "title": "Final test F1 by category",
                "subtitle": (
                    "120 images from three held-out families per category."
                ),
                "type": "bar",
                "intent": "comparison",
                "dataset": "classification",
                "sourceId": "classification",
                "valueFormat": "percent",
                "encodings": {
                    "x": {
                        "field": "class",
                        "type": "nominal",
                        "label": "Category",
                    },
                    "y": {
                        "field": "f1",
                        "type": "quantitative",
                        "label": "Test F1",
                    },
                    "tooltip": [
                        {
                            "field": "precision",
                            "type": "quantitative",
                            "label": "Precision",
                        },
                        {
                            "field": "recall",
                            "type": "quantitative",
                            "label": "Recall",
                        },
                    ],
                },
            },
            {
                "id": "confidence_chart",
                "title": "Final test prediction confidence",
                "subtitle": (
                    "Correct and incorrect predictions in 0.10-wide bins."
                ),
                "type": "bar",
                "intent": "distribution",
                "dataset": "confidence",
                "sourceId": "confidence",
                "encodings": {
                    "x": {
                        "field": "confidence_bin",
                        "type": "nominal",
                        "label": "Maximum probability",
                    },
                    "y": {
                        "field": "predictions",
                        "type": "quantitative",
                        "label": "Predictions",
                    },
                    "color": {
                        "field": "outcome",
                        "type": "nominal",
                        "label": "Outcome",
                    },
                },
            },
            {
                "id": "error_chart",
                "title": "Final errors by true and predicted category",
                "subtitle": "Observed directions among 80 incorrect predictions.",
                "type": "bar",
                "intent": "ranking",
                "dataset": "errors",
                "sourceId": "errors",
                "encodings": {
                    "x": {
                        "field": "category",
                        "type": "nominal",
                        "label": "True category",
                    },
                    "y": {
                        "field": "errors",
                        "type": "quantitative",
                        "label": "Errors",
                    },
                    "color": {
                        "field": "predicted_category",
                        "type": "nominal",
                        "label": "Predicted category",
                    },
                },
            },
        ],
        "tables": [
            {
                "id": "confusion_table",
                "title": "Final test confusion matrix cells",
                "subtitle": "All 25 true-to-predicted category counts.",
                "dataset": "confusion",
                "sourceId": "confusion",
                "defaultSort": {
                    "field": "true_category",
                    "direction": "asc",
                },
                "columns": [
                    {
                        "field": "true_category",
                        "label": "True class",
                        "type": "text",
                    },
                    {
                        "field": "predicted_category",
                        "label": "Predicted class",
                        "type": "text",
                    },
                    {
                        "field": "test_images",
                        "label": "Images",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "classification_table",
                "title": "Final per-class test metrics",
                "subtitle": "Exact precision, recall, F1, and support.",
                "dataset": "classification",
                "sourceId": "classification",
                "defaultSort": {"field": "f1", "direction": "desc"},
                "columns": [
                    {"field": "class", "label": "Class", "type": "text"},
                    {
                        "field": "precision",
                        "label": "Precision",
                        "format": "percent",
                    },
                    {
                        "field": "recall",
                        "label": "Recall",
                        "format": "percent",
                    },
                    {
                        "field": "f1",
                        "label": "F1",
                        "format": "percent",
                    },
                    {
                        "field": "support",
                        "label": "Images",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "uncertainty_table",
                "title": "Uncertainty behavior by true category",
                "subtitle": (
                    "Accepted and uncertain counts at threshold 0.60."
                ),
                "dataset": "uncertainty",
                "sourceId": "uncertainty",
                "defaultSort": {
                    "field": "accepted_accuracy",
                    "direction": "desc",
                },
                "columns": [
                    {
                        "field": "category",
                        "label": "Category",
                        "type": "text",
                    },
                    {
                        "field": "accepted_predictions",
                        "label": "Accepted",
                        "format": "number",
                    },
                    {
                        "field": "uncertain_predictions",
                        "label": "Uncertain",
                        "format": "number",
                    },
                    {
                        "field": "coverage",
                        "label": "Coverage",
                        "format": "percent",
                    },
                    {
                        "field": "accepted_accuracy",
                        "label": "Accepted accuracy",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "errors_table",
                "title": "Final error directions",
                "subtitle": "Every observed true-to-predicted mistake pair.",
                "dataset": "errors",
                "sourceId": "errors",
                "defaultSort": {"field": "errors", "direction": "desc"},
                "columns": [
                    {
                        "field": "category",
                        "label": "True class",
                        "type": "text",
                    },
                    {
                        "field": "predicted_category",
                        "label": "Predicted class",
                        "type": "text",
                    },
                    {
                        "field": "errors",
                        "label": "Errors",
                        "format": "number",
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# FontSense final CNN evaluation",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "headline",
                "body": (
                    "## Final held-out result\n\n"
                    f"The selected CNN achieved **{metrics['test_results']['macro_f1']:.3f} "
                    "macro F1** and "
                    f"**{metrics['test_results']['accuracy']:.1%} accuracy** "
                    "on 600 images from 15 previously untouched font "
                    "families. It made "
                    f"{metrics['test_results']['correct_predictions']} "
                    "correct and "
                    f"{metrics['test_results']['incorrect_predictions']} "
                    "incorrect predictions. This is the single final test "
                    "evaluation; the model and threshold were not changed "
                    "afterward."
                ),
            },
            {
                "id": "metrics",
                "type": "metric-strip",
                "cardIds": [
                    "macro_f1_card",
                    "accuracy_card",
                    "accepted_accuracy_card",
                    "uncertain_card",
                ],
            },
            {
                "id": "class_findings",
                "type": "markdown",
                "sourceId": "classification",
                "body": (
                    "## Performance differs by font category\n\n"
                    f"**{strongest['class'].replace('_', ' ').title()}** "
                    f"was strongest at {strongest['f1']:.3f} F1. "
                    f"**{weakest['class'].replace('_', ' ').title()}** "
                    f"was weakest at {weakest['f1']:.3f} F1. Equal "
                    "support means these differences are not caused by "
                    "class-count imbalance."
                ),
            },
            {
                "id": "class_chart_block",
                "type": "chart",
                "chartId": "class_f1_chart",
            },
            {
                "id": "class_note",
                "type": "markdown",
                "sourceId": "classification",
                "body": (
                    "Display and handwriting generalized strongly. Sans "
                    "serif recall was much lower, showing that many true "
                    "sans-serif examples were assigned to visually similar "
                    "categories."
                ),
            },
            {
                "id": "class_table_block",
                "type": "table",
                "tableId": "classification_table",
            },
            {
                "id": "confusion_findings",
                "type": "markdown",
                "sourceId": "errors",
                "body": (
                    "## Most mistakes involve sans serif, serif, and monospace\n\n"
                    f"The largest error direction was "
                    f"**{top_error['category'].replace('_', ' ')} → "
                    f"{top_error['predicted_category'].replace('_', ' ')}** "
                    f"({top_error['errors']} images). These categories share "
                    "straight strokes and can look similar in short crops."
                ),
            },
            {
                "id": "confusion_chart_block",
                "type": "table",
                "tableId": "confusion_table",
            },
            {
                "id": "confusion_note",
                "type": "markdown",
                "sourceId": "confusion",
                "body": (
                    "The diagonal contains correct predictions. Off-diagonal "
                    "cells show exactly where the model confused one true "
                    "category for another."
                ),
            },
            {
                "id": "error_chart_block",
                "type": "chart",
                "chartId": "error_chart",
            },
            {
                "id": "error_note",
                "type": "markdown",
                "sourceId": "errors",
                "body": (
                    "The ranked error view makes the main failure directions "
                    "easier to compare than the complete matrix."
                ),
            },
            {
                "id": "errors_table_block",
                "type": "table",
                "tableId": "errors_table",
            },
            {
                "id": "uncertainty_findings",
                "type": "markdown",
                "sourceId": "headline",
                "body": (
                    "## The validation-only threshold improves accepted accuracy\n\n"
                    f"Threshold 0.60 marked "
                    f"{metrics['uncertainty_results']['uncertain_predictions']} "
                    "predictions uncertain. The remaining "
                    f"{metrics['uncertainty_results']['accepted_predictions']} "
                    "accepted predictions were "
                    f"{metrics['uncertainty_results']['accepted_accuracy']:.1%} "
                    "accurate. This threshold was fixed using validation "
                    "data before the test split was loaded."
                ),
            },
            {
                "id": "confidence_chart_block",
                "type": "chart",
                "chartId": "confidence_chart",
            },
            {
                "id": "confidence_note",
                "type": "markdown",
                "sourceId": "confidence",
                "body": (
                    "Incorrect predictions are concentrated at lower "
                    "confidence, but some errors remain highly confident. "
                    "The warning therefore reduces risk without guaranteeing "
                    "that every accepted result is correct."
                ),
            },
            {
                "id": "uncertainty_table_block",
                "type": "table",
                "tableId": "uncertainty_table",
            },
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## Scope and metric definitions\n\n"
                    "- **Macro F1** averages F1 equally across the five "
                    "categories.\n"
                    "- **Accuracy** is correct predictions divided by 600.\n"
                    "- **Accepted accuracy** uses predictions with maximum "
                    "probability at least 0.60.\n"
                    "- The test cohort contains three unseen font families "
                    "per category and 40 images per family."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Frozen evaluation method\n\n"
                    "The epoch-14 Reference small CNN checkpoint, class "
                    "order, seed, grayscale 112×48 preprocessing, manifest "
                    "hash, and family-split hash were recorded before test "
                    "access. The threshold was chosen from validation "
                    "predictions only. The test command had no training or "
                    "threshold-tuning option and is guarded against reruns."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and remaining uncertainty\n\n"
                    "- Test data are synthetic rendered crops, not real "
                    "screenshots.\n"
                    "- Only three held-out families represent each category.\n"
                    "- Sans serif remains weak, especially after confidence "
                    "filtering.\n"
                    "- Some wrong predictions are confident, so uncertainty "
                    "warnings are not a correctness guarantee.\n"
                    "- This single test result should not be used for further "
                    "model selection or tuning."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next step\n\n"
                    "Keep this checkpoint and threshold fixed. Continue with "
                    "post-evaluation error analysis and then update the "
                    "inference demo to use the checkpoint’s exact 112×48 "
                    "preprocessing and the frozen 0.60 warning threshold."
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "How well does the frozen model handle real screenshots, "
                    "cropping noise, mixed backgrounds, and font families "
                    "outside this Google Fonts sample?"
                ),
            },
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "classification": classification_rows,
                "confusion": confusion_rows,
                "confidence": confidence_rows,
                "uncertainty": uncertainty_rows,
                "errors": error_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chart_map = [
        {
            "section": "Class performance",
            "question": "Which categories generalized best and worst?",
            "chart": "bar",
            "fields": ["class", "f1", "precision", "recall"],
            "supported_claim": (
                f"{strongest['class']} is strongest and "
                f"{weakest['class']} is weakest."
            ),
            "palette_policy": "single-root blue",
            "artifact": (
                "reports/final_evaluation/final_evaluation_report.html"
            ),
        },
        {
            "section": "Confusion matrix",
            "question": "Where did true and predicted categories differ?",
            "chart": "static matrix",
            "fields": [
                "true_category",
                "predicted_category",
                "test_images",
            ],
            "supported_claim": "Shows every final test confusion cell.",
            "palette_policy": "single-root blue",
            "artifact": (
                "reports/final_evaluation/figures/"
                "final_test_confusion_matrix.png"
            ),
        },
        {
            "section": "Confidence distribution",
            "question": (
                "How does confidence differ for correct and incorrect results?"
            ),
            "chart": "stacked bar",
            "fields": ["confidence_bin", "outcome", "predictions"],
            "supported_claim": (
                "Errors are lower-confidence overall, with confident "
                "exceptions."
            ),
            "palette_policy": "hard two-root cap",
            "artifact": (
                "reports/final_evaluation/final_evaluation_report.html"
            ),
        },
        {
            "section": "Error directions",
            "question": "Which mistake directions occurred most often?",
            "chart": "grouped bar",
            "fields": ["category", "predicted_category", "errors"],
            "supported_claim": (
                f"{top_error['category']} to "
                f"{top_error['predicted_category']} is the top error."
            ),
            "palette_policy": "relaxed multi-category",
            "artifact": (
                "reports/final_evaluation/final_evaluation_report.html"
            ),
        },
    ]
    (REPORT_DIR / "chart_map.json").write_text(
        json.dumps(chart_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    build_report_artifact()
