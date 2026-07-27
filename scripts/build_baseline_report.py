"""Build the canonical portable report input for the FontSense baselines."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "reports" / "baseline"
ARTIFACT_PATH = BASELINE_DIR / "baseline_report_artifact.json"


def _records(frame: pd.DataFrame) -> list[dict]:
    """Convert pandas missing values to JSON nulls."""
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
    summary = json.loads(
        (BASELINE_DIR / "baseline_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if summary["status"] != "passed":
        raise RuntimeError(
            "Baseline validation must pass before building the report."
        )
    comparison = pd.read_csv(BASELINE_DIR / "validation_comparison.csv")
    per_class = pd.read_csv(
        BASELINE_DIR / "best_hog_classification_report.csv"
    )
    predictions = pd.read_csv(
        BASELINE_DIR / "best_hog_validation_predictions.csv"
    )

    connection = sqlite3.connect(":memory:")
    comparison.to_sql("validation_comparison", connection, index=False)
    per_class.to_sql("best_hog_per_class", connection, index=False)
    predictions.to_sql("best_hog_predictions", connection, index=False)
    headline = pd.DataFrame(
        [
            {
                "majority_macro_f1": summary["majority_baseline"][
                    "validation_macro_f1"
                ],
                "best_hog_macro_f1": summary["best_hog_run"][
                    "validation_macro_f1"
                ],
                "best_hog_accuracy": summary["best_hog_run"][
                    "validation_accuracy"
                ],
                "test_images_evaluated": summary["test_rows_evaluated"],
            }
        ]
    )
    headline.to_sql("baseline_headline", connection, index=False)

    headline_sql = """
        SELECT majority_macro_f1, best_hog_macro_f1,
               best_hog_accuracy, test_images_evaluated
        FROM baseline_headline
    """.strip()
    comparison_sql = """
        SELECT run_order, run_name, model_type, reason,
               image_width, image_height, pixels_per_cell, C,
               validation_macro_f1, validation_accuracy,
               inference_ms_per_image, model_size_bytes
        FROM validation_comparison
        ORDER BY run_order
    """.strip()
    per_class_sql = """
        SELECT class, precision, recall, f1, support
        FROM best_hog_per_class
        ORDER BY f1 DESC, class
    """.strip()
    confusion_sql = """
        SELECT category AS actual_category,
               predicted_category,
               COUNT(*) AS validation_images
        FROM best_hog_predictions
        WHERE correct = 0
        GROUP BY category, predicted_category
        ORDER BY validation_images DESC, category, predicted_category
    """.strip()

    headline_rows = _records(pd.read_sql_query(headline_sql, connection))
    comparison_rows = _records(
        pd.read_sql_query(comparison_sql, connection)
    )
    class_rows = _records(pd.read_sql_query(per_class_sql, connection))
    confusion_rows = _records(
        pd.read_sql_query(confusion_sql, connection)
    )
    connection.close()

    best = summary["best_hog_run"]
    majority = summary["majority_baseline"]
    weakest = min(class_rows, key=lambda row: row["f1"])
    strongest = max(class_rows, key=lambda row: row["f1"])
    if confusion_rows:
        most_common_confusion = confusion_rows[0]
        confusion_sentence = (
            f"The most frequent error was "
            f"{most_common_confusion['actual_category'].replace('_', ' ')} "
            f"predicted as "
            f"{most_common_confusion['predicted_category'].replace('_', ' ')} "
            f"({most_common_confusion['validation_images']} images)."
        )
    else:
        confusion_sentence = "No validation errors were recorded."

    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    sources = [
        _source(
            "headline",
            "Baseline validation headline metrics",
            "reports/baseline/baseline_validation_summary.json",
            headline_sql,
            "Reads the reviewed majority and selected HOG validation metrics.",
            generated_at,
            ["baseline_headline"],
        ),
        _source(
            "comparison",
            "Validation experiment comparison",
            "reports/baseline/validation_comparison.csv",
            comparison_sql,
            "Reads settings, metrics, runtime, and model size for each real run.",
            generated_at,
            ["validation_comparison"],
        ),
        _source(
            "per_class",
            "Best HOG per-class validation metrics",
            "reports/baseline/best_hog_classification_report.csv",
            per_class_sql,
            "Reads precision, recall, F1, and support for all five classes.",
            generated_at,
            ["best_hog_per_class"],
        ),
        _source(
            "confusions",
            "Best HOG validation predictions",
            "reports/baseline/best_hog_validation_predictions.csv",
            confusion_sql,
            "Counts incorrect validation predictions by actual and predicted class.",
            generated_at,
            ["best_hog_predictions"],
        ),
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FontSense validation baseline report",
            "description": (
                "Validation-only comparison of a majority baseline and "
                "HOG with multinomial Logistic Regression."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "majority_card",
                    "description": "Sanity-check macro F1 on validation families.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "Majority macro F1",
                            "field": "majority_macro_f1",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "hog_f1_card",
                    "description": "Primary selection metric for the best HOG run.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "Best HOG macro F1",
                            "field": "best_hog_macro_f1",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "hog_accuracy_card",
                    "description": "Validation accuracy for the selected HOG run.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "Best HOG accuracy",
                            "field": "best_hog_accuracy",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "test_card",
                    "description": "Test images loaded, predicted, or scored in this task.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "Test images evaluated",
                            "field": "test_images_evaluated",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "run_comparison_chart",
                    "title": "Validation macro F1 by baseline run",
                    "subtitle": (
                        "Primary selection metric on 600 validation images "
                        "from unseen families."
                    ),
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "comparison",
                    "sourceId": "comparison",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {
                            "field": "run_name",
                            "type": "nominal",
                            "label": "Run",
                        },
                        "y": {
                            "field": "validation_macro_f1",
                            "type": "quantitative",
                            "label": "Validation macro F1",
                        },
                        "tooltip": [
                            {
                                "field": "validation_accuracy",
                                "type": "quantitative",
                                "label": "Validation accuracy",
                            },
                            {
                                "field": "inference_ms_per_image",
                                "type": "quantitative",
                                "label": "Inference ms/image",
                            },
                        ],
                    },
                },
                {
                    "id": "class_f1_chart",
                    "title": "Best HOG validation F1 by class",
                    "subtitle": (
                        "Each category has 120 validation images from three "
                        "families not used for fitting."
                    ),
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "per_class",
                    "sourceId": "per_class",
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
                            "label": "Validation F1",
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
                            {
                                "field": "support",
                                "type": "quantitative",
                                "label": "Images",
                            },
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "comparison_table",
                    "title": "Validation run comparison",
                    "subtitle": (
                        "Controlled settings and measured validation results; "
                        "no test metrics."
                    ),
                    "dataset": "comparison",
                    "sourceId": "comparison",
                    "defaultSort": {
                        "field": "validation_macro_f1",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "run_name", "label": "Run", "type": "text"},
                        {
                            "field": "validation_macro_f1",
                            "label": "Macro F1",
                            "format": "percent",
                        },
                        {
                            "field": "validation_accuracy",
                            "label": "Accuracy",
                            "format": "percent",
                        },
                        {
                            "field": "inference_ms_per_image",
                            "label": "Inference ms/image",
                            "format": "number",
                        },
                        {
                            "field": "model_size_bytes",
                            "label": "Model bytes",
                            "format": "number",
                        },
                        {
                            "field": "reason",
                            "label": "Reason for run",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "class_table",
                    "title": "Best HOG per-class validation metrics",
                    "subtitle": "Precision, recall, and F1 for all five categories.",
                    "dataset": "per_class",
                    "sourceId": "per_class",
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
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# FontSense validation baseline report",
                },
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "headline",
                    "body": (
                        "## Technical summary\n\n"
                        f"**{best['name']} was selected using validation macro F1 "
                        f"({best['validation_macro_f1']:.3f}).** Its validation "
                        f"accuracy is {best['validation_accuracy']:.3f}. The majority "
                        f"sanity check scored {majority['validation_macro_f1']:.3f} "
                        "macro F1. These are validation results, not final test results."
                    ),
                },
                {
                    "id": "cards",
                    "type": "metric-strip",
                    "cardIds": [
                        "majority_card",
                        "hog_f1_card",
                        "hog_accuracy_card",
                        "test_card",
                    ],
                },
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "sourceId": "comparison",
                    "body": (
                        "## Key findings\n\n"
                        "**Every HOG experiment is a real controlled change.** "
                        "The comparison varies resolution, HOG cell size, or "
                        "regularization from one reference configuration."
                    ),
                },
                {
                    "id": "comparison_chart",
                    "type": "chart",
                    "chartId": "run_comparison_chart",
                },
                {
                    "id": "comparison_note",
                    "type": "markdown",
                    "sourceId": "comparison",
                    "body": (
                        f"**The selected run is {best['name']}.** The majority result "
                        "is included only to prove that the learned model beats a "
                        "simple class-frequency guess."
                    ),
                },
                {
                    "id": "comparison_detail",
                    "type": "table",
                    "tableId": "comparison_table",
                },
                {
                    "id": "class_findings",
                    "type": "markdown",
                    "sourceId": "per_class",
                    "body": (
                        "## Per-class validation results\n\n"
                        f"**{strongest['class'].replace('_', ' ').title()} has the "
                        f"highest F1 ({strongest['f1']:.3f}), while "
                        f"{weakest['class'].replace('_', ' ').title()} has the "
                        f"lowest ({weakest['f1']:.3f}).**"
                    ),
                },
                {
                    "id": "class_chart",
                    "type": "chart",
                    "chartId": "class_f1_chart",
                },
                {
                    "id": "class_note",
                    "type": "markdown",
                    "sourceId": "confusions",
                    "body": (
                        f"**{confusion_sentence}** The saved confusion-matrix PNG "
                        "shows every actual-versus-predicted count."
                    ),
                },
                {
                    "id": "class_detail",
                    "type": "table",
                    "tableId": "class_table",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Scope, data, and metric definitions\n\n"
                        "- **Fit data:** 2,400 train images from 60 font families.\n"
                        "- **Selection data:** 600 validation images from 15 different families.\n"
                        "- **Macro F1:** the average of the five class F1 scores, giving every class equal weight.\n"
                        "- **Accuracy:** the share of validation images classified correctly.\n"
                        "- **Inference time:** full saved pipeline time, including grayscale resize and HOG extraction.\n"
                        "- **Model size:** compressed saved pipeline size in bytes."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Methodology and checks\n\n"
                        "Every image follows the same deterministic grayscale, fit-to-size, "
                        "and HOG process for its run. Multinomial Logistic Regression returns "
                        "five probabilities. The HOG transformer records 2,400 fit rows, so "
                        "validation images are never fitted. Family assignments and the frozen "
                        "split SHA-256 are checked before and after training. The saved winner "
                        "is reloaded and used for one five-probability prediction."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and uncertainty\n\n"
                        "- The validation set contains only three unseen families per category.\n"
                        "- Synthetic rendered images may be easier than real screenshots.\n"
                        "- HOG describes edge direction and cannot learn higher-level shapes like a CNN.\n"
                        "- No test image was loaded or scored, so this report is not final assessed performance."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next step\n\n"
                        "Train and compare small CNN configurations using the same train and "
                        "validation family split. Keep the test set untouched until both model "
                        "families have been selected."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does a small CNN improve the weakest HOG category?\n"
                        "- Which individual validation families cause the largest errors?\n"
                        "- How much slower is CNN inference than this HOG baseline?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "comparison": comparison_rows,
                "per_class": class_rows,
                "confusions": confusion_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chart_map = [
        {
            "section": "Run comparison",
            "question": "Which real baseline run has the strongest validation macro F1?",
            "chart": "bar",
            "fields": ["run_name", "validation_macro_f1"],
            "palette_policy": "single-root blue",
            "artifact": "reports/baseline/baseline_summary_report.html",
        },
        {
            "section": "Per-class results",
            "question": "Which categories are strongest and weakest for the selected HOG model?",
            "chart": "bar",
            "fields": ["class", "f1"],
            "palette_policy": "single-root blue",
            "artifact": "reports/baseline/baseline_summary_report.html",
        },
        {
            "section": "Validation confusion",
            "question": "Which actual and predicted categories are confused?",
            "chart": "static confusion matrix",
            "fields": ["category", "predicted_category", "count"],
            "palette_policy": "single-root blue",
            "artifact": "reports/figures/hog_validation_confusion_matrix.png",
        },
    ]
    (BASELINE_DIR / "chart_map.json").write_text(
        json.dumps(chart_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    build_report_artifact()
