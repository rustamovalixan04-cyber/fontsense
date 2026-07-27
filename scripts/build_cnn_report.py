"""Build the canonical portable report input for FontSense CNN experiments."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = ROOT / "reports" / "cnn"
ARTIFACT_PATH = CNN_DIR / "cnn_report_artifact.json"


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
    summary = json.loads(
        (CNN_DIR / "cnn_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if summary["status"] != "passed":
        raise RuntimeError(
            "CNN validation must pass before building the report."
        )
    cnn_runs = pd.read_csv(CNN_DIR / "cnn_experiment_comparison.csv")
    model_comparison = pd.read_csv(CNN_DIR / "model_comparison.csv")
    per_class = pd.read_csv(
        CNN_DIR / "best_cnn_classification_report.csv"
    )
    history = pd.read_csv(CNN_DIR / "cnn_training_history.csv")
    predictions = pd.read_csv(
        CNN_DIR / "best_cnn_validation_predictions.csv"
    )
    best_name = summary["best_cnn_run"]["name"]

    connection = sqlite3.connect(":memory:")
    cnn_runs.to_sql("cnn_experiments", connection, index=False)
    model_comparison.to_sql("model_comparison", connection, index=False)
    per_class.to_sql("best_cnn_per_class", connection, index=False)
    history.to_sql("cnn_training_history", connection, index=False)
    predictions.to_sql("best_cnn_predictions", connection, index=False)
    headline = pd.DataFrame(
        [
            {
                "best_cnn_macro_f1": summary["best_cnn_run"][
                    "validation_macro_f1"
                ],
                "best_cnn_accuracy": summary["best_cnn_run"][
                    "validation_accuracy"
                ],
                "best_cnn_model_bytes": summary["best_cnn_run"][
                    "saved_model_size_bytes"
                ],
                "test_images_evaluated": summary["test_rows_evaluated"],
            }
        ]
    )
    headline.to_sql("cnn_headline", connection, index=False)

    headline_sql = """
        SELECT best_cnn_macro_f1, best_cnn_accuracy,
               best_cnn_model_bytes, test_images_evaluated
        FROM cnn_headline
    """.strip()
    cnn_runs_sql = """
        SELECT run_order, run_name, reason, learning_rate, width, dropout,
               image_width, image_height, batch_size, epochs_trained,
               best_epoch, stopped_early, validation_macro_f1,
               validation_accuracy, training_seconds,
               inference_ms_per_image, model_size_bytes, parameter_count
        FROM cnn_experiments
        ORDER BY run_order
    """.strip()
    model_comparison_sql = """
        SELECT model, model_family, validation_macro_f1,
               validation_accuracy, inference_ms_per_image,
               model_size_bytes, selection_basis
        FROM model_comparison
        ORDER BY validation_macro_f1 DESC
    """.strip()
    per_class_sql = """
        SELECT class, precision, recall, f1, support
        FROM best_cnn_per_class
        ORDER BY f1 DESC, class
    """.strip()
    learning_sql = f"""
        SELECT epoch, 'Train loss' AS series, train_loss AS value,
               train_accuracy, validation_accuracy, validation_macro_f1
        FROM cnn_training_history
        WHERE run = '{best_name.replace("'", "''")}'
        UNION ALL
        SELECT epoch, 'Validation loss', validation_loss,
               train_accuracy, validation_accuracy, validation_macro_f1
        FROM cnn_training_history
        WHERE run = '{best_name.replace("'", "''")}'
        ORDER BY epoch, series
    """.strip()
    confusion_sql = """
        SELECT category AS actual_category,
               predicted_category,
               COUNT(*) AS validation_images
        FROM best_cnn_predictions
        WHERE correct = 0
        GROUP BY category, predicted_category
        ORDER BY validation_images DESC, category, predicted_category
    """.strip()

    headline_rows = _records(pd.read_sql_query(headline_sql, connection))
    cnn_run_rows = _records(
        pd.read_sql_query(cnn_runs_sql, connection)
    )
    model_rows = _records(
        pd.read_sql_query(model_comparison_sql, connection)
    )
    class_rows = _records(
        pd.read_sql_query(per_class_sql, connection)
    )
    learning_rows = _records(
        pd.read_sql_query(learning_sql, connection)
    )
    confusion_rows = _records(
        pd.read_sql_query(confusion_sql, connection)
    )
    connection.close()

    best = summary["best_cnn_run"]
    selected = summary["model_selected_for_final_test"]
    strongest = max(class_rows, key=lambda row: row["f1"])
    weakest = min(class_rows, key=lambda row: row["f1"])
    hog = next(
        row for row in model_rows if row["model_family"] == "hog"
    )
    cnn = next(
        row for row in model_rows if row["model_family"] == "cnn"
    )
    difference = cnn["validation_macro_f1"] - hog["validation_macro_f1"]
    if confusion_rows:
        top_confusion = confusion_rows[0]
        confusion_sentence = (
            f"The most frequent CNN error was "
            f"{top_confusion['actual_category'].replace('_', ' ')} "
            f"predicted as "
            f"{top_confusion['predicted_category'].replace('_', ' ')} "
            f"({top_confusion['validation_images']} images)."
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
            "CNN validation headline metrics",
            "reports/cnn/cnn_validation_summary.json",
            headline_sql,
            "Reads the reviewed selected-CNN metrics and test-access guard.",
            generated_at,
            ["cnn_headline"],
        ),
        _source(
            "cnn_runs",
            "CNN validation experiment comparison",
            "reports/cnn/cnn_experiment_comparison.csv",
            cnn_runs_sql,
            "Reads controlled settings, results, runtime, and size for each CNN run.",
            generated_at,
            ["cnn_experiments"],
        ),
        _source(
            "model_comparison",
            "Majority, HOG, and CNN validation comparison",
            "reports/cnn/model_comparison.csv",
            model_comparison_sql,
            "Compares the selected validation result from each required model family.",
            generated_at,
            ["model_comparison"],
        ),
        _source(
            "per_class",
            "Best CNN per-class validation metrics",
            "reports/cnn/best_cnn_classification_report.csv",
            per_class_sql,
            "Reads precision, recall, F1, and support for all five categories.",
            generated_at,
            ["best_cnn_per_class"],
        ),
        _source(
            "learning",
            "Best CNN training history",
            "reports/cnn/cnn_training_history.csv",
            learning_sql,
            "Reads train and validation loss from the selected CNN run by epoch.",
            generated_at,
            ["cnn_training_history"],
        ),
        _source(
            "confusions",
            "Best CNN validation predictions",
            "reports/cnn/best_cnn_validation_predictions.csv",
            confusion_sql,
            "Counts incorrect validation predictions by actual and predicted category.",
            generated_at,
            ["best_cnn_predictions"],
        ),
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FontSense CNN validation experiments",
            "description": (
                "Validation-only comparison of three small CNN experiments "
                "and the earlier majority and HOG baselines."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "cnn_f1_card",
                    "description": "Primary validation selection metric.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "Best CNN macro F1",
                            "field": "best_cnn_macro_f1",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "cnn_accuracy_card",
                    "description": "Validation accuracy at the selected epoch.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "Best CNN accuracy",
                            "field": "best_cnn_accuracy",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "cnn_size_card",
                    "description": "Saved selected checkpoint size in bytes.",
                    "dataset": "headline",
                    "sourceId": "headline",
                    "metrics": [
                        {
                            "label": "CNN checkpoint bytes",
                            "field": "best_cnn_model_bytes",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "test_card",
                    "description": "Test images loaded, predicted, or scored.",
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
                    "id": "cnn_runs_chart",
                    "title": "Validation macro F1 by CNN experiment",
                    "subtitle": (
                        "Three controlled runs on the same 600 validation images."
                    ),
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "cnn_runs",
                    "sourceId": "cnn_runs",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {
                            "field": "run_name",
                            "type": "nominal",
                            "label": "CNN run",
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
                                "label": "Accuracy",
                            },
                            {
                                "field": "training_seconds",
                                "type": "quantitative",
                                "label": "Training seconds",
                            },
                        ],
                    },
                },
                {
                    "id": "model_comparison_chart",
                    "title": "Validation macro F1 by required model family",
                    "subtitle": (
                        "Majority, selected HOG, and selected CNN results; "
                        "no test metrics."
                    ),
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "model_comparison",
                    "sourceId": "model_comparison",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {
                            "field": "model",
                            "type": "nominal",
                            "label": "Model",
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
                                "label": "Accuracy",
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
                    "id": "learning_chart",
                    "title": "Selected CNN train and validation loss",
                    "subtitle": (
                        f"{best_name}; early stopping monitors validation macro F1."
                    ),
                    "type": "line",
                    "intent": "trend",
                    "dataset": "learning",
                    "sourceId": "learning",
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "Epoch",
                        },
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "label": "Cross-entropy loss",
                        },
                        "color": {
                            "field": "series",
                            "type": "nominal",
                            "label": "Series",
                        },
                    },
                },
                {
                    "id": "class_f1_chart",
                    "title": "Selected CNN validation F1 by category",
                    "subtitle": (
                        "Each category has 120 images from three unseen font families."
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
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "cnn_runs_table",
                    "title": "CNN experiment settings and validation results",
                    "subtitle": (
                        "One controlled parameter change per alternative run."
                    ),
                    "dataset": "cnn_runs",
                    "sourceId": "cnn_runs",
                    "defaultSort": {
                        "field": "validation_macro_f1",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "run_name", "label": "Run", "type": "text"},
                        {
                            "field": "learning_rate",
                            "label": "Learning rate",
                            "format": "number",
                        },
                        {
                            "field": "width",
                            "label": "Base filters",
                            "format": "number",
                        },
                        {
                            "field": "best_epoch",
                            "label": "Best epoch",
                            "format": "number",
                        },
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
                            "field": "training_seconds",
                            "label": "Training seconds",
                            "format": "number",
                        },
                        {
                            "field": "model_size_bytes",
                            "label": "Model bytes",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "class_table",
                    "title": "Selected CNN per-class validation metrics",
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
                    "body": "# FontSense CNN validation experiments",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "headline",
                    "body": (
                        "## Technical summary\n\n"
                        f"**{best_name} was the strongest CNN at validation macro "
                        f"F1 {best['validation_macro_f1']:.3f} and accuracy "
                        f"{best['validation_accuracy']:.3f}.** The model-selection "
                        f"comparison recommends **{selected['model']}** for the one "
                        "final test evaluation. No test image was loaded or scored."
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "cnn_f1_card",
                        "cnn_accuracy_card",
                        "cnn_size_card",
                        "test_card",
                    ],
                },
                {
                    "id": "cnn_findings",
                    "type": "markdown",
                    "sourceId": "cnn_runs",
                    "body": (
                        "## Controlled CNN experiments identify the best training setup\n\n"
                        "**Each alternative changes one setting from the reference.** "
                        "This makes the learning-rate and filter-width comparisons "
                        "meaningful instead of increasing run count with duplicates."
                    ),
                },
                {
                    "id": "cnn_runs_chart_block",
                    "type": "chart",
                    "chartId": "cnn_runs_chart",
                },
                {
                    "id": "cnn_runs_note",
                    "type": "markdown",
                    "sourceId": "cnn_runs",
                    "body": (
                        f"**{best_name} produced the highest CNN validation macro F1.** "
                        f"Its best checkpoint came from epoch {best['best_epoch']}; "
                        "later epochs do not replace it unless macro F1 improves."
                    ),
                },
                {
                    "id": "cnn_runs_table_block",
                    "type": "table",
                    "tableId": "cnn_runs_table",
                },
                {
                    "id": "model_selection",
                    "type": "markdown",
                    "sourceId": "model_comparison",
                    "body": (
                        "## Validation evidence determines the final-test candidate\n\n"
                        f"**The best CNN differs from the selected HOG model by "
                        f"{difference:+.3f} macro F1.** The recommendation follows "
                        "the primary metric first; test performance remains unknown."
                    ),
                },
                {
                    "id": "model_comparison_block",
                    "type": "chart",
                    "chartId": "model_comparison_chart",
                },
                {
                    "id": "model_comparison_note",
                    "type": "markdown",
                    "sourceId": "model_comparison",
                    "body": (
                        f"**Select {selected['model']} for final evaluation.** "
                        "This choice is based only on train/validation evidence and "
                        "does not claim that the same ranking will hold on test families."
                    ),
                },
                {
                    "id": "learning",
                    "type": "markdown",
                    "sourceId": "learning",
                    "body": (
                        "## The selected checkpoint protects against later overfitting\n\n"
                        "**Train and validation loss are recorded for every completed "
                        "epoch.** Mild random affine and sharpness changes appear only "
                        "in training; validation preprocessing is deterministic."
                    ),
                },
                {
                    "id": "learning_chart_block",
                    "type": "chart",
                    "chartId": "learning_chart",
                },
                {
                    "id": "learning_note",
                    "type": "markdown",
                    "sourceId": "learning",
                    "body": (
                        f"**The saved state is epoch {best['best_epoch']}, not "
                        "automatically the final epoch.** This is the early-stopping "
                        "checkpoint selected by validation macro F1."
                    ),
                },
                {
                    "id": "class_results",
                    "type": "markdown",
                    "sourceId": "per_class",
                    "body": (
                        "## Performance still varies by font category\n\n"
                        f"**{strongest['class'].replace('_', ' ').title()} is strongest "
                        f"at F1 {strongest['f1']:.3f}; "
                        f"{weakest['class'].replace('_', ' ').title()} is weakest "
                        f"at {weakest['f1']:.3f}.**"
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
                    "sourceId": "confusions",
                    "body": (
                        f"**{confusion_sentence}** The saved confusion matrices show "
                        "exact counts for every CNN run. Per-class differences matter "
                        "because overall accuracy can hide a weak category."
                    ),
                },
                {
                    "id": "class_table_block",
                    "type": "table",
                    "tableId": "class_table",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Scope, data, and metric definitions\n\n"
                        "- **Fit cohort:** 2,400 train images from 60 font families.\n"
                        "- **Selection cohort:** 600 validation images from 15 different families.\n"
                        "- **Input:** one grayscale 112×48 tensor normalized to [-1, 1].\n"
                        "- **Macro F1:** equal-weight average of the five category F1 scores.\n"
                        "- **Inference time:** validation preprocessing and model forward pass per image.\n"
                        "- **Model size:** saved PyTorch checkpoint bytes."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Model specification and validation design\n\n"
                        "The compact network uses four 3×3 convolution blocks, batch "
                        "normalization, ReLU, max pooling, global average pooling, "
                        "dropout, and one five-output linear layer. AdamW minimizes "
                        "cross-entropy. Fixed seed 42 and deterministic algorithms "
                        "support repeatability. Only training tensors receive mild "
                        "augmentation. The checkpoint with the highest validation macro "
                        "F1 is retained, then reloaded for a five-probability smoke test."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations, uncertainty, and robustness checks\n\n"
                        "- Validation contains only three unseen families per category.\n"
                        "- All training ran on CPU and synthetic rendered images.\n"
                        "- A single seed gives reproducibility but not uncertainty across seeds.\n"
                        "- Real screenshots may contain layout and capture noise absent here.\n"
                        "- Test families remain untouched, so final generalization is unknown."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next step\n\n"
                        f"Evaluate **{selected['model']}** once on the frozen test split. "
                        "Do not tune after viewing test results. Then complete error "
                        "analysis and update the inference demo only after the final "
                        "model choice is documented."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does the validation winner remain strongest on unseen test families?\n"
                        "- Which individual families drive the weakest category?\n"
                        "- How large is the synthetic-to-real screenshot gap?"
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
                "cnn_runs": cnn_run_rows,
                "model_comparison": model_rows,
                "per_class": class_rows,
                "learning": learning_rows,
                "confusions": confusion_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    CNN_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    chart_map = [
        {
            "section": "CNN experiment comparison",
            "question": "Which controlled CNN run has the best validation macro F1?",
            "chart": "bar",
            "fields": ["run_name", "validation_macro_f1"],
            "supported_claim": f"{best_name} is the strongest CNN run.",
            "palette_policy": "single-root blue",
            "artifact": "reports/cnn/cnn_experiment_report.html",
        },
        {
            "section": "Required model comparison",
            "question": "Which model family should reach final test evaluation?",
            "chart": "bar",
            "fields": ["model", "validation_macro_f1"],
            "supported_claim": f"{selected['model']} has the highest validation macro F1.",
            "palette_policy": "single-root blue",
            "artifact": "reports/cnn/cnn_experiment_report.html",
        },
        {
            "section": "Selected CNN learning",
            "question": "How do train and validation loss change by epoch?",
            "chart": "line",
            "fields": ["epoch", "series", "value"],
            "supported_claim": "The best validation checkpoint is not assumed to be the final epoch.",
            "palette_policy": "hard two-root cap",
            "artifact": "reports/cnn/cnn_experiment_report.html",
        },
        {
            "section": "Per-class validation",
            "question": "Which font categories are strongest and weakest?",
            "chart": "bar",
            "fields": ["class", "f1"],
            "supported_claim": (
                f"{strongest['class']} is strongest and {weakest['class']} is weakest."
            ),
            "palette_policy": "single-root blue",
            "artifact": "reports/cnn/cnn_experiment_report.html",
        },
        {
            "section": "Run-level confusion matrices",
            "question": "Which actual categories are confused with which predictions?",
            "chart": "static matrix",
            "fields": ["actual_category", "predicted_category", "count"],
            "supported_claim": confusion_sentence,
            "palette_policy": "single-root blue",
            "artifact": "reports/cnn/figures/*_confusion_matrix.png",
        },
    ]
    (CNN_DIR / "chart_map.json").write_text(
        json.dumps(chart_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    build_report_artifact()
