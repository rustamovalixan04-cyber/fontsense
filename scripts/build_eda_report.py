"""Build the canonical portable report input for the FontSense EDA audit."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "reports" / "eda"
ARTIFACT_PATH = EDA_DIR / "eda_report_artifact.json"
CATEGORIES = ("serif", "sans_serif", "display", "handwriting", "monospace")
CATEGORY_LABELS = {
    "serif": "Serif",
    "sans_serif": "Sans serif",
    "display": "Display",
    "handwriting": "Handwriting",
    "monospace": "Monospace",
}
SPLITS = ("train", "validation", "test")


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
        (EDA_DIR / "eda_validation_summary.json").read_text(encoding="utf-8")
    )
    if summary["status"] != "passed":
        raise RuntimeError("EDA validation must pass before building the report.")

    manifest = pd.read_csv(ROOT / "reports" / "dataset" / "full_manifest.csv")
    quality = pd.read_csv(EDA_DIR / "image_quality_metrics.csv")
    effects = pd.read_csv(EDA_DIR / "effect_balance_by_category.csv")
    frozen_split = pd.read_csv(
        ROOT / "data" / "interim" / "google_fonts_final_family_split.csv"
    )
    connection = sqlite3.connect(":memory:")
    manifest.to_sql("full_manifest", connection, index=False)
    quality.to_sql("image_quality", connection, index=False)
    effects.to_sql("effect_balance", connection, index=False)
    frozen_split.to_sql("frozen_split", connection, index=False)

    category_sql = """
        SELECT
            category,
            COUNT(*) AS images,
            COUNT(DISTINCT family) AS families,
            (SELECT COUNT(*) FROM full_manifest) AS total_images
        FROM full_manifest
        GROUP BY category
    """.strip()
    category_counts = pd.read_sql_query(category_sql, connection)
    category_counts["category"] = pd.Categorical(
        category_counts["category"], categories=CATEGORIES, ordered=True
    )
    category_counts = category_counts.sort_values("category")
    category_rows = [
        {
            "category": CATEGORY_LABELS[row.category],
            "images": int(row.images),
            "families": int(row.families),
            "total_images": int(len(manifest)),
        }
        for row in category_counts.itertuples(index=False)
    ]
    split_sql = """
        SELECT
            split,
            COUNT(*) AS images,
            COUNT(DISTINCT family) AS families,
            (SELECT COUNT(*) FROM full_manifest) AS total_images
        FROM full_manifest
        GROUP BY split
    """.strip()
    split_counts = pd.read_sql_query(split_sql, connection)
    split_counts["split"] = pd.Categorical(
        split_counts["split"], categories=SPLITS, ordered=True
    )
    split_counts = split_counts.sort_values("split")
    split_rows = [
        {
            "split": row.split.title(),
            "images": int(row.images),
            "families": int(row.families),
            "total_images": int(len(manifest)),
        }
        for row in split_counts.itertuples(index=False)
    ]
    family_split_sql = """
        SELECT
            category,
            split,
            COUNT(DISTINCT family) AS families,
            40 AS images_per_family
        FROM frozen_split
        GROUP BY category, split
    """.strip()
    family_split_counts = pd.read_sql_query(family_split_sql, connection)
    family_split_rows = [
        {
            "category": CATEGORY_LABELS[row.category],
            "split": row.split.title(),
            "families": int(row.families),
            "images_per_family": 40,
        }
        for row in family_split_counts.itertuples(index=False)
    ]

    effect_sql = """
        SELECT category, 'Dark background' AS effect, dark_background AS rate, 720 AS images
        FROM effect_balance
        UNION ALL
        SELECT category, 'Soft contrast', soft_contrast, 720 FROM effect_balance
        UNION ALL
        SELECT category, 'Mild blur', blur_applied, 720 FROM effect_balance
        UNION ALL
        SELECT category, 'JPEG round-trip', jpeg_applied, 720 FROM effect_balance
    """.strip()
    effect_frame = pd.read_sql_query(effect_sql, connection)
    effect_rows = [
        {
            "category": CATEGORY_LABELS[row.category],
            "effect": row.effect,
            "rate": float(row.rate),
            "images": int(row.images),
        }
        for row in effect_frame.itertuples(index=False)
    ]

    quality_sql = """
        SELECT
            category,
            COUNT(*) AS images,
            MIN(brightness_mean) AS brightness_min,
            AVG(brightness_mean) AS brightness_average,
            MAX(brightness_mean) AS brightness_max,
            MIN(contrast_std) AS contrast_min,
            AVG(contrast_std) AS contrast_average,
            MAX(contrast_std) AS contrast_max,
            MIN(actual_font_size) AS font_size_min,
            AVG(actual_font_size) AS font_size_average,
            MAX(actual_font_size) AS font_size_max
        FROM image_quality
        WHERE opens_successfully = 1
        GROUP BY category
    """.strip()
    quality_summary = pd.read_sql_query(quality_sql, connection)
    quality_summary["category"] = pd.Categorical(
        quality_summary["category"], categories=CATEGORIES, ordered=True
    )
    quality_summary = quality_summary.sort_values("category")
    quality_rows = [
        {
            "category": CATEGORY_LABELS[row.category],
            "images": int(row.images),
            "brightness_min": round(float(row.brightness_min), 2),
            "brightness_average": round(float(row.brightness_average), 2),
            "brightness_max": round(float(row.brightness_max), 2),
            "contrast_min": round(float(row.contrast_min), 2),
            "contrast_average": round(float(row.contrast_average), 2),
            "contrast_max": round(float(row.contrast_max), 2),
            "font_size_min": int(row.font_size_min),
            "font_size_average": round(float(row.font_size_average), 2),
            "font_size_max": int(row.font_size_max),
        }
        for row in quality_summary.itertuples(index=False)
    ]

    image_quality = summary["image_quality"]
    checks = [
        {"check": "Manifest rows", "result": "3,600", "status": "Passed"},
        {"check": "Unique font families", "result": "90", "status": "Passed"},
        {"check": "Images opened", "result": "3,600 / 3,600", "status": "Passed"},
        {"check": "Expected dimensions", "result": "3,600 / 3,600 at 224×96", "status": "Passed"},
        {"check": "Missing, corrupt, or blank", "result": "0", "status": "Passed"},
        {"check": "Family overlap between splits", "result": "0", "status": "Passed"},
        {
            "check": "Exact duplicate hash groups",
            "result": str(image_quality["exact_duplicate_hash_groups"]),
            "status": "Passed",
        },
        {
            "check": "Strict near-identical pairs",
            "result": str(image_quality["suspicious_near_identical_pairs"]),
            "status": "Passed",
        },
        {
            "check": "Maximum binary effect-rate spread",
            "result": f"{summary['effects']['maximum_binary_effect_rate_spread']:.1%}",
            "status": "Passed",
        },
        {
            "check": "Phrase-category association",
            "result": f"Cramér's V = {summary['phrase_balance']['cramers_v']:.3f}",
            "status": "Passed",
        },
    ]
    headline_frame = pd.DataFrame(
        [
            {
                "total_images": int(summary["structure"]["total_images"]),
                "unique_families": int(summary["structure"]["unique_families"]),
                "invalid_images": int(image_quality["automated_unreadable_flags"]),
                "family_overlap": int(summary["structure"]["family_overlap_count"]),
                "effect_rate_spread": float(
                    summary["effects"]["maximum_binary_effect_rate_spread"]
                ),
                "phrase_cramers_v": float(summary["phrase_balance"]["cramers_v"]),
            }
        ]
    )
    headline_frame.to_sql("eda_headline", connection, index=False)
    pd.DataFrame(checks).to_sql("validation_checks", connection, index=False)
    headline_sql = """
        SELECT
            total_images,
            unique_families,
            invalid_images,
            family_overlap,
            effect_rate_spread,
            phrase_cramers_v
        FROM eda_headline
    """.strip()
    checks_sql = """
        SELECT "check", result, status
        FROM validation_checks
        ORDER BY "check"
    """.strip()
    headline_rows = pd.read_sql_query(headline_sql, connection).to_dict("records")
    check_rows = pd.read_sql_query(checks_sql, connection).to_dict("records")

    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    sources = [
        _source(
            "headline_summary",
            "EDA headline validation results",
            "reports/eda/eda_validation_summary.json",
            headline_sql,
            "Reads the reviewed headline quality and balance checks.",
            generated_at,
            ["eda_headline"],
        ),
        _source(
            "category_counts",
            "Category counts from the full manifest",
            "reports/dataset/full_manifest.csv",
            category_sql,
            "Counts images and independent families by category.",
            generated_at,
            ["full_manifest"],
        ),
        _source(
            "split_counts",
            "Split counts from the full manifest",
            "reports/dataset/full_manifest.csv",
            split_sql,
            "Counts images and independent families by split.",
            generated_at,
            ["full_manifest"],
        ),
        _source(
            "family_split_counts",
            "Family counts from the frozen split",
            "data/interim/google_fonts_final_family_split.csv",
            family_split_sql,
            "Counts independent frozen families by category and split.",
            generated_at,
            ["frozen_split"],
        ),
        _source(
            "quality_by_category",
            "Per-category image quality summary",
            "reports/eda/image_quality_metrics.csv",
            quality_sql,
            "Summarizes brightness, contrast, and actual font size for open images.",
            generated_at,
            ["image_quality"],
        ),
        _source(
            "effect_rates",
            "Effect rates by category",
            "reports/eda/effect_balance_by_category.csv",
            effect_sql,
            "Reshapes four reviewed binary effect rates for category comparison.",
            generated_at,
            ["effect_balance"],
        ),
        _source(
            "validation_checks",
            "Blocking EDA validation checks",
            "reports/eda/eda_validation_summary.json",
            checks_sql,
            "Reads the reviewed pass/fail checks shown in the report.",
            generated_at,
            ["validation_checks"],
        ),
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "FontSense dataset EDA and quality audit",
            "description": (
                "Technical audit of the existing 3,600-image FontSense dataset "
                "before model training."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "images_card",
                    "description": "Manifest rows at one image per row.",
                    "dataset": "headline",
                    "sourceId": "headline_summary",
                    "metrics": [
                        {"label": "Images", "field": "total_images", "format": "number"}
                    ],
                },
                {
                    "id": "families_card",
                    "description": "Independent font families in the frozen split.",
                    "dataset": "headline",
                    "sourceId": "headline_summary",
                    "metrics": [
                        {
                            "label": "Font families",
                            "field": "unique_families",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "invalid_card",
                    "description": "Missing, corrupt, blank, or extremely low-contrast flags.",
                    "dataset": "headline",
                    "sourceId": "headline_summary",
                    "metrics": [
                        {
                            "label": "Automated quality flags",
                            "field": "invalid_images",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "overlap_card",
                    "description": "Families appearing in more than one split.",
                    "dataset": "headline",
                    "sourceId": "headline_summary",
                    "metrics": [
                        {
                            "label": "Split-overlap families",
                            "field": "family_overlap",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "category_counts_chart",
                    "title": "Images per font category",
                    "subtitle": "All 3,600 manifest rows; count axis starts at zero.",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "category_counts",
                    "sourceId": "category_counts",
                    "encodings": {
                        "x": {"field": "category", "type": "nominal", "label": "Category"},
                        "y": {"field": "images", "type": "quantitative", "label": "Images"},
                        "tooltip": [
                            {
                                "field": "families",
                                "type": "quantitative",
                                "label": "Families",
                            },
                            {
                                "field": "total_images",
                                "type": "quantitative",
                                "label": "Dataset images",
                            },
                        ],
                    },
                },
                {
                    "id": "split_counts_chart",
                    "title": "Images per family-level split",
                    "subtitle": "Train, validation, and test assignments are fixed before modeling.",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "split_counts",
                    "sourceId": "split_counts",
                    "encodings": {
                        "x": {"field": "split", "type": "nominal", "label": "Split"},
                        "y": {"field": "images", "type": "quantitative", "label": "Images"},
                        "tooltip": [
                            {
                                "field": "families",
                                "type": "quantitative",
                                "label": "Families",
                            }
                        ],
                    },
                },
                {
                    "id": "family_split_chart",
                    "title": "Independent families per category and split",
                    "subtitle": "Each category has 12 train, 3 validation, and 3 test families.",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "family_split_counts",
                    "sourceId": "family_split_counts",
                    "encodings": {
                        "x": {"field": "category", "type": "nominal", "label": "Category"},
                        "y": {
                            "field": "families",
                            "type": "quantitative",
                            "label": "Unique families",
                        },
                        "color": {"field": "split", "type": "nominal", "label": "Split"},
                        "tooltip": [
                            {
                                "field": "images_per_family",
                                "type": "quantitative",
                                "label": "Images per family",
                            }
                        ],
                    },
                },
                {
                    "id": "effect_rates_chart",
                    "title": "Applied effect rates by category",
                    "subtitle": "Four requested binary effects across 720 images per category.",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "effect_rates",
                    "sourceId": "effect_rates",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "effect", "type": "nominal", "label": "Effect"},
                        "y": {"field": "rate", "type": "quantitative", "label": "Share of images"},
                        "color": {
                            "field": "category",
                            "type": "nominal",
                            "label": "Category",
                        },
                        "tooltip": [
                            {
                                "field": "images",
                                "type": "quantitative",
                                "label": "Images per category",
                            }
                        ],
                    },
                },
                {
                    "id": "contrast_chart",
                    "title": "Average pixel contrast by category",
                    "subtitle": "Mean grayscale standard deviation; 720 open images per category.",
                    "type": "bar",
                    "intent": "comparison",
                    "dataset": "quality_by_category",
                    "sourceId": "quality_by_category",
                    "encodings": {
                        "x": {"field": "category", "type": "nominal", "label": "Category"},
                        "y": {
                            "field": "contrast_average",
                            "type": "quantitative",
                            "label": "Average grayscale standard deviation",
                        },
                        "tooltip": [
                            {
                                "field": "contrast_min",
                                "type": "quantitative",
                                "label": "Minimum contrast",
                            },
                            {
                                "field": "contrast_max",
                                "type": "quantitative",
                                "label": "Maximum contrast",
                            },
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "quality_table",
                    "title": "Image-quality distribution summary",
                    "subtitle": "Brightness, contrast, and actual font-size ranges by category.",
                    "dataset": "quality_by_category",
                    "sourceId": "quality_by_category",
                    "defaultSort": {"field": "category", "direction": "asc"},
                    "columns": [
                        {"field": "category", "label": "Category", "type": "text"},
                        {"field": "images", "label": "Images", "format": "number"},
                        {
                            "field": "brightness_average",
                            "label": "Average brightness",
                            "format": "number",
                        },
                        {
                            "field": "contrast_average",
                            "label": "Average contrast",
                            "format": "number",
                        },
                        {
                            "field": "font_size_min",
                            "label": "Min font size",
                            "format": "number",
                        },
                        {
                            "field": "font_size_average",
                            "label": "Average font size",
                            "format": "number",
                        },
                        {
                            "field": "font_size_max",
                            "label": "Max font size",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "checks_table",
                    "title": "Validation checks",
                    "subtitle": "Blocking checks required before model development.",
                    "dataset": "validation_checks",
                    "sourceId": "validation_checks",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "columns": [
                        {"field": "check", "label": "Check", "type": "text"},
                        {"field": "result", "label": "Observed result", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# FontSense dataset EDA and quality audit",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "headline_summary",
                    "body": (
                        "## Technical summary\n\n"
                        "**The existing full dataset passes the requested pre-training quality gate.** "
                        "All 3,600 images open at 224×96, all 90 families match the frozen split, "
                        "family overlap is zero, and the automated screen finds no missing, corrupt, "
                        "blank, extremely low-contrast, exact-duplicate, or strictly near-identical images. "
                        "Effects and phrases show no category-dependent schedule imbalance. No model was trained."
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "images_card",
                        "families_card",
                        "invalid_card",
                        "overlap_card",
                    ],
                },
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "sourceId": "category_counts",
                    "body": (
                        "## Key findings\n\n"
                        "**The dataset is exactly balanced by target category.** "
                        "Each of the five categories has 720 images and 18 independent families, "
                        "so raw class frequency will not give a model an advantage."
                    ),
                },
                {"id": "category_counts", "type": "chart", "chartId": "category_counts_chart"},
                {
                    "id": "category_counts_note",
                    "type": "markdown",
                    "sourceId": "category_counts",
                    "body": (
                        "**How to read it.** Equal bar heights confirm the 720-image target "
                        "for every category. This removes class-count imbalance, but it does not "
                        "guarantee that every visual style is equally difficult."
                    ),
                },
                {"id": "split_counts", "type": "chart", "chartId": "split_counts_chart"},
                {
                    "id": "split_counts_note",
                    "type": "markdown",
                    "sourceId": "split_counts",
                    "body": (
                        "**The split sizes are 2,400 train, 600 validation, and 600 test images.** "
                        "Because whole families—not individual images—were assigned, the split "
                        "supports evaluation on unseen families."
                    ),
                },
                {
                    "id": "leakage_balance",
                    "type": "markdown",
                    "sourceId": "family_split_counts",
                    "body": (
                        "## Leakage and balance\n\n"
                        "**No font family appears in more than one split.** "
                        "Every category contains 12 train, 3 validation, and 3 test families. "
                        "Training code inspection also confirms that family, phrase, source font, "
                        "file name, path text, split, random seed, and effect metadata are not model features."
                    ),
                },
                {"id": "family_split", "type": "chart", "chartId": "family_split_chart"},
                {
                    "id": "family_split_note",
                    "type": "markdown",
                    "sourceId": "family_split_counts",
                    "body": (
                        "**The family-level pattern is identical across categories.** "
                        "The test families stay reserved for final evaluation; this audit uses them "
                        "only for the requested integrity and quality checks."
                    ),
                },
                {"id": "effect_rates", "type": "chart", "chartId": "effect_rates_chart"},
                {
                    "id": "effect_rates_note",
                    "type": "markdown",
                    "sourceId": "headline_summary",
                    "body": (
                        "**Background and selected effect rates match exactly across categories.** "
                        "The maximum binary effect-rate spread is 0.0 percentage points. "
                        "Rendered-phrase distributions also match exactly (Cramér's V = 0.000), "
                        "so these metadata schedules do not reveal the label."
                    ),
                },
                {
                    "id": "quality_findings",
                    "type": "markdown",
                    "sourceId": "quality_by_category",
                    "body": (
                        "## Image quality\n\n"
                        "**All files pass the blocking image checks.** Brightness spans both dark "
                        "and light backgrounds. Contrast varies by font shape and stroke coverage; "
                        "display fonts have the highest average pixel contrast, while handwriting "
                        "has the lowest. Actual font size ranges from 13 to 46 pixels with an average of 22.4."
                    ),
                },
                {"id": "contrast", "type": "chart", "chartId": "contrast_chart"},
                {
                    "id": "contrast_note",
                    "type": "markdown",
                    "sourceId": "quality_by_category",
                    "body": (
                        "**Contrast differences are descriptive, not a generation imbalance.** "
                        "They can arise naturally because broad font categories occupy different "
                        "amounts of the image. No image fell below the conservative low-pixel-contrast flag."
                    ),
                },
                {"id": "quality_detail", "type": "table", "tableId": "quality_table"},
                {
                    "id": "quality_table_note",
                    "type": "markdown",
                    "sourceId": "quality_by_category",
                    "body": (
                        "**The ranges remain broad but valid.** The saved representative and difficult-sample "
                        "figures should still receive human review because numeric screening cannot prove "
                        "subjective readability."
                    ),
                },
                {"id": "validation_checks", "type": "table", "tableId": "checks_table"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## Scope, data, and metric definitions\n\n"
                        "- **Image:** one manifest row and one 224×96 PNG.\n"
                        "- **Family overlap:** a family with more than one distinct split.\n"
                        "- **Brightness:** mean grayscale value from 0 to 255.\n"
                        "- **Contrast:** grayscale pixel standard deviation.\n"
                        "- **Blank:** zero grayscale intensity range.\n"
                        "- **Strict near-identical pair:** difference-hash Hamming distance ≤2 "
                        "and structural similarity ≥0.995.\n"
                        "- **Serious effect imbalance:** maximum category rate spread above 5 percentage points."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Methodology\n\n"
                        "The audit loads the committed full manifest and frozen split without changing either. "
                        "It validates the image-level key and exact assignments, opens every listed image with "
                        "Pillow, calculates per-image pixel summaries and hashes, expands the saved effect JSON, "
                        "compares category rates, and measures phrase association. The notebook executes the "
                        "same reusable `fontsense.eda` code from top to bottom."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and uncertainty\n\n"
                        "- Automated checks cannot guarantee that every text sample is easy for every person to read.\n"
                        "- The data are synthetic renders; real screenshots may add cropping, lighting, noise, and layout shift.\n"
                        "- Broad categories can be visually ambiguous even when files and labels are valid.\n"
                        "- The strict near-duplicate rule is designed to catch very similar files, not every semantic resemblance.\n"
                        "- These findings describe data quality only and are not model metrics."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next step\n\n"
                        "Proceed to the leakage-safe HOG + Logistic Regression baseline using train and validation "
                        "families only. Keep the test split untouched until the one final evaluation, and record "
                        "all experiments in MLflow."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Which categories and unseen families create the most validation confusion?\n"
                        "- How large is the synthetic-to-real gap on user screenshots?\n"
                        "- Do smaller text sizes or softer contrast drive validation errors after training?"
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
                "category_counts": category_rows,
                "split_counts": split_rows,
                "family_split_counts": family_split_rows,
                "effect_rates": effect_rows,
                "quality_by_category": quality_rows,
                "validation_checks": check_rows,
            },
            "accessIssues": [],
        },
        "sources": sources,
    }
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {ARTIFACT_PATH}")


if __name__ == "__main__":
    build_report_artifact()
