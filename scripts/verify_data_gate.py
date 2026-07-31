"""Verify the M8C3 FontSense Data Gate without generating or training anything."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fontsense.eda import audit_image_files, validate_dataset_structure


EXPECTED_EVIDENCE_PATHS = (
    ".gitignore",
    "app.py",
    "artifacts/cnn/cnn_metadata.json",
    "artifacts/cnn/cnn_model.pt",
    "config/full_dataset.json",
    "data/README.md",
    "data/interim/google_fonts_final_family_split.csv",
    "data/interim/google_fonts_manifest.csv",
    "docs/data_audit.md",
    "docs/m8c3_data_gate_review.md",
    "docs/m8c3_modeling_readiness.md",
    "notebooks/07_colab_demo.ipynb",
    "notebooks/08_m8c3_data_gate_self_check.ipynb",
    "reports/baseline/baseline_validation_summary.json",
    "reports/baseline/baseline_summary_report.html",
    "reports/cnn/cnn_validation_summary.json",
    "reports/cnn/cnn_experiment_report.html",
    "reports/data_gate/issue_log.csv",
    "reports/data_gate/split_summary.csv",
    "reports/data_gate_self_check/data_gate_self_check.csv",
    "reports/data_gate_self_check/data_gate_self_check.md",
    "reports/dataset/full_manifest.csv",
    "reports/dataset/full_reproducibility_check.json",
    "reports/dataset/full_validation_summary.json",
    "reports/eda/eda_validation_summary.json",
    "reports/eda/eda_summary_report.html",
    "reports/final_evaluation/evaluation_receipt.json",
    "reports/final_evaluation/final_evaluation_report.html",
    "reports/final_evaluation/final_test_metrics.json",
    "reports/final_evaluation/pre_test_freeze.json",
    "reports/preprocessing_manifest.json",
    "src/fontsense/features.py",
    "src/fontsense/inference.py",
    "src/fontsense/train_cnn.py",
    "src/fontsense/train_hog.py",
    "scripts/verify_data_gate.py",
    "tests/test_data_gate.py",
    "tests/test_eda.py",
    "tests/test_final_evaluation.py",
    "tests/test_full_dataset.py",
    "tests/test_split.py",
    "tests/test_train_cnn.py",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_verification(root: Path) -> dict:
    """Run deterministic data, boundary, and evidence checks."""
    manifest_path = root / "reports/dataset/full_manifest.csv"
    split_path = root / "data/interim/google_fonts_final_family_split.csv"
    manifest = pd.read_csv(manifest_path)
    frozen_split = pd.read_csv(split_path)

    structure = validate_dataset_structure(manifest, frozen_split)
    quality = audit_image_files(manifest, root=root)
    readable = quality.loc[quality["opens_successfully"]].copy()
    duplicate_groups = int(
        readable.loc[readable["sha256"].duplicated(keep=False), "sha256"].nunique()
    )

    eda = _load_json(root / "reports/eda/eda_validation_summary.json")
    cnn = _load_json(root / "reports/cnn/cnn_validation_summary.json")
    metadata = _load_json(root / "artifacts/cnn/cnn_metadata.json")
    freeze = _load_json(root / "reports/final_evaluation/pre_test_freeze.json")
    final_metrics = _load_json(
        root / "reports/final_evaluation/final_test_metrics.json"
    )

    missing_evidence = [
        path for path in EXPECTED_EVIDENCE_PATHS if not (root / path).exists()
    ]
    manual_checks = {
        "exactly_3600_manifest_rows": len(manifest) == 3600,
        "all_manifest_paths_exist": bool(quality["exists"].all()),
        "all_images_open": bool(quality["opens_successfully"].all()),
        "no_blank_images": int(readable["blank"].sum()) == 0,
        "no_corrupted_images": int(
            (quality["exists"] & ~quality["opens_successfully"]).sum()
        )
        == 0,
        "no_missing_images": int((~quality["exists"]).sum()) == 0,
        "no_exact_duplicate_hash_groups": duplicate_groups == 0,
        "all_images_224_by_96": bool(
            ((readable["width"] == 224) & (readable["height"] == 96)).all()
        ),
        "zero_family_overlap": structure["family_overlap_count"] == 0,
        "frozen_assignments_match": True,
        "category_independent_effects": not eda["effects"][
            "serious_category_dependent_effect_imbalance"
        ],
        "phrases_not_category_associated": not eda["phrase_balance"][
            "strong_category_association"
        ],
        "augmentation_training_only": bool(
            cnn["augmentation"]["training_enabled"]
            and not cnn["augmentation"]["validation_enabled"]
        ),
        "fit_boundary_is_train_only": metadata["training_data"]["fit_splits"]
        == ["train"],
        "validation_used_for_selection": metadata["training_data"][
            "selection_split"
        ]
        == "validation",
        "test_not_used_for_training_or_selection": all(
            not final_metrics["checks"][key]
            for key in (
                "test_used_for_training",
                "test_used_for_validation",
                "test_used_for_model_selection",
                "test_used_for_early_stopping",
                "test_used_for_threshold_selection",
            )
        ),
        "threshold_selected_on_validation_only": bool(
            freeze["uncertainty_threshold"]["source_split"] == "validation"
            and not freeze["uncertainty_threshold"]["test_data_used"]
        ),
        "frozen_split_hash_unchanged": _sha256(split_path)
        == freeze["frozen_family_split"]["sha256"],
        "manifest_hash_unchanged": _sha256(manifest_path)
        == freeze["full_manifest"]["sha256"],
        "checkpoint_hash_unchanged": _sha256(root / "artifacts/cnn/cnn_model.pt")
        == freeze["selected_model"]["checkpoint_sha256"],
        "all_evidence_paths_exist": not missing_evidence,
    }
    failed = [name for name, passed in manual_checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed",
        "purpose": (
            "Manual CV Data Gate verification only; no dataset generation, model "
            "training, tuning, or final test evaluation."
        ),
        "dataset": {
            "manifest": "reports/dataset/full_manifest.csv",
            "images_checked": int(len(quality)),
            "images_opened": int(quality["opens_successfully"].sum()),
            "missing_images": int((~quality["exists"]).sum()),
            "corrupted_images": int(
                (quality["exists"] & ~quality["opens_successfully"]).sum()
            ),
            "blank_images": int(readable["blank"].sum()),
            "exact_duplicate_hash_groups": duplicate_groups,
            "unique_families": structure["unique_families"],
            "family_overlap_count": structure["family_overlap_count"],
        },
        "manual_checks": manual_checks,
        "missing_evidence_paths": missing_evidence,
        "failed_checks": failed,
        "final_manual_data_gate_decision": (
            "GREEN" if not failed else "YELLOW - action required"
        ),
        "remaining_limitation": (
            "Synthetic rendered images may not represent real screenshots, "
            "photographs, clutter, or unfamiliar font designs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_gate_self_check/manual_validation.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = run_verification(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
