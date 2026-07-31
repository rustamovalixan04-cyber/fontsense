# M8C3 Data Gate review

## Where did the data come from?

Font files came from the official Google Fonts repository. Official metadata
provided the broad category labels, and each audit row records its source URL,
licence code, Latin support, local font path, and validation result. The image
dataset was then rendered locally from the accepted fonts. Evidence:
`scripts/download_google_fonts.py`,
`data/interim/google_fonts_manifest.csv`, and
`config/full_dataset.json`.

## What is predicted?

The target is one of five broad categories: display, handwriting, monospace,
sans serif, or serif. One sample is a cropped Latin-text image. Exact font
family recognition is outside the project scope. Evidence:
`reports/dataset/full_manifest.csv`,
`artifacts/cnn/cnn_metadata.json`, and `app.py`.

## Why is the split family-based?

Images rendered with one font family share distinctive shapes. A random
image-level split would place near-related examples of the same family in both
fitting and evaluation data, making performance look better than true
generalisation to unseen fonts. The frozen split therefore keeps every family
in exactly one of train, validation, or test. Evidence:
`src/fontsense/split.py`,
`data/interim/google_fonts_final_family_split.csv`, and
`tests/test_split.py`.

## What is the main leakage risk?

The main risk is family leakage. Other risks are using paths or family metadata
as features, fitting preprocessing on non-training data, applying random
augmentation to validation/test, or using test results to tune the model or
threshold. The controls and their current status are listed in
`reports/data_gate/issue_log.csv` and supported by
`reports/preprocessing_manifest.json`.

## Where is preprocessing implemented?

CNN training preprocessing is in `src/fontsense/train_cnn.py`. Frozen
inference preprocessing is in `src/fontsense/inference.py`. Both use grayscale
112 x 48 images and mean/std 0.5 normalisation. HOG pixel feature extraction is
in `src/fontsense/features.py`. The reusable configuration and fitting boundary
are summarised in `reports/preprocessing_manifest.json`.

## What is the final Data Gate status?

The final project decision is **GREEN after manual CV evidence review**. The
generic teacher validator suggests YELLOW because it correctly leaves CV image
checks for manual confirmation and warns about a constant manifest field. The
full 3,600-image check, evidence-path validation, zero-overlap check, and
training/test boundary records pass. Evidence:
`reports/data_gate_self_check/data_gate_self_check.md`,
`reports/data_gate_self_check/manual_validation.json`, and
`docs/data_audit.md`.

## What limitations remain?

The dataset uses controlled synthetic renders. Real photographs, screenshots,
cluttered layouts, and unfamiliar fonts can differ, so confident mistakes
remain possible. Category labels are broad metadata labels and exact family
recognition is not supported. Automated readability checks do not replace
human inspection. These are accepted limitations, not fabricated fixes.
Evidence: `reports/final_evaluation/final_evaluation_report.html`,
`data/README.md`, and `reports/data_gate/issue_log.csv`.
