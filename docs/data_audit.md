# FontSense data audit

## Decision

**Final M8C3 Data Gate decision: GREEN after manual CV evidence review.**

The teacher self-check is intentionally kept unchanged and reports a suggested
YELLOW status because it flags a constant manifest field and requires manual
computer-vision checks. Those warnings are not hidden. The CV-specific checks
below were completed against all 3,600 local images and saved in
`reports/data_gate_self_check/manual_validation.json`.

Green means the existing data evidence is sufficient to support the work
already completed. It does not mean the dataset is perfect or that the model is
guaranteed to generalise to real screenshots.

## Repository evidence inventory

| Evidence | Repository path |
|---|---|
| Full image manifest | `reports/dataset/full_manifest.csv` |
| Frozen family split | `data/interim/google_fonts_final_family_split.csv` |
| Generation configuration | `config/full_dataset.json` |
| EDA report | `reports/eda/eda_summary_report.html` |
| Missing, corrupt, blank, duplicate, and near-duplicate checks | `reports/eda/eda_validation_summary.json`, `reports/eda/image_quality_metrics.csv`, `reports/eda/suspicious_image_pairs.csv` |
| CNN preprocessing and train-only augmentation | `src/fontsense/train_cnn.py`, `src/fontsense/inference.py` |
| HOG experiment report | `reports/baseline/baseline_summary_report.html` |
| CNN experiment report | `reports/cnn/cnn_experiment_report.html` |
| Final held-out evaluation | `reports/final_evaluation/final_evaluation_report.html` |
| Frozen checkpoint | `artifacts/cnn/cnn_model.pt` |
| Frozen hashes and pre-test decisions | `reports/final_evaluation/pre_test_freeze.json` |
| Gradio application | `app.py` |
| Colab demonstration | `notebooks/07_colab_demo.ipynb` |

The frozen checkpoint SHA-256 is
`c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`.
The frozen split SHA-256 is
`f6cdd858449a4143993b051e9de83578cd423544135dbf6cc17f004359d2e17b`,
and the full manifest SHA-256 is
`93afae99ea6ac7dc65e8cf03a1e5743f9581c189f1feaf115ad7439fad58d69c`.
All three values are recorded before test access in
`reports/final_evaluation/pre_test_freeze.json` and rechecked in
`reports/data_gate_self_check/manual_validation.json`.

## Context and counts

FontSense predicts one of five broad font categories from a cropped rendered
Latin-text image. One manifest row is one image. The full dataset has 3,600
images from 90 families, with exactly 40 images per family. Each category has
720 images and 18 families. Evidence:
`data/README.md`, `reports/dataset/full_manifest.csv`, and
`reports/eda/eda_validation_summary.json`.

The frozen family split contains 2,400 training images from 60 families, 600
validation images from 15 families, and 600 test images from 15 families.
Within each category this is 12 train, 3 validation, and 3 test families.
Evidence: `data/interim/google_fonts_final_family_split.csv`,
`reports/dataset/full_manifest.csv`, and
`reports/data_gate/split_summary.csv`.

## Source, labels, and licences

Source font files came from the official Google Fonts repository. Category
labels came from official Google Fonts metadata. Latin support, the actual
font file, source URL, licence code, and Pillow/font validation status were
recorded before balanced selection. Failed or unusable audit rows were not
silently relabelled. Evidence: `scripts/download_google_fonts.py`,
`data/interim/google_fonts_manifest.csv`, and
`data/interim/google_fonts_audit_summary.csv`.

The selected set contains licence codes recorded as `OFL`, `APACHE2`, or `UFL`.
This audit preserves per-family attribution instead of claiming one licence for
all fonts. Evidence: `data/interim/google_fonts_final_family_split.csv`.

## Image dimensions and quality

Generated source images are 224 x 96 pixels. The full audit checked every
manifest path and opened every image. It found no missing, corrupted, blank, or
wrong-dimension images. Brightness ranged from 11.2568 to 251.5986 with a median
of 234.8136; contrast standard deviation ranged from 7.5245 to 76.2661 with a
median of 26.6531. Evidence:
`reports/eda/eda_validation_summary.json` and
`reports/eda/image_quality_metrics.csv`.

Exact SHA-256 file hashes produced zero duplicate groups. The strict
near-duplicate screen produced zero suspicious pairs; its rule is a 16 x 16
difference-hash Hamming distance of at most 2 plus structural similarity of at
least 0.995. Evidence: `src/fontsense/eda.py`,
`reports/eda/eda_validation_summary.json`, and
`reports/eda/suspicious_image_pairs.csv`.

Automated checks cannot prove that every image is typographically ideal. The
preview and full contact sheets provide human-review evidence, while the
synthetic-to-real gap remains an accepted limitation. Evidence:
`reports/preview/contact_sheets/`,
`reports/dataset/contact_sheets/`, and
`reports/data_gate/issue_log.csv`.

## Split integrity and leakage controls

The split unit is the font family, not the image. The full manifest matches the
frozen assignments and no family appears in more than one split. This directly
controls the main leakage risk: memorising family-specific shapes and then
being evaluated on the same family. Evidence:
`src/fontsense/split.py`, `tests/test_split.py`,
`data/interim/google_fonts_final_family_split.csv`, and
`reports/eda/eda_validation_summary.json`.

File names, paths, family names, rendered phrases, split labels, source-font
paths, random seeds, and effect metadata are excluded from model inputs. HOG is
calculated from grayscale pixels; the CNN receives only a normalised grayscale
tensor. Evidence: `src/fontsense/features.py`,
`src/fontsense/train_hog.py`, `src/fontsense/train_cnn.py`, and the
`model_feature_audit` section of
`reports/eda/eda_validation_summary.json`.

The saved generation schedule is category-independent. The maximum binary
effect-rate spread across categories is 0.0, and the phrase/category
association check reports Cramer's V of 0.0. Evidence:
`config/full_dataset.json`,
`reports/dataset/full_effect_balance.csv`, and the `effects` and
`phrase_balance` sections of `reports/eda/eda_validation_summary.json`.

## Preprocessing and augmentation boundary

The final CNN converts images to grayscale, resizes them to 112 x 48, converts
them to a tensor, and normalises with mean 0.5 and standard deviation 0.5.
Validation and test use deterministic preprocessing. Evidence:
`src/fontsense/train_cnn.py`, `src/fontsense/inference.py`,
`artifacts/cnn/cnn_metadata.json`, and
`reports/preprocessing_manifest.json`.

Random affine and sharpness augmentation is enabled only for the training
transform. It is absent from validation and inference transforms. The
automated boundary test checks this explicitly. Evidence:
`src/fontsense/train_cnn.py`,
`tests/test_train_cnn.py`, and
`reports/cnn/cnn_validation_summary.json`.

For HOG, preprocessing and classification fitting use only 2,400 training
rows; validation is used for comparison. For the CNN, fitting and augmentation
use only train, while validation is used for early stopping and model
selection. Evidence:
`reports/baseline/baseline_validation_summary.json`,
`reports/cnn/cnn_validation_summary.json`,
`src/fontsense/train_hog.py`, and
`src/fontsense/train_cnn.py`.

## Test boundary

Before test access, the checkpoint, preprocessing, class order, split and
manifest hashes, seed, and 0.60 threshold were frozen. The threshold was chosen
from validation predictions only. The selected CNN was evaluated once on all
600 held-out test images, and the evaluation receipt forbids rerunning for
tuning. Evidence:
`reports/final_evaluation/pre_test_freeze.json`,
`reports/final_evaluation/evaluation_receipt.json`,
`reports/final_evaluation/final_test_metrics.json`, and
`tests/test_final_evaluation.py`.

## Remaining risks

The main unresolved limitation is distribution shift. Synthetic Google Fonts
renders are cleaner and more controlled than real screenshots, photographs,
and design layouts. The model can therefore make confident mistakes on
unfamiliar real inputs. This risk is documented rather than treated as fixed.
Evidence: `reports/final_evaluation/final_evaluation_report.html`,
`docs/m8c3_data_gate_review.md`, and
`reports/data_gate/issue_log.csv`.

The generic self-check's YELLOW suggestion and its manual CV item remain in the
saved output. The final Green decision is based on the separate complete-image
validation, evidence-path check, split audit, augmentation inspection, and test
boundary records, not on editing the validator to obtain a preferred colour.

The remaining validator warning is `image_size`, which is constant at the raw
render size by design. Keeping it in the manifest documents source-image
dimensions; it is explicitly excluded from model features in
`reports/preprocessing_manifest.json`. No change is required for modeling.
