# FontSense product V3: larger data, measured quality

## Purpose and boundaries

The owner requested a standalone commercial product and up to 500,000 images.
This pipeline expands the five-category Latin-text task. It does not identify
exact font names, guarantee revenue, or establish worldwide accuracy. V1 and V2
datasets, splits, checkpoints, and evaluation reports remain preserved.

The running app remains V1. V3 is a development candidate, not a silently
replaced production model. AI assistance supported the code, checks, execution,
and documentation; the owner remains responsible for product decisions.

The complete corpus now passed generation and independent archive checks:
500,000 images opened, zero blank/corrupt/missing images, zero exact pixel
duplicates after one deterministic replacement, and zero family overlap.
One hundred sampled training images reproduce byte-for-byte. The JPEG shards
and compressed metadata use 3,651,968,603 bytes; the SQLite index uses
462,598,144 bytes (about 4.1 GB combined). These are measured data results,
not model accuracy results.

## Sources and selected families

Official source: `https://github.com/google/fonts`, pinned to commit
`e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7`.

The audit uses official category/subset metadata, downloads OFL or Apache 2
fonts and their licence texts, verifies required Latin character mappings with
fontTools, and opens/renders each usable font with Pillow. This is technical
provenance checking, not a substitute for legal review before redistribution.
Do not discard licence notices or commit the downloaded font collection.

Evidence:

- `data/v3/font_candidates.csv`: 1,876 eligible candidates.
- `data/v3/font_audit.csv`: 1,847 technically usable fonts, 29 actual failures.
- `data/v3/font_exclusions.csv`: explicit selection exclusions, not invented
  categories or hidden download failures.
- `data/v3/frozen_readable_family_split.csv`: 1,641 selected families.
- `reports/v3/family_split_summary.json`: counts and frozen split hash.

| Category | Train families | Validation families | Test families | Total |
|---|---:|---:|---:|---:|
| Display | 361 | 47 | 46 | 454 |
| Handwriting | 195 | 26 | 26 | 247 |
| Monospace | 27 | 6 | 5 | 38 |
| Sans serif | 466 | 60 | 59 | 585 |
| Serif | 251 | 33 | 33 | 317 |
| Total | 1,300 | 172 | 169 | 1,641 |

All classes receive 100,000 images, with 70,000/15,000/15,000 images per
class in train/validation/test. Families within each class/fold are cycled
evenly. Equal class counts do not mean equal independent family diversity;
monospace is still the smallest family pool.

## Quality findings that changed selection

Different Google Fonts family names can share identical default Latin glyphs.
Exact probe matches are deduplicated. Whole alias groups crossing historical
V2 folds are quarantined rather than moved into a different fold. Protected V1
test families and exact aliases are excluded. Retained V2 families keep their
old folds; new families are assigned deterministically with seed 42.

`reports/v3/historical_alias_risks.csv` records three affected groups containing
six historical V2 families. Historical metrics are preserved, but a name-only
family split did not remove that alias risk. This check does not prove that all
visually related or near-identical typefaces have been removed.

An assistant visual inspection of the first contact sheets found unreadable
barcode and redaction fonts. The pinned source descriptions confirm those
purposes (`ofl/librebarcode39extended/DESCRIPTION.en_us.html`,
`ofl/flowcircular/DESCRIPTION.en_us.html`, and
`ofl/redacted/DESCRIPTION.en_us.html`). The readable-text track excludes all
Libre Barcode, Flow, Redacted, and Wavefont families. Eleven selected utility
families were removed before training. Their official labels are not changed.

The first draft stopped intentionally at 366,000 images. Its split remains in
`data/v3/frozen_family_split.csv`; its bulk files are preserved under the ignored
`data/v3/cache/pre_review/` directory. They are not product-training input.

The readable draft then exposed a Pillow/FreeType outline-stroking failure on
Rubik Pixels at seed 14629533 (`OSError: array allocation size too large`). A
bounded one-pixel alpha-mask dilation replaces the failing outline operation;
the same sample now renders. A regression test and an eight-seed stress check
of all 1,641 selected fonts passed (13,128 renders). See
`reports/v3/font_render_stress_check.json`. A failed shard now pauses pending
work and records its family, sample ID, and seed instead of leaving stale
progress while other workers continue.

## Rendering and storage

`config/v3/dataset.json` defines seed 42, RGB 320x128 JPEGs, bounded sizes,
readable short English word combinations, light/dark backgrounds, mild spacing,
supported variable weights, small translation/rotation/shear, and mild
blur/noise/compression. The renderer receives no category label. Effect seeds
are paired across categories; family metadata never becomes an image feature.

`src/fontsense/v3_dataset.py` writes deterministic 1,000-image TAR shards and
compressed CSV metadata. Shards avoid 500,000 separate Windows files. Every
JPEG is opened and checked during generation; archive and metadata checksums
are verified afterward. A SQLite manifest validates counts, family assignment,
exact decoded-pixel duplicates, and category effect rates. Exact duplicate
renders receive deterministic replacement seeds, with a bounded retry count.

Expected compressed corpus size is about 3–6 GB, plus the manifest, downloaded
fonts, and local environments/checkpoints. Budget about 10–20 GB for this pass,
including the preserved rejected draft; leave at least 30 GiB free. The pilot
and GPU preflight reports contain measurements rather than storage guarantees.

Bulk data, environments, caches, MLflow storage, and temporary resume state
are ignored by Git. Small configs, audits, checksums, inventories, reports,
contact sheets, tests, and the eventual compact best checkpoint can be tracked.

## Reproduce or resume

Run from the FontSense repository root. The measured renderer used Python
3.12.14; `requirements-product-data.txt` pins its rendering stack. Existing
`.venv` setup can be reused. For a fresh rendering environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-product-data.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

The pinned official metadata checkout must exist at
`data/v2/cache/google-fonts`. If missing, create a metadata-only sparse checkout:

```powershell
git clone --filter=blob:none --no-checkout https://github.com/google/fonts.git data/v2/cache/google-fonts
git -C data/v2/cache/google-fonts sparse-checkout set --no-cone '/ofl/*/METADATA.pb' '/apache/*/METADATA.pb' '/ufl/*/METADATA.pb'
git -C data/v2/cache/google-fonts checkout --detach e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7
.\.venv\Scripts\python.exe -u -m fontsense.v3_data --workers 8
.\.venv\Scripts\python.exe -u scripts/check_product_font_rendering.py
.\.venv\Scripts\python.exe -u -m fontsense.v3_dataset --workers 4
.\.venv\Scripts\python.exe -u scripts/verify_product_dataset.py
.\.venv\Scripts\python.exe scripts/create_product_contact_sheets.py
```

Do not run clone commands over an existing checkout. Auditing reuses verified
downloads. Existing frozen reviewed assignments cannot be silently overwritten.
Generation reuses completed shards only when its config, split, renderer,
replacement seeds, and archive/metadata hashes match. Rendering-library versions
and platform can affect pixel identity; keep the measured stack for exact
reproduction. Full corpus generation is never part of the normal test suite.

The raw SQLite-file SHA verifies the current local index. Rebuilding that index
can change SQLite's page-layout bytes without changing a sample. Training and
comparison use a separate stable identity built from sorted archive/metadata
hashes, the generation config, and the family split, so a verified equivalent
index does not invalidate an otherwise matching training resume. Changed sample
bytes or labels do invalidate it.

For a graceful generation stop, create `data/v3/STOP`; current shards finish.
Before an intentional resume, remove that one stop marker, not the data.
`reports/v3/progress.json` is live progress. Only a `passed` result in
`reports/v3/dataset_validation_summary.json` means the full data gate passed.

## GPU training contract

The original CPU environment is preserved. An isolated `.venv-gpu` used the
verified matching PyTorch 2.13.0 / torchvision 0.28.0 CUDA 13.0 wheels from the
official PyTorch index on this RTX 3050 laptop. No driver change or paid cloud
compute is needed for this measured local setup:

```powershell
python -m venv --system-site-packages .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-product-gpu.txt
.\.venv-gpu\Scripts\python.exe -m pip install -e .
.\.venv-gpu\Scripts\python.exe -u -m fontsense.v3_train
```

This GPU environment is intentionally hardware-specific; the CPU CI setup is
unchanged. A GPU preflight used disposable weights and is not a performance run.
The measured batch-256 preflight reserved about 1.4 GiB of GPU memory. Training
uses grayscale 192x64, normalization mean/std 0.5, the existing width-24 CNN,
Candidate B initialization, training-only mild affine augmentation, and no
augmentation on validation. See `config/v3/training.json`.

Training uses mixed precision; validation uses FP32 for both Candidate B and
V3. Even a small new best macro F1 is checkpointed. The early-stopping minimum
delta controls patience separately; it does not discard a slightly better model.

The streaming loader refuses the test split. All fitting uses 350,000 training
images; selection uses 75,000 validation images and macro F1. Full resumable
state is saved after every completed epoch. The run has a 12-epoch ceiling,
three-epoch early-stopping patience, and a two-hour training budget. Create
`reports/v3/training/STOP` to stop before another epoch. GPU progress, MLflow
records, history, and the best candidate are separate from V1/V2 evidence.

After data validation, compare Candidate B on the same new validation set:

```powershell
.\.venv-gpu\Scripts\python.exe -u scripts/compare_product_validation.py
```

This uses B's own saved preprocessing, never product test shards. Running it
again after V3 training adds the new candidate's saved validation evidence to
`reports/v3/validation_comparison/comparison.csv`. Older V2 validation scores
and newer V3 validation scores are not directly comparable across datasets.

## Release limitations

500,000 synthetic samples are not 500,000 independent real-world observations.
Near-related fonts, ambiguous official categories, limited monospace diversity,
the Latin/English scope, and synthetic-to-real shift remain important. Real
screenshots can contain mixed fonts, logos, unusual cropping, photographs,
compression, or non-Latin text outside this dataset's scope.

No new real-world accuracy, calibrated uncertainty threshold, held-out V3 test
score, or production promotion is claimed until the corresponding work is
actually completed and recorded. V1's reported results apply only to V1.
