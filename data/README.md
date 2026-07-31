# FontSense dataset

## Dataset and source

FontSense uses a synthetic computer-vision dataset made from font files in the
official [Google Fonts repository](https://github.com/google/fonts). The
download and audit implementation is in `scripts/download_google_fonts.py`.
The accepted family records, official source URLs, licence codes, Latin support,
and validation results are preserved in
`data/interim/google_fonts_manifest.csv` and
`data/interim/google_fonts_final_family_split.csv`.

Licensing is recorded per font family rather than replaced with one blanket
claim. The selected families use the licence codes present in the official
metadata (`OFL`, `APACHE2`, or `UFL`). Reusers must follow the licence recorded
for each source font and retain the relevant attribution. Evidence:
`data/interim/google_fonts_manifest.csv`.

## Unit of observation and target

One sample is one cropped 224 x 96 raster image containing a short
Latin-script word or phrase rendered with one font family. The prediction
target is a broad font category:

- display
- handwriting
- monospace
- sans serif
- serif

FontSense does not identify an exact font family. Evidence:
`reports/dataset/full_manifest.csv` and `config/full_dataset.json`.

## Counts and frozen split

The dataset contains 90 independent font families and 3,600 images:

| Split | Images | Families | Images per category | Families per category |
|---|---:|---:|---:|---:|
| Train | 2,400 | 60 | 480 | 12 |
| Validation | 600 | 15 | 120 | 3 |
| Test | 600 | 15 | 120 | 3 |
| Total | 3,600 | 90 | 720 | 18 |

Every family contributes exactly 40 images. The frozen split was selected
deterministically with seed 42 before image generation. A family appears in
only one split, which prevents the model from seeing the same font family in
both fitting and evaluation data. Evidence:
`data/interim/google_fonts_final_family_split.csv`,
`reports/dataset/full_manifest.csv`, and
`reports/eda/eda_validation_summary.json`.

## Generation process

`python -m fontsense.generate_full_dataset --verify-reproducibility` reads the
frozen family split and the settings in `config/full_dataset.json`. It renders
40 readable images per family using seed 42. Controlled variation includes
short English text, font size, spacing, position, foreground contrast, light or
dark background, small translation and scaling, mild rotation, mild blur, and
optional JPEG compression. The same effect schedule is used for every category
so an effect cannot reveal the target. Evidence:
`src/fontsense/generate_full_dataset.py`,
`config/full_dataset.json`,
`reports/dataset/full_validation_summary.json`, and
`reports/dataset/full_reproducibility_check.json`.

The committed full manifest is `reports/dataset/full_manifest.csv`. It records
the image path, family, category, split, rendered text, source font, seed, image
size, and applied effects.

## Reproduction

Reproduction requires the official Google Fonts source files and should be done
from the repository root:

```powershell
python scripts/download_google_fonts.py
python -m fontsense.generate_full_dataset --verify-reproducibility
python -m fontsense.eda
```

The checked-in split must remain frozen for an assessed reproduction. Do not run
`python -m fontsense.split` when reproducing the already assessed dataset; use
`data/interim/google_fonts_final_family_split.csv` exactly as committed.
Generation settings and expected hashes are documented in
`config/full_dataset.json`,
`reports/dataset/full_validation_summary.json`, and
`reports/dataset/full_reproducibility_check.json`.

Downloaded fonts and the 3,600 generated images are intentionally ignored by
Git because they are large and reproducible. The manifest, split, configuration,
validation summaries, and small contact sheets are versioned. Evidence:
`.gitignore`.

## Quality checks

The complete image audit opened all 3,600 images and found:

- 0 missing files
- 0 corrupted files
- 0 blank files
- 0 exact duplicate hash groups
- 0 suspicious near-identical pairs under the documented strict screen
- 0 families shared between train, validation, and test

Evidence:
`reports/eda/eda_validation_summary.json`,
`reports/eda/image_quality_metrics.csv`,
`reports/eda/suspicious_image_pairs.csv`, and
`src/fontsense/eda.py`.

## Limitations

- The images are synthetic renders, so photographs, screenshots, compression,
  clutter, and unfamiliar designs may differ from the training distribution.
- Google Fonts category labels are broad metadata labels, not objective
  typographic ground truth.
- A family-level split tests generalisation to unseen families, but only 18
  families per category are included.
- Automated blank and readability checks do not replace human visual review.
- The final model predicts categories, not exact family names.

The complete audit and Data Gate decision are documented in
`docs/data_audit.md`.
