# Data documentation

## Final source

The assessed dataset is generated from legally usable open-source font files in the official Google Fonts repository. `scripts/download_google_fonts.py` downloads a curated subset and records:

- family name
- broad category read from official Google Fonts metadata
- official Google Fonts source and exact source URL
- font file path
- license code
- Latin subset availability
- Pillow validation status and any failure reason

The audit writes `data/interim/google_fonts_manifest.csv` and a small category
summary. Downloaded font files remain ignored by Git. This audit does not
generate the image dataset or train a model.

## Final family split

`python -m fontsense.split` selects 18 usable families per category with random
seed 42, then assigns 12 to training, 3 to validation, and 3 to test. It writes:

- `data/interim/google_fonts_final_family_split.csv`
- `data/interim/google_fonts_balancing_exclusions.csv`

This split is created before image generation. A font family must never move
between splits, and the test families must remain untouched until final
evaluation.

## Preview dataset

`python -m fontsense.generate_preview` reads the frozen split without making new
assignments. It creates exactly two inspection images per family with seed 42.
The 180 preview images are kept in the ignored
`data/processed/fontsense_preview/` folder, separate from the future full
dataset. The small manifest, validation summary, and category contact sheets
are saved in `reports/preview/`.

The preview is only for checking image generation quality. It is not a final
training dataset and does not contain model results.

## Full image dataset

`python -m fontsense.generate_full_dataset --verify-reproducibility` reads the
frozen family split without creating or changing assignments. With seed 42 it
creates:

- 90 independent font families
- 40 images per family
- 3,600 images in total
- 720 images per category
- 2,400 training images
- 600 validation images
- 600 test images

The complete settings are saved in `config/full_dataset.json`. Images are
224 by 96 pixels and use short English text with controlled changes to font
size, spacing, horizontal position, foreground contrast, background, scale,
translation, rotation, blur, and JPEG quality. Rotation is limited to 2.5
degrees, scaling to 0.94–1.06, and blur to at most 0.55 pixels so the text
stays readable.

The same deterministic effect schedule is used in every category. This stops
background or augmentation frequency from acting as a shortcut for the class.
Validation results are stored in `reports/dataset/`, including the full
manifest, effect-balance table, reproducibility check, and contact sheets.
The 3,600 PNG files remain ignored in `data/processed/fontsense_full/`.

## EDA and data-quality audit

`python -m fontsense.eda` reads the existing full manifest, frozen split, and
3,600 generated images. It does not regenerate images and does not train a
model.

The audit verifies:

- 3,600 unique image paths and 90 independent families
- 720 images and 18 families per category
- 2,400 train, 600 validation, and 600 test images
- exactly 40 images per family
- zero family overlap between splits
- all images open at 224 by 96 pixels
- no missing, corrupt, blank, exact-duplicate, or strictly near-identical files
- the saved effect and phrase schedules are balanced across categories
- manifest metadata is not used as a model feature

The full per-image measurements, validation summary, notebook, report, and
figures are saved under `reports/eda/`, `reports/figures/`, and
`notebooks/03_eda.ipynb`.

The automated readability screen catches missing, corrupt, blank, or extremely
low-contrast files. It does not replace human review. These findings describe
data quality only; they are not model-performance results.

## Unit of observation

One record is one raster image containing a short Latin-script word or text line rendered with a single font family. The target is one of five broad categories.

## Split policy

The project splits **font families**, not images. All images rendered with one family remain in one split. This is the core leakage-prevention rule.

## Generated variations

Images vary in text, size, position, background, foreground contrast, letter
spacing, scaling, translation, blur, JPEG compression, and mild rotation.
Augmentation is applied independently of category and must not make the text
unreadable.

## Git policy

Do not commit the complete generated dataset or downloaded font collection. Commit the scripts, configuration, family manifest, dataset description, and small examples. Another person must be able to reproduce the dataset using the documented commands.
