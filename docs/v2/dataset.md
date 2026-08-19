# FontSense V2 dataset

The V2 dataset was generated from the frozen 200-family split with seed 42. It contains exactly 20,000 cropped Latin-text images:

- 100 images for every family;
- 4,000 images for every category;
- 14,000 training images;
- 3,000 validation images;
- 3,000 test images.

Images are 224×96 RGB PNG files. Model preprocessing later converts them to grayscale and resizes them to 112×48. Generation includes readable short English phrases, light and dark backgrounds, strong and softer contrast, small spacing and position changes, mild rotation/scale/blur/JPEG effects, plus mild V2 resampling, shear, stroke, and fine-noise options. The complete configuration is `config/v2/dataset.json`.

Effects and phrases use the same complete schedule in every category. Families receive deterministic offsets into that schedule, which prevents look-alike fonts in different categories from producing exact duplicate renders without changing the category-level distribution.

Validation opened all 20,000 files and found:

- 0 missing files;
- 0 empty files;
- 0 blank images;
- 0 corrupted images;
- 0 exact SHA-256 duplicate groups;
- 0 family overlap between train, validation, and test.

A bounded 64-bit difference-hash audit made 2,999,356 comparisons and found many visually similar pairs. This is expected because short text crops often share similar backgrounds and overall edge layouts. The first 5,000 representative pairs are recorded in `reports/v2/data/near_duplicates.csv`; this is a warning for error analysis, not evidence of exact duplicates. Exact byte hashes are the strict duplicate gate.

The manifest plan reproduced exactly on a second seed-42 build, and 25 regenerated sample images matched their original SHA-256 hashes. The manifest SHA-256 is `95fa9642c8bbc0ecfe6af1d4d1e893ed041a238fa1b6a4da59760f58407132e7`.

Manual review of the five contact sheets confirmed that the sampled train, validation, and test images are readable, uncropped, and visually varied. The raw PNG files live under `data/v2/processed/` and are ignored by Git. They can be regenerated with:

```powershell
.\.venv\Scripts\python.exe -m fontsense.v2_generate
```

Evidence:

- `reports/v2/data/full_manifest.csv`
- `reports/v2/data/dataset_validation_summary.json`
- `reports/v2/data/reproducibility.json`
- `reports/v2/data/effect_balance.csv`
- `reports/v2/data/exact_duplicates.csv`
- `reports/v2/data/near_duplicates.csv`
- `reports/v2/data/contact_sheets/`
