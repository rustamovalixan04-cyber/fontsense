# Data documentation

## Final source

The assessed dataset is generated from legally usable open-source font files in the official Google Fonts repository. `scripts/download_google_fonts.py` downloads a curated subset and records:

- family name
- broad Google Fonts category
- font file path
- license code
- Latin subset availability
- exact source URL

## Unit of observation

One record is one raster image containing a short Latin-script word or text line rendered with a single font family. The target is one of five broad categories.

## Split policy

The project splits **font families**, not images. All images rendered with one family remain in one split. This is the core leakage-prevention rule.

## Generated variations

Images may vary in text, size, position, background, foreground, blur, contrast, noise, JPEG compression, and a small rotation. Augmentation must not make the task unrealistic.

## Git policy

Do not commit the complete generated dataset or downloaded font collection. Commit the scripts, configuration, family manifest, dataset description, and small examples. Another person must be able to reproduce the dataset using the documented commands.
