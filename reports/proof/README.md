# Proof-of-concept run — not final assessed results

This folder proves that the full FontSense code path works: font audit, family-level split, image generation, HOG baseline, CNN training, saved artifacts, inference, and held-out-family evaluation.

The proof run used **locally installed system fonts**, not the final approved Google Fonts dataset. Some system families were assigned to approximate categories only for technical testing, so these scores must not be presented as final project results.

## Proof dataset

- Usable independent font families: 46
- Generated images: 920
- Training images: 640
- Validation images: 140
- Test images: 140
- Font-family overlap across splits: 0

## Proof results

| Model | Test macro F1 | Test accuracy | Mean inference time |
|---|---:|---:|---:|
| HOG + Logistic Regression | 0.388 | 0.386 | 1.74 ms |
| Small CNN | 0.209 | 0.229 | 8.47 ms |

The low proof scores are useful: they show why label quality, independent family counts, and a proper Google Fonts category source matter. The final run must download Google Fonts, audit mismatches, regenerate the dataset, rerun MLflow experiments, and evaluate once on untouched Google Font families.
