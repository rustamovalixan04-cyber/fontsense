# Font matching in the web product

This product extension addresses obvious errors such as Times New Roman being
classified as sans serif. It does not retrain or replace the assessed capstone
checkpoint, nor change any saved training, validation, or final test results.
Codex assisted with implementation, tests, and documentation.

## How it works

1. The browser prepares one white-composited, metadata-free JPEG, up to 1800px
   on its longest edge. No upload session or keepalive is required.
2. `/identify` normalizes dark/light backgrounds and crops excess whitespace
   for retrieval. RapidOCR's English PP-OCRv4 recognition model reads one line.
   Users can correct the text and select Refine, skipping OCR.
3. The same text is rendered in 159 real regular, bold and italic styles from
   56 licensed Google Fonts reference families.
   Normalized ink-image cosine similarity and a small aspect-ratio penalty
   rank alternatives. This is visual retrieval, not a learned family classifier.
4. A small trained category model combines reference similarities, the original
   CNN outputs and simple image geometry. An exceptionally close reference can
   supply direct evidence. An unclear OCR result asks for text instead of
   inventing family names. Ambiguous decorative results are marked Possible.
5. The page displays three alternatives with server-rendered previews and
   Google Fonts links. Retrieval scores are not probabilities and are not
   displayed as confidence percentages.

## Runtime and reproducibility

`api/compact.py` executes the unchanged float32 V1 CNN weights in NumPy.
Its 112×48 input, grayscale normalization, layers, frozen threshold, class
order, and `/predict` response are preserved. The retrieval crop is never fed
to this CNN. `tests/test_compact_parity.py` compares its probabilities with
the original PyTorch architecture on fresh synthetic inputs.

`scripts/export_web_checkpoint.py` regenerates the 234KB NumPy export from the
SHA-256-verified original checkpoint using PyTorch with `weights_only=True`.
Only the export and manifest are deployed. Web requests load neither PyTorch
nor Gradio; optional OpenAI code is loaded only by its own endpoints.

`scripts/prepare_web_assets.py` downloads the original font files, their OFL
license texts, and the Apache-2.0 OCR model during installation. Every asset
has a pinned source and SHA-256 in `artifacts/web/`. A bad checksum fails the
build. There are no downloads or third-party model calls during matching.
Font files and OCR weights are ignored by Git and reproducible from manifests.
`artifacts/web/OCR-LICENSE.txt` and per-font OFL files accompany the deployment.

Run locally with Python 3.12:

```text
python -m pip install -r requirements-vercel.txt uvicorn pytest
python scripts/prepare_web_assets.py
python -m uvicorn api.index:app
python -m pytest tests/test_analysis.py tests/test_vercel_deployment.py tests/test_font_matching.py
```

Numerical parity also requires the project's PyTorch development dependency.
The original Gradio, Colab, and Windows applications retain their original flow.

## Original product check, 1 November 2026

`scripts/evaluate_font_matching.py` renders three fresh phrases (`Quiet forms`,
`Design 42`, `Pixel craft`) in clean, small JPEG, and inverted dark variants.
The check reads no historical training or final test images. These controlled
synthetic examples are development evidence, not an estimate of real-world
accuracy. Families in the reference library are an intentionally easy setting.

| Set | Images | Original category correct | Updated category correct | Family top 1 / top 3 |
|---|---:|---:|---:|---:|
| 39 library families | 351 | 253 | 350 | 344 / 349 |
| 16 Windows families outside the library | 144 | 95 | 129 | Not applicable |

OCR text was correct ignoring spaces/case for 336/351 library examples and
140/144 outside-library examples. Local median matching was 73.2ms and 73.6ms;
p95 was 129.8ms and 90.2ms respectively. These measurements include recognition,
matching and CNN fallback computation, exclude network upload/preview requests,
and are specific to the development machine. Initial OCR/model loading is slower.

Reproduce the portable subset:

```text
python scripts/evaluate_font_matching.py --output work/product-check.json
```

On Windows, optionally add `--windows-fonts C:/Windows/Fonts` if the fonts listed
in the script are installed. Windows font binaries are never redistributed.

## Trained category combiner, 2 November 2026

`api/category_ensemble.py` uses a 1,592-byte NumPy export of a standard-scaled,
class-balanced multinomial logistic regression. It combines 28 signals: the
three closest distinct-family scores in each category, original CNN probabilities
and log probabilities, and ink density, ink-run count and aspect ratio. A
high-quality reference match takes precedence when its similarity is at least
0.90 with at least 0.06 separation from another category. Output probabilities
are not calibrated and are never presented as measured confidence percentages.

`scripts/train_category_ensemble.py` generated 3,816 synthetic examples using six
new phrases, real reference styles, random 48–80px type sizes, compression,
polarity and small rotations. Source families are assigned to exactly one learner
split: 42 train families (2,904 images), 14 validation families (912 images).
Each image's own source family, including every style, is excluded from its
reference features. The learner never receives the input's font name, path or
ground-truth category. The reference database itself is fixed independently;
these checks are not a replacement for a larger real-screenshot benchmark.

One actual training run used C=1, balanced weights, seed 20261102, and max_iter
1500. Learner validation accuracy was 89.47%, macro F1 0.8787. MLflow run
`2fbf6927a66c4839a0b1586ea39db923` records real parameters, metrics and provenance.
No old training image, checkpoint, capstone split or final test result changed.
See `artifacts/web/category_ensemble.json` for the full source-family split,
metrics, hash and portable inference fixtures. The unchanged original classifier
remains independently available at `/predict`.

New recognition normalizes empty spacing, straightens small rotations when row
alignment clearly improves, compares real font weights/italics, and returns each
family once. Preview rendering copies cached images before resizing, preventing
later comparisons from being changed by a preview request. Reference vectors use
one rendering pass with bounded caches. Training needs
`requirements-product-training.txt`; production needs no sklearn, MLflow or Torch.

`scripts/build_font_catalog.py` audits pinned font metadata, OFL licenses and
Latin coverage. Monospace advances are checked with one-pixel hinting tolerance.
Comic Relief has an explicit recorded taxonomy audit: upstream calls its intended
use DISPLAY, while FontSense groups casual comic-style hand lettering with
handwriting. The upstream category is preserved in each record. This is a
reference-label convention, not a special case on uploaded font names.

The earlier 20-family external check exposed weaknesses and became development
evidence; it is not reported as untouched final evidence. After selecting the
learner, a fresh 20-family set absent from the library and learner splits was
rendered: two phrases, up to two real styles, clean/small JPEG/dark/2.5-degree
rotation variants. No changes were fitted against those final images. The final
cosmetic uncertainty flag and catalog integrity check do not change categories.

| Check | Previous web matcher | New matcher |
|---|---:|---:|
| External development families: category correct | 164/248 | 212/248 |
| External development macro F1 | 0.6627 | 0.8510 |
| Fresh family check: category correct | 150/248 | 202/248 |
| Fresh family check macro F1 | 0.6061 | 0.7244 |
| Earlier library examples: category correct | 350/351 | 342/351 |

These are controlled synthetic product checks, not real-world accuracy claims.
There are regressions: the library check loses eight correct examples. On the
fresh set, decorative Display recall falls from 15/32 to 4/32; handwriting rises
24/40 to 32/40, monospace 49/64 to 64/64, sans serif 38/64 to 57/64, and serif
24/48 to 45/48. Decorative fonts are a material limitation; ambiguous decorative/
handwritten disagreements are shown as Possible. Broader decorative training
families and real screenshots are needed before claiming uniform improvement.

Fresh-check local paired median runtime was 86.8ms before and 105.7ms after;
p95 was 115.5ms and 141.4ms. More styles improve coverage with some added CPU
work. Measurements exclude uploads, preview requests and hosting cold starts.

Reproduce training (requires the prepared assets):

```text
python -m pip install -r requirements-product-training.txt
python scripts/train_category_ensemble.py
```

Reproduce the independent family check from the Git checkout with baseline
history available (commit 7e899e9):

```text
python scripts/build_font_catalog.py --final-holdout --output work/final-fonts
python scripts/evaluate_matching_upgrade.py --phase holdout --assets work/final-fonts --output work/final-check.json
```

## Limits

The library covers 56 families and 159 actual styles. Unseen fonts,
unsupported weights, logos, perspective, textured backgrounds, mixed fonts,
multiline text and very short or illegible crops can return poor alternatives.
Use one clear Latin-script line; correct recognized text if needed. Even a strong
visual match cannot establish the original typeface's identity.

Image normalization assumes a mostly uniform background. The raw CNN remains
available at `/predict`; its known errors persist because the weights are frozen.
Historical capstone metrics describe that classifier, not this matching system.
The app uses bounded in-process caches, which may be lost when hosting scales
down. The UI has no idle-session dependency, but hosting can still cold-start.

Optional private OpenAI analysis remains independent, server-authenticated, and
disabled until configured. The simplified public page does not expose that pilot.
