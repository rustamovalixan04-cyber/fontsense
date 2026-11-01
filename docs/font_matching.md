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
3. The same text is rendered in 39 actual Google Fonts reference families.
   Normalized ink-image cosine similarity and a small aspect-ratio penalty
   rank alternatives. This is visual retrieval, not a learned family classifier.
4. A reasonably close reference supplies the broad category; weak matches use
   the original CNN and restrict suggestions to that broad category. An unclear
   OCR result asks for text instead of inventing family names.
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

## Product check, 1 November 2026

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

## Limits

Only 39 reference families and their default styles are compared. Unseen fonts,
bold/italic styles, logos, perspective, textured backgrounds, mixed fonts,
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
