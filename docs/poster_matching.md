# Poster text detection and unfamiliar-font suggestions

The reported ALLOVER poster exposed two product failures. The single-line OCR
reader received the subtitle and illustration with the headline. Then a weak
same-text pixel match replaced useful neural serif neighbors with heavy display
lookalikes. This release changes the input and fallback stages; it does not
retrain or replace the assessed classifier or the existing retrieval encoder.
Codex assisted with diagnosis, implementation and verification.

## Text before matching

`api/text_regions.py` runs the checksum-pinned Apache-2.0 PP-OCRv5 mobile text
detector on complex layouts. A 640-pixel maximum side bounds its cost. The
input uses BGR order and mean/std 0.5, matching the RapidOCR interface. Pure
NumPy/Pillow run-length components recover horizontal text boxes, without a
new OpenCV runtime dependency. At most 128 plausible components are grouped
by baseline, height and horizontal distance; adjacent words remain one line.
Expanded boxes preserve thin strokes. Up to six lines are proposed and four
readable regions are exposed. The largest lettering is selected first.

The existing clean-line path is retained when the normalized image is already
wide and has one substantial horizontal band. On complex images, the selected
crop is used consistently for OCR, category features, neural name references
and same-text font comparisons. The original `/predict` classifier contract
and frozen checkpoint remain unchanged.

The page highlights the selected text and offers **Text to match** when more
than one readable line exists. The `/identify` endpoint accepts an optional
integer `region` identifier. Identifiers belong to deterministic detection
results, so filtering an illustration or manually correcting text cannot
silently shift selection to a different line. Invalid identifiers return 400.
Manual text replaces the selected line's reading. Reset/upload clears region
selection, highlights and request caches. Uploads are not persisted.

This detector is bounded horizontal scene-text handling, not a full document,
curved-text or vertical-script OCR system. Small rotation correction remains
available. A drawing can resemble text; recognition rejects unreadable regions.

## A second reading requires visual evidence

`api/ocr_refine.py` supplies an optional PP-OCRv5 English reader when the first
reader's score is below 0.98. It does not choose between readers just by their
uncalibrated confidence scores. An alternative must have score at least 0.75
and improve actual rendered-letter agreement by at least 0.015 across up to
ten distinct open-family references from the image-only shortlist. Otherwise,
the original reading is retained. No word dictionary, family label, filename
or supplied expected answer is used to correct text.

On 120 existing real browser **development** images, case/whitespace-insensitive
exact text was 115/120 for each reader separately. This selective visual
comparison reached 116/120. These are component-selection results, not a new
independent accuracy estimate. Model score thresholds are engineering rules,
not displayed correctness percentages. Errors can remain.

## Similar designs without false identity

When the trained shortlist has direct pixel evidence at least 0.8, its
existing glyph comparisons and selected name acceptance rules are used as
before. When direct evidence is weaker, neural appearance neighbors are kept
as **similar fonts**. This avoids spending time rebuilding thousands of
text-specific references only to prefer a poorer heavy-font pixel alignment.
The original exhaustive engine remains explicitly available with
`FONTSENSE_RETRIEVAL_ENGINE=legacy`, and remains a fallback if no appearance
alternatives can be formed.

Appearance-only open neighbors cannot produce an asserted font name. They
keep their neural order rather than being reordered by an uncertain category.
A category consensus needs at least 60% of the first five neighbors (at least
two votes). Existing strict native reference gates remain separate. This
helps unfamiliar decorative serif designs while preserving uncertainty about
their exact source.

The user named the advertised face ALLOVER. Its publisher lists
[Allover Modern by 177Studio](https://creativemarket.com/177studio/27125891-Allover-Modern-Font).
It is not in this enrolled library. The screenshot regression now reads
`ALLOVER`, selects serif alternatives including Gloock, Playfair Display and
Kalnia, and withholds an asserted exact identity. No commercial font program
or example-specific recognition rule was added.

## Checks and protocol

The first generated diagnostic set contained 24 mixed-layout posters from 12
reserved test families, outside earlier browser-query families and detected
shared-Latin groups. A second query set exposed word splitting (`Cedar Valley`
became `Valley`). Both sets are diagnostic, not untouched final evidence.

After the line-grouping correction, the detector, OCR refinement and matching
rules were locked. The final set uses fresh phrases `Maple Harbor` and
`SPRING MEADOW` on the same 12 diagnostically inspected families, with light
and dark backgrounds, a separate subtitle and line art. Thus this is a fresh
query check, not a claim that the families were never inspected. Families are
excluded from new neural gradient training; enrollment references necessarily
contain searchable families. These are generated graphics, not real browser
or camera screenshots. The real user screenshot is a separate regression.

| Final generated poster check | Previous release | Updated |
|---|---:|---:|
| Images / families | 24 / 12 | 24 / 12 |
| Exact headline text, ignoring case/whitespace | 0/24 | 22/24 |
| Correct family among three returned matches | 0/24 | 24/24 |
| Broad category matches source metadata | 8/24 | 24/24 |
| Asserted names returned / correct | 0 / 0 | 18 / 18 |
| Requests requiring manual text | 24 | 0 |

The previous path treated the whole mixed design as one text line. Zeroes
therefore describe this particular layout failure, not its clean-crop accuracy.
Two final text readings remain wrong. Neither this small generated sample nor
18 accepted names establishes universal or 100% recognition accuracy.

The committed manifests and numerical summary are in
[`reports/poster_matching`](../reports/poster_matching). Model/source hashes are
recorded with the results. New tests cover the user poster, word grouping,
line switching, stable manual correction, invalid selections, secondary-reader
errors and identity withholding. The existing Impact, Times New Roman, style,
parity, API and deployment regressions are also run.

## Reproduction and deployment

```powershell
python -m pip install -r requirements-vercel.txt
python scripts/prepare_web_assets.py
python scripts/evaluate_poster_matching.py --prepare
python scripts/evaluate_poster_matching.py --final --prepare
python scripts/evaluate_poster_matching.py --final --engine candidate --output poster-final.json
python -m pytest -q tests/test_poster_matching.py
```

The fixed committed manifests reproduce the same generated images. Images and
raw comparison outputs are supplied separately in `fontsense-poster-checks.zip`;
generated corpora stay outside Git. To compare the prior release, place its API
and runtime artifacts from commit `cf5de74` in an isolated directory, prepare
its pinned reference assets, and pass `--api-root /path/to/prior/release`
with `--engine baseline`. Labels are used only by evaluation, never passed to
`identify`.

`ocr_layout_source.json` pins the 4.82 MB detector and 7.87 MB secondary reader
by SHA-256 and versioned source URL. They are downloaded and checked at build
time, ignored by Git, and bundled by the existing artifact include rule.
Runtime verifies their hashes before creating CPU ONNX sessions. There are no
per-request model downloads, new API keys or paid-model calls.

Primary implementation references:
[RapidOCR model registry](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml),
[detector preprocessing](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/ch_ppocr_det/utils.py),
and [configuration](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml).
