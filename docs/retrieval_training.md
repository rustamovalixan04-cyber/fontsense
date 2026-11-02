# Trained product font retrieval

This is a separate product experiment. It does not modify the assessed
five-category checkpoint, training dataset, test results, Gradio demo or Colab
demo. Codex assisted with implementation, debugging, training and measurement;
the student remains responsible for understanding and defending the work.

## Data and learning

The pinned OFL catalog supplies 1,840 eligible Latin families / 3,288 real
styles. Families are split before query generation: 1,325 training, 254
validation and 261 test. Exact renderings of the probe `Hamburgefonts 012`
are grouped across styles before splitting, keeping detected shared Latin
lettering in one partition. This probe does not detect every possible alias.

The new small CNN starts from random weights. It learns a 96-dimensional
normalized image representation from **64,752 patches**: 63,600 synthetic
patches and 1,152 additional patches from 96 real browser training images.
The synthetic augmentation includes size, case, real styles, shear, width
changes, rotation, blur, contrast, polarity, noise, downsampling and JPEG damage.
Simulated damage is not described as a real camera photograph.

Training uses 160x64 grayscale patches, AdamW (learning rate 0.0015,
weight decay 0.0002), cosine scheduling, angular class margin 0.1 at scale 24,
and paired-family agreement weighted 0.15. Batches contain up to 48 families
with two different examples each. Six CPU threads ran 16 epochs. Epoch 15
was selected using validation top-100 then top-10 retrieval. The actual MLflow
run is `20a4ec617cde467ba747d6219ea81a86`.

On the same 320 synthetic development queries, candidate retrieval changed:

| Encoder | Top 1 | Top 10 | Top 100 |
|---|---:|---:|---:|
| Previous small encoder | 26.88% | 57.81% | 89.06% |
| Newly trained encoder | 60.94% | 94.06% | 98.75% |

These measure neural candidate retrieval, not final exact-name accuracy.

## Enrollment and runtime

The compact reference index contains **1,862 families / 3,350 styles /
10,050 vectors**, using four enrollment phrases in three cases. Enrollment
references include all catalog families, including validation and test families:
this is necessary to search for their names. Test families are excluded from
new neural gradient training, not from reference enrollment. Enrollment phrases
and final query images are different.

The 22 native Windows families are enrolled from locally installed fonts.
Only non-rendering statistics of complete phrases are deployed. Native font
programs, outlines and per-letter bitmap atlases are not copied or served.
Rebuilding these optional references requires the corresponding installed
fonts; normal deployment uses the checked-in derived index.

At runtime, ONNX Runtime embeds three image patches and searches the index.
Up to 400 real source styles are compared using the recognized text, both
full-line and spacing-tolerant descriptors. Reliable separated glyphs add
letter-shape evidence. The five strongest open families can be fitted to real
supported weight, width and optical-size axes, one axis at a time. The
exhaustive matcher remains a fallback when direct shape evidence is weak.
Native candidates survive that fallback.

The 12-patch ONNX parity check has maximum absolute error 3.82e-7 against
the actual PyTorch vectors. Runtime verifies model and index hashes. No
PyTorch, training data, API key or paid model call is needed for matching.

## Naming and uncertainty

The open-font correctness filter is fitted on one development-family group
and selected on a different group. Its inputs include direct shape scores,
different-family gaps, glyph evidence and embedding similarities. Additional
negative examples remove the true family from the shortlist. Its selected
minimum direct score is 0.92; its fitted evidence threshold is about 0.595.
These constructed-mixture scores are not universal probabilities and are not
displayed as confidence percentages.

The separate native-name gate requires embedding similarity at least 0.9 and
global prototype margin at least 0.05. Nine native query families are excluded
from positive confidence-query fitting and selection. Actual outside-index
native calibration fonts are distinct from final outside-index fonts; failed
renderings are recorded as exclusions.

Possible native names use a weaker development-derived floor (about 0.774)
and a top-ten global family rank. This controls an explicitly uncertain list,
not the stricter **Likely font** result. Similar open fonts remain available
with actual previews and source links.

## Evaluation protocol

The browser corpus contains 620 actual screenshots with saved DOM crop
coordinates, source hashes and loaded-face checks. Of these, 96 are training,
120 development validation, 160 initial diagnostic OFL queries, 42 initial
native/unknown queries, 160 fresh OFL queries and 42 fresh native/unknown
queries. Every final query is also evaluated after separately reported
JPEG/downsampling damage. The resulting conditions are correlated views,
not twice as many independent screenshots.

The initial diagnostic test exposed lost native information in the fallback
and an overly strict uncertain-candidate display rule. Those were corrected
using development data. The initial diagnostic results are therefore not
presented as untouched final evidence.

The fresh OFL check uses 80 different reserved test families, excluding
duplicate-lettering groups from the first diagnostic set, and new phrases.
The fresh native check uses new phrases on the same nine diagnostically seen
native families, plus six outside-index families. Those nine families remain
excluded from new neural gradient training and positive confidence-query
fitting. Thus the native check is a fresh-query check, not a claim that the
families had never been inspected. Model weights, gates and matching rules
were locked before the fresh comparison.

The locked fresh comparison produced these results:

| Fresh OFL check: 320 conditions from 160 screenshots / 80 families | Previous | Trained |
|---|---:|---:|
| Asserted names returned | 71 | 97 |
| Correct asserted names | 71 | 97 |
| Name coverage | 22.19% | 30.31% |
| First displayed match correct | 73.44% | 75.31% |
| Correct font in three similar matches | 81.88% | 81.88% |
| Correct font in three identity candidates | 74.38% | 82.19% |
| Conditions needing text correction | 4 | 4 |

| Fresh native check: 60 known + 24 outside-index conditions | Previous | Trained |
|---|---:|---:|
| Asserted names returned | 4 | 14 |
| Correct asserted names | 0 | 14 |
| Known-font name coverage | 6.67% | 23.33% |
| Correct font in three identity candidates | 0% | 90% |
| Outside-index asserted names withheld | 24/24 | 24/24 |

Across both checks the trained system returned 111 correct asserted names and
no incorrect ones, but only named **111 of 380 known-font conditions (29.21%)**.
This is observed selective precision on a small constructed check, not 100%
recognition accuracy. The remaining known inputs rely on candidates or text
correction. The native check is especially small. Identical top-three OFL
retrieval also shows where added training did not improve this measure.

The machine-readable summary is
[`fresh_browser_evaluation.json`](../reports/product_retrieval/fresh_browser_evaluation.json).
Raw per-image results accompany the separate browser corpus ZIP. Timing from
concurrent CPU evaluation is not presented as hosted user-request latency.

No measured subset establishes worldwide accuracy. Sparse text, OCR errors,
clones, connected scripts, private fonts, unsupported glyphs and heavily altered
artwork can remain ambiguous. The app should withhold an asserted name and
offer candidates when evidence is insufficient.

## Reproduction

Use a separate development environment; do not install training dependencies
in the deployment. The original run used PyTorch 2.10 CPU, ONNX 1.20.1 and
ONNX Runtime 1.31.1. Install the project editable for the training model import.

```powershell
python -m pip install -r requirements-name-training.txt
python -m pip install -e .
python scripts/prepare_web_assets.py
python scripts/retrieval_data.py --samples 48
```

`scripts/browser_benchmark.py` serves deterministic specimen pages on localhost
port 8771, using OFL references and installed native fonts only. Capture the
pages with the browser after `data-ready=true`, save the specimen rectangles
and source screenshots under `../retrieval-training/browser/captures`, then run
`scripts/import_browser_captures.py`. It verifies registration colors before
cropping; browser zoom must not be assumed to equal screenshot pixel scale.
The supplied separate browser corpus ZIP contains the cropped images and
provenance. It can be extracted under `../retrieval-training/browser` before
training; it contains no font programs. Local raw captures are retained for
audit, outside Git.

```powershell
python scripts/train_retrieval_encoder.py --epochs 16
python scripts/build_retrieval_index.py
python scripts/calibrate_retrieval_confidence.py
python scripts/compare_retrieval_encoder.py
python scripts/evaluate_browser_matching.py --split test2 --engine legacy --output baseline.json
python scripts/evaluate_browser_matching.py --split test2 --engine candidate --output candidate.json
python scripts/evaluate_browser_matching.py --split native_test2 --engine candidate --output native.json
```

Training outputs and MLflow tracking live outside Git in
`../retrieval-training`. ONNX, index, confidence and policy artifacts are the
small runtime exports that are committed. Exact floating-point training
reproduction can vary across hardware and library versions; parity verifies
the selected exported checkpoint.
