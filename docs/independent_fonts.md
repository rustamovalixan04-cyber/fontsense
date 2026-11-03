# Independent fonts in FontSense

This is a separate web-product expansion. It does not change the assessed
five-category classifier, its checkpoint, dataset or historical course results.
Codex assisted with source research, implementation, training and verification;
the saved reports contain measured results rather than promised accuracy.

## Sources and coverage

The new catalog contains **81 families, 154 searchable styles and 153 source
font files**. Sources include FontSpace listings by GGBotNet, the designer's
CC0 collection, Collletttivo and Velvetyne repositories. Every accepted file
has an actual OFL 1.1 or CC0 1.0 license, a recorded source page, original and
subset hashes, and a full license hash. This is not permission to use every
font found on FontSpace: personal-use-only and unclear grants are excluded.

Mirrored TTF/OTF releases and existing Google families do not create additional
classes. During image inspection, Auseklis (symbols) and Naganoshi (repeated
placeholder boxes) were excluded despite their Latin character mappings.
Fourteen families also received documented genre annotations from Latin
enrollment renderings, since missing Panose metadata had left clear serif and
brush designs with a provisional display label. Genre labels are not encoder
inputs. This reference-label review preceded final query evaluation.
The complete audit, including excluded sources, is in
[`source_audit.json`](../reports/independent_fonts/source_audit.json).

References are downloaded at build time, checksum-verified, subset to Latin
and renamed deterministically. Full copyright and license files accompany
the subsets. Archive members are read in memory; archive paths are never
extracted to disk. No native Windows font programs are distributed.
The originally observed ZIP-container hash is retained for provenance.
FontSpace ZIP containers can vary independently of the selected contents, so
builds pin the exact original font, full license and derived subset bytes;
changed font or license bytes still fail before any asset is written.

## Actual training

Training and queries were separated before generation. Source-related families
and identical Latin renderings remain together. The 81 supported new families
are split into 57 training, 13 validation and 11 test families. Another seven
non-enrolled FontSpace families are reserved for unknown-font checks: two for
calibration and five for final evaluation. Their query images do not enter
gradient training, and their references do not enter the deployed library.

Known validation/test families necessarily have enrollment reference phrases
in the search library. Enrollment is separate from query images and is not
evidence of training on those families. This is retrieval, rather than a
closed-set classifier with a fixed output label for each supported font.

The previous product checkpoint was fine-tuned on **20,700 image patches**:
15,900 old synthetic replay patches, 1,152 patches from old training-family
browser screenshots, and 3,648 simulated screenshots from 57 new families.
Image damage includes blur, compression, rotation, scaling and inversion.
These simulations are not photographs or browser captures.

The model uses the existing 96-dimensional CNN embedding. Batches contain 75%
old-family examples and 25% new-family examples. A teacher embedding penalty
limits forgetting, and batch-normalization statistics stay frozen. Epoch one
adapts the enlarged training head; later epochs update the image encoder at a
small learning rate. The head is not part of runtime font identification.

One real MLflow run, `6f448467e8f8402e8ea9063880eefbc7`, ran six epochs.
Epoch **6** was selected on validation data only. Selection averages old/new
top-1/top-10 retrieval and requires old top-10/top-100 to remain within the
declared 1.5/1 percentage-point tolerances. Final test queries were not used.

With identical expanded reference libraries:

| Validation sample | Previous encoder | Selected encoder |
| --- | ---: | ---: |
| New independent fonts, top-1 (52 queries) | 84.62% | 88.46% |
| New independent fonts, top-10 | 100% | 100% |
| Existing fonts, top-1 (320 queries) | 61.88% | 62.19% |
| Existing fonts, top-10 | 94.06% | 93.75% |
| Existing fonts, top-100 | 98.75% | 98.75% |

These are development retrieval measurements, not full OCR accuracy or a
claim about all fonts. Enrollment alone accounts for the new fonts becoming
searchable; the table isolates the additional effect of weight adaptation.
Detailed training history and hashes are saved in
[`training.json`](../reports/independent_fonts/training.json).

## Final post-selection check

The locked model was evaluated on new phrases at 83px, both clean and with
simulated damage. Known-family enrollment references remained available; query
images did not enter training or calibration. The warm-start arrays were also
compared against the previous release's ONNX output: maximum absolute error
was 3.82e-7 over 12 patches. Both comparison columns use the same expanded
gallery, which differs from the historical gallery and its tied-font ordering.

| Encoder-only final retrieval | Previous encoder | Selected encoder |
| --- | ---: | ---: |
| New fonts: 11 test families / 44 queries, top-1 | 44/44 | 44/44 |
| Existing fonts: 40 test families / 160 queries, top-1 | 120/160 | 121/160 |
| Existing fonts, top-10 | 155/160 | 155/160 |
| Existing fonts, macro F1 over query families | 0.79794 | 0.79698 |

The independent-font final set tied; it does **not** demonstrate an additional
retraining gain on that sample. The larger library supplies the new identities.
The small old-font F1 decline is retained in the report rather than hidden.

The complete pipeline was then checked on 44 known and 20 actual non-enrolled
font images. The five unknown families form four independent source groups.

| Full pipeline | Automatic OCR | Supplied correct text |
| --- | ---: | ---: |
| Known font in top three | 43/44 | 44/44 |
| Likely names returned / correct | 36/36 | 42/42 |
| Known-name coverage | 36/44 | 42/44 |
| Unknown names withheld | 20/20 | 20/20 |
| Exact normalized text across all 64 images | 47/64 | 64/64 |
| Median local time | 1.766s | 1.502s |

Full details, individual errors and timings are in
[`final_evaluation.json`](../reports/independent_fonts/final_evaluation.json).
The automatic-text script initially passed `None` to a string parameter and
stopped before pipeline measurement. Only that invocation was corrected;
the same locked weights, gates and queries were rerun. No model selection or
gate adjustment used these final outcomes.

Calibration separately returned 30/30 correct names and accepted zero of 137
constructed/unknown selection rows. These are empirical counts on small
samples. Synthetic rendering, repeated conditions within each family and
selective naming limit what can be inferred about real-world accuracy.

## Build and runtime

`independent_catalog.json` extends the original catalog without replacing it.
`prepare_web_assets.py` calls `independent_font_assets.prepare` to rebuild
verified font and license assets. Runtime recognition remains server-side,
offline and key-free. It uses the learned shortlist followed by actual text
and letter-shape comparison. Similar-font suggestions and the existing
classifier remain available. Training dependencies and the warm-start
checkpoint are excluded from deployment.

The index must be rebuilt with the selected checkpoint, and the correctness
filter must then be recalibrated for that index. Hash checks reject a mixture
of incompatible artifacts. Calibration uses old validation screenshots,
validation-only synthetic fonts, native development examples and actual
unfamiliar FontSpace fonts. Family/source groups stay together in its fitting
and selection partitions. A guessed font must still pass the shape and
correctness gates before being displayed as a likely name.

## Reproduction

Use the pinned development requirements from `requirements-name-training.txt`
and the normal editable project install. `scripts/prepare_web_assets.py`
reproduces the deployed references directly from their committed manifest.
The small `artifacts/training/retrieval_base_state.npz` stores the previous
product checkpoint safely as NumPy arrays; its JSON contracts document the
original run and hash. It is needed to reproduce the warm start, and contains
neither font programs nor assessed classifier weights.

To reproduce the training run, restore the separately supplied
`fontsense-browser-benchmark.zip` (14,883,740 bytes) under
`../retrieval-training/browser`. Its `samples.json` and images are required;
the script fails explicitly if they are absent. The corpus was supplied with
the previous product run, and is described in [retrieval_training.md](retrieval_training.md).
No new browser screenshots are implied by this experiment.

```powershell
python scripts/prepare_web_assets.py
python scripts/independent_training_data.py
python scripts/train_independent_encoder.py --epochs 6
python scripts/build_retrieval_index.py --checkpoint ../independent-training/selected-model.pt
python scripts/calibrate_retrieval_confidence.py --data-dir ../independent-training
python scripts/evaluate_independent_fonts.py
```

Data preparation restores new font sources using the committed audit and saved
splits. It can regenerate old replay patches if the old training cache is
absent. Development queries can also be regenerated without their old cache.
Training data, raw font downloads and MLflow files stay outside Git under
`../independent-training` and `../independent-fonts`. The optional discovery
script `build_independent_catalog.py` audits a new registry; discovery is not
required to reproduce this fixed release. Do not rerun discovery after looking
at final results and then describe those results as an untouched test.

CPU floating-point results may vary across machines. The committed parity
fixture contains actual selected PyTorch outputs and checks the ONNX export.
The selected model, index and confidence contract must be published together.

## Limits

This remains a finite Latin font library. OCR can fail on heavily decorated
letters, and distinct fonts can have identical or nearly identical glyphs.
All newly reported test images are synthetic; real-world font specimens need
separate evaluation. Unknown-font evidence is a small sample, not a universal
guarantee. A withheld name and useful alternatives are preferable when the
evidence is ambiguous. Adding these sources does not imply coverage of every
FontSpace listing or of the ALLOVER font shown earlier.
