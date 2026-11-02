# Font names and similar fonts

The original 86-family encoder described below remains available. The product
now also searches the [expanded open-font catalog](extended_font_retrieval.md)
and shows possible names when exact identity is ambiguous. Metrics below belong
to the earlier encoder evaluation, not the expanded retrieval system.

The earlier web product returned a category and open-font alternatives. Impact
was correctly categorized as Display but Anton was the first alternative; there
was no font-name recognizer and no Impact reference. Adding more category data
alone could not make the page identify an absent family by name.

The result now separates **Likely font** from **Similar fonts**. A supported,
confident match can show Impact, with Anton and the other alternatives below.
When the evidence is weak, the page says **Not identified confidently** and still
offers visual alternatives. Similar font names are never presented as proof of
the uploaded image's original family. No calibrated probability is displayed.

## Data and models

The separate name reference collection covers **86 families / 214 styles**:
76 OFL families and 10 locally available Windows families. Six proposed extra
sources were excluded after their pinned OFL URLs returned 404; their names and
reasons are recorded in `font_name_gallery.json`. Actual included counts are used.
The existing alternative-rendering library remains 56 families / 159 styles.

The name encoder uses the small four-block CNN architecture already understood
by this project, followed by a 64-dimensional projection. Training compares
normalized image vectors with font-family targets using cosine logits, scale 20
and positive margin 0.1. Font names and file paths are labels/provenance, not
inputs to the image model. The page's recognized text is used only for case and
simple geometry in the secondary comparison.

One actual 12-epoch training run used **7,296 whole-word image patches** from
57 OFL families. Nineteen other OFL families were held out from fine-tuning and
used with disjoint reference/query words for epoch selection. Epoch 10 returned
96/114 top-one names (84.21%) in that development retrieval check. The encoder
starts from the original category CNN: families are held out from this new
fine-tuning run, not necessarily every earlier pretraining source. This is not a
replacement for or rerun of the capstone's assessed final test.

MLflow run: `2b0aa33a4f254918a56f52d932cdf7ec`.
The trained encoder is exported as a 251KB NumPy archive; production needs no
PyTorch or MLflow. Portable-vector tests compare its outputs with the real
PyTorch export. The original checkpoint, category model, `/predict`, Colab app
and historical evaluation evidence remain preserved.

The reference gallery averages image features from **7,704 complete word/phrase
renderings**, using lower, title and upper case. The selected retrieval score is
75% new encoder similarity plus 25% normalized features from the frozen CNN and
simple geometry. Gallery-fitted normalization uses fine-tuning training families
only. Family references themselves are required by the name-lookup task; known
gallery retrieval is distinct from unseen-family category classification.

## Windows fonts and provenance

Windows font programs remain on the local Windows machine. No Impact TTF/WOFF,
font outline or indexed per-letter bitmap font is copied to Git or the server.
Only complete rendered words/phrases are used to derive non-rendering reference
statistics. Native fonts do not participate in encoder gradient training.
The reference metadata records source SHA-256 values and official information
links; it contains no personal absolute paths or credentials.

Microsoft's [font FAQ](https://learn.microsoft.com/en-us/typography/fonts/font-faq)
permits word/phrase graphic output and restricts font redistribution, conversion
and web self-hosting. The implementation keeps that distinction: it does not
serve native font files or use them to render arbitrary text on the server.
See [Impact source information](https://learn.microsoft.com/en-us/typography/font-list/impact).
OFL font source records retain pinned URLs, copyright/license provenance and
hashes. Regression PNGs are whole-word images and are documented in
`tests/fixtures/README.md`.

## Confidence selection and evaluation

Confidence settings were selected on 976 development queries with different
complete phrases from the reference gallery. The selection required at least
97% precision among accepted names and at least 95% rejection on its 40 unknown-
font examples. The selected thresholds are similarity >=0.93, inter-family margin
>=0.02 and a 0.025 penalty for case disagreement. These are empirical gates, not
calibrated probabilities or future guarantees. The selected gates accepted
341 names, 332 correctly (97.36%), and rejected 39/40 unknown examples.

The later synthetic check used two fresh complete phrases, four image conditions
(clean, reduced JPEG, dark, 2-degree rotation), all source styles, and 20 unknown
families absent from the gallery. Typography is the user's reported regression
word and is reported separately rather than counted as unseen evidence.

| Fresh-query result | Count |
|---|---:|
| Images | 1,872 |
| Known-gallery images | 1,712 |
| Unknown-family images | 160 |
| Returned names | 344 |
| Correct returned names | 339 |
| Precision among returned names | 98.55% |
| Known-image coverage | 19.98% |
| Unknown names withheld | 158/160 (98.75%) |

The high precision is selective: approximately 80% of known-gallery images do
not receive a confident name. Alternatives still work for those images. Five of
eight fresh Impact examples were named, all five correctly; the other three were
withheld. The provided Impact screenshot crop is named Impact and has a dedicated
regression test. On the separate Typography sweep, 75/78 returned names were
correct; it is not advertised as a blind accuracy benchmark.

These are controlled synthetic renderings, not a real-world accuracy estimate.
Close families, compatible designs, unsupported fonts, short text, unusual
rendering, mixed text, large rotations and low-resolution screenshots remain
limitations. Reference size is finite; declining to name a font is intentional.

## Reproduce

```text
python -m pip install -r requirements-name-training.txt
python scripts/prepare_web_assets.py
python scripts/prepare_name_data.py
python scripts/train_name_encoder.py
python scripts/build_name_gallery.py
python scripts/evaluate_font_identity.py --output work/name-evaluation.json
```

Full native-reference reproduction requires the corresponding legally available
local Windows fonts. OFL encoder training can run without them. The deployed
app runs from committed model/reference statistics and never requires those
native font files. Evaluation also expects the separately prepared unknown-font
assets described by `scripts/build_font_catalog.py`; source families and seeds
are explicit. No capstone train/validation/test images are read.

Codex assisted with implementation, data preparation, model training, verification
and documentation. No historical capstone accuracy or authorship was changed.
