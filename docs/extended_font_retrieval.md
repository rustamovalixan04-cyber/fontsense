# Broader font identification

The exhaustive matcher described here is retained as a fallback. The newer
[trained retrieval stage](retrieval_training.md) selects references first,
compares individual letter shapes where separation is reliable, fits supported
variable axes, and applies separately selected name acceptance rules.

The name gallery introduced earlier was finite: 86 families, with a deliberately
strict confidence gate. A request to identify every font cannot be satisfied
literally. Private fonts, unreleased designs, scripts outside the supported Latin
scope, nearly identical clones and screenshots containing too few distinctive
letters make universal exact recognition impossible to guarantee.

This update addresses the practical problem by searching all eligible OFL Latin
families in a pinned Google Fonts source tree. It adds direct comparisons of the
**actual visible text** rather than trying to name thousands of fonts from the
small encoder alone. The original assessed classifier, checkpoint, five-class
probabilities, category combiner and assessed evaluation files are preserved.

## Catalog and licensing

`artifacts/web/extended_catalog.json` records the pinned revision, actual family
and style counts, source SHA-256 values, license SHA-256 values, reproducible
rendering-subset hashes, and exclusions with reasons. Discovery checks all 1,994
OFL metadata records in revision
`6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec`; 1,847 declare a Latin subset.
The resulting manifest contains **1,840 families / 3,288 styles / 2,622 source
assets**. Only successfully audited families are counted in the manifest.
Apache/UFL/other licenses are not silently treated as OFL. Fonts lacking the
basic Latin alphabet or supported reference styles are excluded explicitly.

To keep deployment size practical, fontTools 4.61.1 creates U+0020..00FF rendering
subsets. Internal family/PostScript names are changed to `FSReference` plus a
source identifier, respecting possible Reserved Font Names. Copyright, license,
attribution and original source links remain available. The subset programs and
OFL text are downloaded/derived during builds, ignored by Git and not served to
the browser. The product's family labels describe the original reference source;
they are not the modified program's internal names.

Build dependencies explicitly include lxml 6.1.1: the six SVG-bearing source
assets require it for fontTools subsetting. It is required in both hosting and
CI rather than relying on its incidental presence in a training environment.

This follows the [OFL modification FAQ](https://openfontlicense.org/ofl-faq/)
and the source [Google Fonts repository](https://github.com/google/fonts).
No Windows font program is copied, subset, downloaded or deployed. The earlier
native whole-word feature statistics remain available; see
[font-name provenance](font_identification.md).

## Matching and uncertainty

The pipeline reads the text (or accepts a user's text correction), normalizes
polarity and straightens a small rotation. It renders all supported styles at
48px and compares low-resolution normalized image descriptors. It then reranks
the 140 best styles using the earlier higher-resolution full-line and
spacing-insensitive comparisons. Including every style in the first stage
prevents a family's regular weight from hiding its much closer bold or italic
reference. This is a two-stage search, not a new neural
training run. Whole reference lines are rendered locally on the server; requests
do not download fonts or send images to a third-party identification service.

A same-text match is labelled **Likely font** only when its retrieval score is at
least 0.96, its different-family margin is at least 0.025, and enough distinct
letters are visible. These conservative engineering gates are not calibrated
probabilities or proof of identity. They must be assessed through the product
retrieval audit, separately from the old encoder's 98.55% selective result.
The earlier encoder is retained for native references. Its OFL-name decisions
are superseded by the broader rendered comparison; its 86-way confidence gates
do not account for thousands of new lookalikes. Disagreement between a confident
native name and the new rendered comparison withholds the name.

Close results now show **Possible fonts** with linked candidate names, followed
by **Similar fonts** and real previews. Times New Roman and Tinos, Arial and
Arimo, and Courier New and Cousine can be particularly hard to separate in short
images; the UI must not present an open compatible alternative as verified
identity. A confident Impact result remains separate from its alternatives, but
the feature is not specific to Impact.

The known compatible-design pairs are explicitly documented alternatives, not
newly trained or verified native-font identities. For example, upstream describes
[Tinos as metric-compatible with Times New Roman](https://raw.githubusercontent.com/google/fonts/6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec/ofl/tinos/DESCRIPTION.en_us.html)
and [Carlito as metric-compatible with Calibri](https://raw.githubusercontent.com/google/fonts/6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec/ofl/carlito/DESCRIPTION.en_us.html).
The app presents the pair as possible when the open reference is closest; it
does not infer a vendor from width compatibility alone.

The original 56-family reference collection stays fixed for category-combiner
features. The broader search is a separate retrieval stage. Preview IDs must
resolve to a known record; arbitrary filesystem paths are rejected. Caches are
bounded, and concurrent fills on a worker are serialized to avoid duplicate
large scans. The first request for a new text phrase costs more than a warm
request; speed must be measured on the hosted deployment.

## Reproduction and evaluation

The final fresh audit sampled **100 different gallery families**, excluded the
100 development-sample families, and used two new phrases with seed 9103.
Across 600 known-family images and 30 images from five fonts outside the catalog:

| Result | Count |
|---|---:|
| Names returned | 295 |
| Correct returned names | 295 |
| Known-image name coverage | 49.17% |
| Known-family top-three retrieval | 88.17% |
| Unknown-image names withheld | 30/30 |

Zero naming errors in this finite check is **not 100% worldwide accuracy**.
Roughly half the known-font queries still do not receive a single likely name.
The check supplies the visible text, uses synthetic rendering and covers only a
sample of the gallery. OCR, screenshots, clones, other styles and arbitrary
unknown fonts remain limitations. Synthetic elapsed times were recorded while
other local verification ran and should not be used as hosting latency claims.

The earlier development audit exposed errors caused by omitting bold styles in
coarse search and by applying the old 86-way encoder's OFL naming gate to a much
larger universe. Those results are development evidence; the fresh family/phrase
audit above was run after the corrections. The assessed category benchmark was
not read or rerun.

Real MLflow retrieval evaluation run: `aaf98796b271419e8ead490ed008b367`.
It records catalog/search parameters, metrics and the actual fresh audit output;
it is not represented as a neural-training run.

```text
python -m pip install -r requirements-vercel.txt
python scripts/build_extended_catalog.py
python scripts/prepare_web_assets.py
python scripts/evaluate_extended_matching.py --families 100 --output work/extended-evaluation.json
python scripts/evaluate_extended_matching.py --families 100 --fresh --seed 9103 --output work/extended-final-evaluation.json
python -m pytest tests/test_extended_matching.py tests/test_font_matching.py tests/test_font_identity.py -q
```

`--resume` on the catalog builder is an explicit local recovery option for an
interrupted audit; ordinary builds regenerate derived bytes from pinned sources.
The deployment preparation script verifies source, transformed font and license
hashes and fails on a mismatch. It uses the committed catalog instead of
repeating discovery, making catalog changes reviewable.

The audit samples gallery families with seed 7211, selects real styles, renders
query phrases at 73px, and checks clean, dark JPEG and 2-degree rotation. It also
checks locally available fonts absent from the expanded catalog. It measures
precision **among names returned**, name coverage and top-three retrieval. Those
are controlled known-catalog product checks with supplied text, not blind OCR
accuracy, an untouched category-classification test, or worldwide accuracy.
Screenshot regressions and public browser tests separately check the OCR path.

Codex assisted with catalog preparation, matching, verification and documentation.
No new neural training, historical capstone accuracy or human authorship is
claimed for this retrieval expansion.
