# Screenshot crops and font coverage

The user's Moga example has two separate issues. OCR already reads `Moga
Regular` correctly, but Moga is absent from the 1,943-family reference index.
Cropping that image more tightly cannot create a missing reference. The user's
image is now a regression fixture: readable text, sans-serif alternatives and
no invented exact identity. Font-name words in an image are never accepted as
proof that the lettering was rendered in that font.

## Changes

The shared text normalizer used by product retrieval now estimates foreground
contrast when text occupies less than 0.5% of a screenshot. Previously, the
99.5th percentile could describe only the background and reject readable text.
It also excludes a distant caption fragment from the bounding box only when
the fragment is short, faint, isolated from other occupied rows and small
relative to the main text. Strong dots, accents, repeated decorative grid rows
and complete second lines remain. It changes the crop bounds rather
than erasing glyph pixels inside the main line.

If the category check rejects the original screenshot because resizing it to
112 by 48 makes thin letters disappear, `/identify` retries that check on the
already validated text crop. The frozen `CompactCNN.predict` implementation,
original `/predict` route, assessed checkpoint, retrieval encoder, reference
index and confidence thresholds are unchanged. No new gradient training or
MLflow training run was performed for this preprocessing fix.

An unresolved readable image now says **No verified match**, explains that the
original font may be outside the library, and offers an optional
[WhatTheFont page](https://www.myfonts.com/pages/whatthefont). That is a normal
external link: no image, recognized text, API key or account information is
automatically sent to that service. It is not an integrated identification API
or a guarantee that the external service contains the missing font.

## Selection and separate evaluation

The source images are existing genuine browser crops from the separately
supplied browser benchmark, with source-capture hashes retained in each report.
Four conditions use the original crop, a tight glyph boundary, a simulated
faint partial-caption strip and simulated large blank margins. JPEG quality 95
and an 1800-pixel maximum side match the current public client's preparation.
Only the base images are genuine browser captures; the added defects are
simulations, not new photographs or independently captured screenshots.

Fourteen development families / 56 images selected the initial fragment rule
and category fallback. An initial separate `test2` check covered 15 different
families / 60 images. A later library-wide audit found that 24 of 41,304 clean
enrollment renderings lost faint decorative grid details. That regression led
to the added nearest-band isolation requirement and three explicit tests.

After that correction, settings were frozen again and a fresh `test3` selection
used the remaining 13 families / 52 images from the browser test partition,
excluding all 15 families in the first selection. No settings were changed
after the final results below. These existing source families are held out
from encoder gradient training and disjoint from development families, but
appeared in historical browser audits. This is a separate check for the
current crop change, not a claim of newly collected or globally unseen data.
Known-family enrollment references are necessary for retrieval and are not
query images. Neither corpus is the assessed capstone final test.

| Separate crop check | Previous | Selected change |
|---|---:|---:|
| Correct family in first three results | 30/52 (57.69%) | 51/52 (98.08%) |
| Top-one macro F1 across queried families | 0.6557 | 0.9319 |
| Likely names returned | 10/52 | 19/52 |
| Correct likely names | 10/10 | 19/19 |
| Text correct, ignoring case and spaces | 37/52 | 49/52 |
| Readable inputs rejected | 12/52 | 0/52 |
| Local median elapsed time | 1329 ms | 1368 ms |

The first three results combine an accepted likely name with the displayed
similar-font list, deduplicated. Macro F1 evaluates the first result against
the 13 queried families; rejected queries are false negatives. It is not a
macro score across all 1,943 searchable families. Timing is local CPU execution
with other checks running, not hosted latency or a speed improvement claim.
Each family contributes four related transformations of one crop, not four
independently captured examples. Per-family precision, recall, F1 and confusion
counts are included in the machine-readable reports.

| Condition, 13 images each | Previous top three | Selected top three |
|---|---:|---:|
| Original browser crop | 13/13 | 13/13 |
| Tight boundary | 12/13 | 12/13 |
| Faint partial caption | 4/13 | 13/13 |
| Large blank margins | 1/13 | 13/13 |

The tight-boundary Zilla Slab Highlight image still misses the exact family,
returning Kay Pho Du, Zilla Slab and Life Savers. That error also occurs in the
baseline. It was not used for retuning.

For transparency, initial development was 36/56 to 56/56 top-three with 22/22
correct names, and the initial separate check was 32/60 to 55/60 with 19/19
correct names. Four Big Shoulders Text SC variants and the margin Epunda Sans
variant missed. These reports are retained as pre-grid-fix history and are
not the final selected implementation's evaluation.

The diagnostic unknown regression reuses five previously evaluated non-enrolled
independent families, representing four source groups. Ten fresh-phrase
tight/margin JPEG images all withheld an asserted name. That is a small reused
regression set, not an untouched final false-positive estimate. Full per-image
records, errors and provenance are in
[`reports/screenshot_crops`](../reports/screenshot_crops).

The final enrollment audit reports **0 changed crops out of 41,304 renderings**
after the grid fix. The selected retrieval index therefore retains its existing
reference preprocessing on all checked open-font enrollment phrases. The local
product regression suite passed 118 tests, including ten new crop tests. The
original classifier and deployed model/index hashes are recorded in
[`checks.json`](../reports/screenshot_crops/checks.json).

## Coverage limit

There is no evidence that this index covers every font. Private designs,
commercial fonts without references, custom lettering and visually identical
clones remain unsolved. A short screenshot can also lack enough distinct glyphs
to separate near-identical families. Similarity scores are not probabilities.

[1001Fonts' Moga listing](https://www.1001fonts.com/moga-font.html) points to
[the publisher's Moga page](https://graphicted.com/product/moga-font/), which
describes a wide geometric sans and offers paid license choices. The listing
describes its free download as personal use only. That description is
consistent with this screenshot's broad traits but does not prove the exact
source of the user's image. No Moga binary was downloaded or enrolled. A
publisher license that permits FontSense's intended use, or an appropriately
licensed external recognition service, is needed for broader exact coverage.

## Reproduce

Prepare the existing deployment assets and product environment as documented
in [retrieval training](retrieval_training.md). Extract the separately supplied
browser-corpus ZIP under `../retrieval-training/browser`; generated query images
and source font programs stay outside Git. Then run:

```text
python scripts/evaluate_screenshot_crops.py --split validation --prepare
python scripts/evaluate_screenshot_crops.py --split validation --mode legacy
python scripts/evaluate_screenshot_crops.py --split validation --mode improved
python scripts/evaluate_screenshot_crops.py --split test2 --prepare
python scripts/evaluate_screenshot_crops.py --split test2 --mode legacy
python scripts/evaluate_screenshot_crops.py --split test2 --mode improved
python scripts/evaluate_screenshot_crops.py --split test3 --prepare
python scripts/evaluate_screenshot_crops.py --split test3 --mode legacy
python scripts/evaluate_screenshot_crops.py --split test3 --mode improved
python scripts/evaluate_screenshot_crops.py --audit-references
python scripts/evaluate_screenshot_crops.py --unknown-regression
python -m pytest -q tests/test_screenshot_crops.py
```

The optional unknown regression also requires the independent source audit and
licensed source subsets described in [independent fonts](independent_fonts.md).
`FONTSENSE_CROP_MODE=legacy` restores the previous product crop behavior for A/B
checks or rollback; the default is `improved`. It is a server-side setting, not
an image input. The reproducible reference audit compares normalized pixels
for all open-font enrollment styles, four phrases and three capitalization
forms, and records any differences without rebuilding the existing index.

Codex assisted with diagnosis, implementation, evaluation and documentation.
The project owner remains responsible for understanding and defending it.
