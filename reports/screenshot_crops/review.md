# Screenshot crop release review

Proposed commit: `Handle sparse screenshot text and preserve decorative details`

## What changed and why

Readable text inside large blank margins now survives crop validation. Isolated
faint caption remnants no longer distort the main line. Repeated decorative
rows, dots, accents and full lines are preserved. A rejected whole-image
category check can retry its validated crop. The original assessed classifier,
model weights, finite 1,943-family enrollment and name thresholds remain.

The unresolved-match interface gives an honest coverage message and optional
WhatTheFont link without automatically transferring an image. The Moga fixture
reads correctly but stays unnamed because it is not enrolled. No font programs,
credentials, generated query sets or downloaded archives are included.

## Checks and remaining limitations

- 118 local product regression tests passed, including ten new crop tests.
- 41,304 clean enrollment crops match the previous normalized pixels exactly.
- Final separate crop check: 30/52 to 51/52 correct in the first three results;
  19/19 returned likely names correct. Top-one macro F1 0.6557 to 0.9319.
- Ten reused unknown-family regressions withheld a name in all ten cases.
- Local HTTP requests checked the actual Moga fixture and a sparse screenshot.
- Syntax compilation and `git diff --check` passed.
- One final Zilla Slab Highlight tight crop still misses. Historical browser
  source reuse, simulated defects and limited family coverage are disclosed.
- The first decorative-grid regression and initial candidate results are kept
  as history. No worldwide or perfect-recognition claim is made.

GitHub CI and the production build are checked after pushing; their outcome
must not be inferred from the local checks. Source revision and hashes are in
`checks.json`. Codex assisted with code, checking and documentation.

## Files included

- `README.md`
- `api/compact.py`
- `api/font_match.py`
- `api/static/index.html`
- `docs/font_identification.md`
- `docs/screenshot_crops.md`
- `reports/screenshot_crops/checks.json`
- `reports/screenshot_crops/reference_parity.json`
- `reports/screenshot_crops/reference_parity_before_grid_fix.json`
- `reports/screenshot_crops/review.md`
- `reports/screenshot_crops/test2-improved.json`
- `reports/screenshot_crops/test2-legacy.json`
- `reports/screenshot_crops/test3-improved.json`
- `reports/screenshot_crops/test3-legacy.json`
- `reports/screenshot_crops/unknown_regression.json`
- `reports/screenshot_crops/validation-improved.json`
- `reports/screenshot_crops/validation-legacy.json`
- `scripts/evaluate_screenshot_crops.py`
- `tests/fixtures/README.md`
- `tests/fixtures/moga_user_screenshot_crop.png`
- `tests/test_screenshot_crops.py`
