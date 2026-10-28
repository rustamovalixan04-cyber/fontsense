# Resume the product V3 pipeline

Work on the current `v2-20k` branch; do not merge to `main` or overwrite V1/V2.
This pass targets 500,000 JPEGs from the reviewed 1,641-family split, not a
claim of worldwide product readiness.

## Inspect before running

Read `AGENTS.md`, `PROJECT_STATUS.md`, `reports/v3/progress.json`, and any existing
`reports/v3/dataset_validation_summary.json` / `reports/v3/training/summary.json`.
Check Git status and whether a generator or trainer is already running. Do not
launch two processes against the same shard output or training-state file.

The final selected split is `data/v3/frozen_readable_family_split.csv`, SHA-256:
`45fb0eb423b3ee3fcce782d0b83beac9cdf1027e83ace758639357d2d67ed7c8`.
The older `data/v3/frozen_family_split.csv` is a rejected pre-review draft.

## Continue safely, in order

```powershell
.\.venv\Scripts\python.exe -u -m fontsense.v3_dataset --workers 4
.\.venv\Scripts\python.exe -u scripts/verify_product_dataset.py
.\.venv\Scripts\python.exe scripts/create_product_contact_sheets.py
.\.venv-gpu\Scripts\python.exe -u scripts/compare_product_validation.py
.\.venv-gpu\Scripts\python.exe -u -m fontsense.v3_train
.\.venv-gpu\Scripts\python.exe -u scripts/compare_product_validation.py
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/web_client.test.cjs
```

If `data/v3/STOP` exists, read its reason and the real error before removing only
that marker to resume. Do not delete completed shards, raw fonts, or manifests.
Generation checks its fingerprints and hashes before reusing a completed shard.
After a code/config change it regenerates incompatible shards rather than
pretending an old manifest belongs to the new generator.

The training command refuses to run before the full data gate passes. It streams
only train/validation shards. Best model:
`artifacts/v3/cnn/product_candidate.pt`. Complete last-epoch resume state:
`artifacts/v3/training_state/latest.pt` (ignored, preserve locally). MLflow:
`reports/v3/mlruns/`. Stop marker: `reports/v3/training/STOP`; the next epoch
will not start. The same training command resumes from a complete state when
the contract matches; do not edit its source/config while the run is active.

## Integrity and release boundaries

V2 TEST REMAINS UNTOUCHED

The new test images may be decoded for integrity checks, but neither V2 nor V3
test images have been used for model fitting, selection, or prediction in this
pass. No product test score or uncertainty calibration is claimed yet. Select
and freeze a candidate using validation before any one-time held-out evaluation.
Do not promote a candidate just because it trained on more images.

Keep V1 SHA-256:
`c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`.
Keep V2 B SHA-256:
`f955f6d0fa3b45ea7c5124a111980cd5d27ec474732d0f5dfe320fde1f9da329`.

Historical alias risk is documented, not erased:
`reports/v3/historical_alias_risk_summary.json`. The pre-review bulk data at
`data/v3/cache/pre_review/` is preserved but never used for product training.

Commit only checked source/configs, small audits/reports/contact sheets, tests,
and a verified compact candidate checkpoint. Do not commit generated bulk data,
GPU environments, caches, resume state, or MLflow databases. Push normally to
the current branch; never force-push or silently merge into `main`.
