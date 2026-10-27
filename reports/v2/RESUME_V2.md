# FontSense V2 resume checkpoint

Validation training and the product reliability pass are complete on `v2-20k`.

## Saved data and boundaries

- 20,000 images from 200 families: 14,000 train, 3,000 validation, 3,000 reserved test.
- Manifest: `reports/v2/data/full_manifest.csv`.
- Manifest SHA-256: `95fa9642c8bbc0ecfe6af1d4d1e893ed041a238fa1b6a4da59760f58407132e7`.
- Split: `data/v2/frozen_family_split.csv`.
- Split SHA-256: `f610d835f71ed0935bab8606d7ce46ba85875db25edf94759350c12ceadae250`.
- V1 checkpoint: `artifacts/cnn/cnn_model.pt`.
- V1 checkpoint SHA-256: `c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`.

All fonts, generated images, previous checkpoints, manifests, data reports, configurations, and MLflow records are preserved locally. Bulk generated data and MLflow storage remain ignored by Git.

## Completed validation comparison

| Candidate | Best epoch | Epochs trained | Validation macro F1 | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 8 | 0.678968 | 0.677333 |
| B | 20 | 25 | 0.795823 | 0.797333 |
| C, completed restart | 13 | 18 | 0.760591 | 0.770333 |
| D, lower-learning-rate refinement of B | 4 | 8 | 0.788264 | 0.787667 |

Candidate B remains selected. Additional training did not beat it. The selected V2 checkpoint is `artifacts/v2/cnn/selected/cnn_model_v2.pt`, with SHA-256 `f955f6d0fa3b45ea7c5124a111980cd5d27ec474732d0f5dfe320fde1f9da329`.

Current evidence:

- `reports/v2/cnn/validation_report.md`
- `reports/v2/cnn/candidate_comparison.csv`
- `reports/v2/cnn/mlflow_run_summary.csv`
- `reports/v2/cnn/selected_reload_check.json`
- `reports/v2/product_verification.json`
- `docs/v2/improvement_pass.md`

The original interrupted C attempt reached 0.765198 at epoch 13 but did not persist weights. Its history remains under `reports/v2/cnn/candidate_c/previous_attempts/` and its interruption record is preserved. The original `validation_comparison_checkpoint.csv` is historical stopping-point evidence. The newly completed C run has a real reloadable checkpoint and is the run used in the current comparison. CPU thread settings differ from the historical run; exact training trajectories can differ.

Refinement D completed training, saved outputs, and finished its MLflow run. A relative-output-path bug prevented the final summary write. The path handling is fixed and tested. `scripts/recover_v2_completed_report.py` recovered the summary from the saved checkpoint, predictions, history, and MLflow run `777d06080e3a48eaae6b1841055ace81`, without repeating training or dataset-image inference.

## Test boundary and current application

V2 TEST REMAINS UNTOUCHED

No V2 test image was used for model fitting, selection, early stopping, threshold selection, or evaluation. The initial dataset-quality audit remains separate evidence.

The application still uses the assessed V1 checkpoint and frozen threshold 0.60. Upload and runtime fixes are verified; V2 has not silently replaced that model. V2 threshold selection, pre-test freeze, final test evaluation, V2 packaging, and a production model upgrade remain pending.

## Exact next step

Review the selected V2 validation report and independent real-world examples. Freeze the chosen V2 checkpoint, preprocessing, class order, hashes, and validation-only threshold decision before the single held-out V2 test evaluation. Do not use test results to choose further training settings.

The completed selection can be reproduced without training:

```powershell
.\.venv\Scripts\python.exe -m fontsense.v2_train select --refinement-report-dir reports/v2/refinement
```

Do not rerun Candidate C merely to recreate the earlier score: its completed restart and checkpoint are now saved.
