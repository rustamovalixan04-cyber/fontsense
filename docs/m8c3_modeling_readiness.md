# M8C3 modeling-readiness plan

This adapts the teacher's `04_M8C3_Modeling_Readiness_Planner.docx` to the real
FontSense project. Because modeling is already complete, it records the
decisions that were actually made rather than presenting them as future work.

## Current gate state

- Data Gate: Green after manual CV evidence review.
- Split: frozen family-level train/validation/test split with zero overlap.
- Primary metric: macro F1, because all five categories matter equally.
- Supporting metrics: accuracy, per-class precision/recall/F1, confusion
  matrices, inference time, and saved model size.
- Evidence: `docs/data_audit.md`,
  `reports/data_gate_self_check/manual_validation.json`, and
  `reports/preprocessing_manifest.json`.

## Baseline and candidate models

| Stage | Purpose | Selection data | Recorded result | Evidence |
|---|---|---|---|---|
| Majority class | Sanity check only | Validation | Macro F1 0.0667; accuracy 0.2000 | `reports/baseline/baseline_validation_summary.json` |
| HOG + multinomial Logistic Regression | Classical pixel-feature baseline | Validation | Best macro F1 0.6933; accuracy 0.6950 | `reports/baseline/baseline_validation_summary.json` |
| Small CNN | Learn spatial image features | Validation | Selected macro F1 0.8331; accuracy 0.8350 | `reports/cnn/cnn_validation_summary.json` |
| Selected small CNN | Single held-out confirmation | Test, once after freeze | Macro F1 0.8653; accuracy 0.8667 | `reports/final_evaluation/final_test_metrics.json` |

The selected model was the Reference small CNN because it had the highest
validation macro F1. Test performance was not used to choose the model.

## Experiment boundary

- Fit models on `train` only.
- Compare configurations and perform early stopping on `validation` only.
- Apply random augmentation to training only.
- Select the uncertainty threshold from validation predictions only.
- Freeze the checkpoint, preprocessing, class order, hashes, seed, and
  threshold before test access.
- Evaluate the complete held-out test split once.
- Do not tune or retrain after reading the test result.

Evidence:
`reports/baseline/baseline_validation_summary.json`,
`reports/cnn/cnn_validation_summary.json`,
`reports/final_evaluation/pre_test_freeze.json`, and
`reports/final_evaluation/evaluation_receipt.json`.

## Readiness conclusion

The Data Gate supported the completed modeling workflow: an honest sanity
baseline, a classical HOG baseline, three meaningful CNN validation
experiments, validation-based selection, and one-time held-out evaluation. The
next work is demonstration and documentation review, not more tuning. The
synthetic-to-real gap remains the main limitation.
