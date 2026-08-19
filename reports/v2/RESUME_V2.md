# FontSense V2 resume checkpoint

This file records the honest stopping point for the `v2-20k` branch. Candidate C was intentionally stopped because of session and token-limit constraints. This is not treated as a training failure, and no further training or evaluation was started.

## Frozen data state

- Full manifest: `reports/v2/data/full_manifest.csv`
- Full manifest SHA-256: `95fa9642c8bbc0ecfe6af1d4d1e893ed041a238fa1b6a4da59760f58407132e7`
- Frozen family split: `data/v2/frozen_family_split.csv`
- Frozen split SHA-256: `f610d835f71ed0935bab8606d7ce46ba85875db25edf94759350c12ceadae250`
- V1 checkpoint: `artifacts/cnn/cnn_model.pt`
- V1 checkpoint SHA-256: `c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`

All downloaded fonts, the generated 20,000-image dataset, manifests, data-validation reports, configurations, Candidate A and B checkpoints, Candidate C history, and local MLflow records remain in place. Generated images and MLflow storage stay ignored by Git as designed.

## Validation evidence at the stopping point

| Candidate | Status | Best epoch | Validation macro F1 | Validation accuracy | Runtime | Parameters | Saved checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | Completed | 3 | 0.678968 | 0.677333 | 303.613 s | 60,901 | `artifacts/v2/cnn/candidate_a/cnn_model.pt` |
| B | Completed | 20 | 0.795823 | 0.797333 | 1,523.761 s | 136,277 | `artifacts/v2/cnn/candidate_b/cnn_model.pt` |
| C | Intentionally stopped | 13 | 0.765198 | 0.764667 | 1,255.635 s so far | 241,605 | Not persisted before interruption |

Candidate B is the current validation leader. No final model selection has been executed.

Candidate C completed 15 logged epochs, with its best validation result at epoch 13. The epoch-13 metrics and complete logged history are preserved in `reports/v2/cnn/candidate_c/history.csv` and MLflow run `bedf251ec4884be38c691fb3f2c6f9ee`. The trainer held the best weights in memory and only serialized at normal run completion, so the interruption left no loadable Candidate C checkpoint. Therefore no Candidate C checkpoint hash or checkpoint file size is claimed. Its 241,605 float32 parameters require 966,420 bytes before checkpoint metadata.

The trainer now atomically saves every new best checkpoint before starting another epoch. This prevents the same loss on a resumed or repeated run.

## Test boundary

V2 TEST REMAINS UNTOUCHED

The Candidate A, B, and C records each report zero test images loaded, zero test rows evaluated, and no test metrics. Threshold selection, pre-test freeze, final test evaluation, packaging, EXE work, and additional validation have not started.

## Exact next step after the limit reset

If a loadable Candidate C checkpoint is required, rerun only Candidate C with:

```powershell
.\.venv\Scripts\python.exe -m fontsense.v2_train C --device cpu
```

The improved trainer will save each new best checkpoint as it trains. After Candidate C has a verified checkpoint, compare A, B, and C using validation macro F1 only. Candidate B remains the leader unless new saved evidence exceeds `0.7958228907353944`. Do not access the V2 test split before model and threshold selection are frozen.
