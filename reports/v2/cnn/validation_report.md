# V2 validation comparison

These are validation results on 3,000 images from 30 unseen font families. The V2 test images remain reserved. These numbers do not establish accuracy on real screenshots or photographs.

| Candidate | Best epoch | Macro F1 | Accuracy | Training seconds | Checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 3 | 0.678968 | 0.677333 | 303.61 | 255505 |
| B | 20 | 0.795823 | 0.797333 | 1523.76 | 557521 |
| C | 13 | 0.760591 | 0.770333 | 1567.37 | 979729 |
| D | 4 | 0.788264 | 0.787667 | 887.52 | 557521 |

Selected candidate: **B**, using validation macro F1 only.

The selected checkpoint is a separate V2 artifact. It does not replace the assessed V1 checkpoint or change the application's default model.

Per-class results: `reports/v2/cnn/selected_classification_report.csv`.
Family-level results: `reports/v2/cnn/selected_family_accuracy.csv`.
Common confusions: `reports/v2/cnn/selected_confusions.csv`.
Confusion matrix: `reports/v2/cnn/selected_confusion_matrix.png`.

The interrupted original C attempt is preserved separately. The completed C restart is the result compared here; a checkpoint that was never saved is not included in model selection.

V2 TEST REMAINS UNTOUCHED

AI assistance supported implementation, execution, tests, and reporting. The student remains responsible for reviewing the evidence and explaining the models.
