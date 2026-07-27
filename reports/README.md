# Reports

`reports/proof/` contains a verified local system-font proof run and must not be presented as final assessed results. Final Google Fonts runs write their metrics, predictions, error analysis, and figures directly into `reports/`.

`reports/preview/` contains the manifest, validation summary, and five category
contact sheets for the 180-image Google Fonts preview. These files document
data-generation checks only; they are not final dataset statistics or model
performance results.

`reports/dataset/` contains the 3,600-row full image manifest, validation
summary, exact category-effect balance table, same-seed reproducibility check,
and five small contact sheets. These are verified dataset-generation records,
not model evaluation results.

`reports/eda/` contains the pre-training data-quality evidence: per-image
measurements, balance tables, duplicate-screen results, a validation summary,
the canonical report input, and the self-contained HTML summary report.

`reports/baseline/` contains validation-only majority and HOG experiment
records, MLflow run identifiers, per-class results, predictions, the canonical
report input, and the self-contained HTML summary report. These are not final
test metrics.

`reports/cnn/` contains validation-only CNN experiment records, learning
curves, confusion matrices, MLflow run identifiers, the HOG comparison, and
the self-contained CNN experiment report. These are not final test metrics.

`reports/figures/` contains the EDA charts and sample grids. Every important
chart has a short conclusion in `notebooks/03_eda.ipynb` and in the HTML
report. It also contains the validation-only HOG confusion matrix.
