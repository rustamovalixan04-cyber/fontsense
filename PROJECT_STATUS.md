# FontSense project status

## M8C3 Data Gate

**Status: GREEN after manual computer-vision evidence review.**

The unchanged teacher self-check suggests YELLOW because a constant metadata
column is present and the generic notebook leaves CV file checks for manual
review. The separate full-image validation resolves the manual checks without
altering the validator. Evidence:
`reports/data_gate_self_check/data_gate_self_check.md` and
`reports/data_gate_self_check/manual_validation.json`.

## Current stage

The project is at completed-model demonstration and documentation review. Data
generation, EDA, validation experiments, frozen model selection, and the single
held-out test evaluation are complete.

## Completed work

- Google Fonts metadata and licence audit:
  `data/interim/google_fonts_manifest.csv`
- Frozen leakage-safe family split:
  `data/interim/google_fonts_final_family_split.csv`
- Reproducible 3,600-image dataset and EDA:
  `reports/dataset/full_validation_summary.json` and
  `reports/eda/eda_summary_report.html`
- Majority and HOG experiments:
  `reports/baseline/baseline_summary_report.html`
- Three validation-only CNN experiments and model selection:
  `reports/cnn/cnn_experiment_report.html`
- One-time held-out test evaluation:
  `reports/final_evaluation/final_evaluation_report.html`
- Frozen CNN Gradio application: `app.py`
- Reproducible teacher demo: `notebooks/07_colab_demo.ipynb`

These are real completed outputs, not proposed work. No dataset generation,
training, tuning, or final-test rerun was performed for the M8C3 integration.

## Current blocker

The technical Data Gate has no blocking issue. The remaining operational
blocker is teacher access to the Colab demo if the GitHub repository stays
private. A teacher must have repository access, or the repository must be made
public before sharing the notebook. Manual Colab verification also remains the
student's responsibility.

## Next smallest task

Run `notebooks/07_colab_demo.ipynb` from top to bottom in a fresh Colab runtime
with teacher-equivalent repository access, record only real observations, and
then perform a final README/report/defence review. Do not retrain or revisit the
held-out test result.
