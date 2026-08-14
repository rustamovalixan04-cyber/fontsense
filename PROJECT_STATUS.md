# FontSense project status

## Current stage

The core FontSense ML project is technically complete and ready for presentation.
The repository is public, the local Gradio app works, and the student manually
tested the Colab demo successfully. The remaining implementation work is behind
the teacher-required EXTC1 human review gate.

## M8C3 Data Gate

**Final human decision: GREEN after manual computer-vision evidence review.**

The unchanged teacher self-check recorded **19 PASS, 1 WARN, and 1 MANUAL** and
therefore suggested **YELLOW** automatically. The warning is the justified
constant `image_size` metadata field. The separate manual CV verification opened
all 3,600 images and checked duplicates, blank/corrupt/missing files, family
isolation, preprocessing, augmentation boundaries, test integrity, hashes, and
evidence paths. It passed, supporting the final human GREEN decision without
changing the teacher validator.

Evidence:

- `reports/data_gate_self_check/data_gate_self_check.md`
- `reports/data_gate_self_check/manual_validation.json`
- `docs/data_audit.md`

## Completed work

- Google Fonts metadata and licence audit:
  `data/interim/google_fonts_manifest.csv`
- Frozen leakage-safe family split:
  `data/interim/google_fonts_final_family_split.csv`
- Reproducible 3,600-image dataset generation and validation:
  `reports/dataset/full_validation_summary.json`
- EDA and image-quality analysis:
  `reports/eda/eda_summary_report.html`
- Majority-class sanity baseline and HOG + Logistic Regression experiments:
  `reports/baseline/baseline_summary_report.html`
- Three validation-only CNN experiments and validation-based model selection:
  `reports/cnn/cnn_experiment_report.html`
- Frozen preprocessing, class order, checkpoint, and confidence threshold:
  `reports/final_evaluation/pre_test_freeze.json`
- One-time held-out evaluation on 600 images from 15 unseen families:
  `reports/final_evaluation/final_evaluation_report.html`
- Gradio application using the frozen final CNN: `app.py`
- Public-repository Colab demo, manually checked by the student:
  `notebooks/07_colab_demo.ipynb`
- Automated tests and GitHub Actions CI: `tests/` and
  `.github/workflows/tests.yml`
- M8C3 Data Gate evidence and final manual decision:
  `reports/data_gate_self_check/`
- Final README and presentation preparation: `README.md` and
  `docs/defense_prep.md`
- EXTC0 no-partner Peer QA review:
  `docs/extc0_peer_qa_review.md`
- EXTC1 initial specification, inspect-only pass, and specification repairs:
  local `docs/feature_spec.md` draft

These are completed saved outputs. This documentation pass did not regenerate
data, train or tune a model, rerun the final test, change the threshold, or
change the final checkpoint.

## Currently blocked or pending

- Exactly two real EXTC1 reviewer or mentor comments
- Owner decisions for both comments
- An approved first implementation task and exact checks
- A Green EXTC1 Specification Gate
- Committing the specification only after that gate is satisfied
- Windows EXE packaging and source/package equivalence verification
- Final live presentation and defense

The Windows EXE does not exist yet. Reviewer names, comments, decisions, and
approval must come from real people and must not be invented.

## Next smallest task

Obtain exactly two genuine comments on the EXTC1 feature specification. Record
the owner's decisions, agree the first small packaging task and its checks, and
make the Specification Gate Green only if the teacher requirements are actually
met. Do not begin Windows packaging before that review.
