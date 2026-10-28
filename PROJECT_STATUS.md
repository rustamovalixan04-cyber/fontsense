# FontSense project status

## Current stage

**Independent product development: V3 500,000-image data gate passed.**

The owner has expanded FontSense beyond the original course scope. New work is
separate from the historical V1 and V2 evidence. The current product-data plan
uses 1,641 selected Google Fonts families: 350,000 training images, 75,000
validation images, and 75,000 reserved test images. The reviewed split is
`data/v3/frozen_readable_family_split.csv`; the earlier, unapproved split is
preserved separately and is not the training input.

The font audit, split, full generation, and independent image checks passed.
All 500,000 JPEGs reopened; 100 training samples reproduced byte-for-byte. There
are zero remaining exact pixel duplicates and zero family overlap. See
`reports/v3/dataset_validation_summary.json` and
`reports/v3/independent_image_verification.json`. Neither V2 nor V3
test images have been model-evaluated in this product pass. The app continues to
use the original V1 checkpoint and threshold.

Visual review excluded 11 barcode/redaction utility fonts. Exact Latin-glyph
aliases spanning historical V2 folds were also quarantined; six historical V2
families are in three affected groups. Historical metrics remain unchanged but
need that qualification. See `reports/v3/historical_alias_risk_summary.json`.

On the new 75,000-image validation set, the existing Candidate B scored macro
F1 0.668795 and accuracy 0.667507. This is a different, broader validation set
than V2; do not compare its score directly with V2's old validation score.
The next step is validation-only GPU training and comparison on these same new
families. The source suite passes 131 Python tests and four JavaScript tests.
An independent real-image evaluation remains necessary before broad product
accuracy claims. Details: `docs/product/v3_data.md`.

### Preserved historical stage

V1 assessed workflow complete; V2 validation comparison and product reliability
pass complete.

FontSense has a complete assessed ML workflow, reproducible evidence, working
Gradio and Colab demos, passing automated verification, and a verified
one-folder Windows standalone distribution. The teacher removed EXTC1 from the
required final scope, so it is not a project blocker or pending task.

The separate `v2-20k` branch contains 20,000 rendered images from 200 audited
families: 14,000 training, 3,000 validation, and 3,000 reserved test images.
The V1 assessed results and frozen checkpoint remain preserved. V2 validation
results must not be described as final held-out results or as measured
real-world accuracy. Current work is recorded in
`reports/v2/v2_pipeline_state.json` and `reports/v2/RESUME_V2.md`.

The completed C restart reached validation macro F1 0.760591. A separate
lower-learning-rate refinement of B reached 0.788264. Neither beat B, which
remains selected at 0.795823 macro F1 and 0.797333 accuracy. Its V2 checkpoint
and comparison evidence are saved in `reports/v2/cnn/selected_model.json` and
`reports/v2/cnn/validation_report.md`. V2 test images remain reserved.

The current app retains the assessed V1 model, with verified upload fixes,
transparent-image handling, bounded request sizes, and stale-request protection.
That earlier pass had 111 Python tests and four JavaScript tests. Technical
HTTP smoke checks pass; visual browser verification was blocked by the Windows
computer-use tool. See `reports/v2/product_verification.json`.

## M8C3 Data Gate

**Final human decision: GREEN after manual computer-vision evidence review.**

The unchanged teacher self-check recorded **19 PASS, 1 WARN, and 1 MANUAL** and
therefore suggested **YELLOW** automatically. The warning is the justified
constant `image_size` metadata field. The separate manual CV verification
opened all 3,600 images and checked duplicates, blank/corrupt/missing files,
family isolation, preprocessing, augmentation boundaries, test integrity,
hashes, and evidence paths. It passed, supporting the final human GREEN decision
without changing the teacher validator.

Evidence:

- `reports/data_gate_self_check/data_gate_self_check.md`
- `reports/data_gate_self_check/manual_validation.json`
- `docs/data_audit.md`

## Completed work

- Google Fonts metadata and licence audit
- Frozen leakage-safe family split
- Reproducible 3,600-image dataset generation and validation
- EDA and image-quality analysis
- Majority-class and HOG + Logistic Regression baselines
- Three validation-only CNN experiments and validation-based model selection
- Frozen preprocessing, class order, checkpoint, and confidence threshold
- One-time held-out evaluation on 600 images from 15 unseen families
- Confidence, uncertainty, and error analysis
- Gradio application using the frozen final CNN
- Reproducible Google Colab demo
- Automated tests and GitHub Actions CI
- M8C3 Data Gate evidence and final manual decision
- Final README, report, and presentation/defense materials
- Verified one-folder Windows x64 standalone package
- Five-image source/package prediction equivalence at `1e-5` tolerance

The final Windows package evidence is in
`reports/windows_package_equivalence.json`. It confirms the bundled checkpoint
hash, frozen contract, package contents, five exact source/package comparisons,
and a successful local HTTP server smoke test. The build process did not
regenerate data, retrain or tune a model, rerun the final held-out evaluation,
or change the threshold, class order, preprocessing, architecture, or final
checkpoint.

## Remaining historical user actions

- Present and defend the project.

V2 model comparison and its final evaluation remain separate from the completed
V1 assessed workflow. The larger dataset alone does not establish an improvement
on real screenshots or photographs.
