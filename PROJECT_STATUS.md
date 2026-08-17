# FontSense project status

## Current stage

**FINAL — TECHNICALLY COMPLETE**

FontSense has a complete assessed ML workflow, reproducible evidence, working
Gradio and Colab demos, passing automated verification, and a verified
one-folder Windows standalone distribution. The teacher removed EXTC1 from the
required final scope, so it is not a project blocker or pending task.

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

## Remaining user actions

- Present and defend the project.

No further technical implementation is currently pending.
