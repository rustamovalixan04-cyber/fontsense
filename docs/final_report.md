# FontSense Final Project Report

**Student:** Rustamov Alixan  
**Track:** Individual Project  
**Field:** Graphic Design / Computer Vision

## 1. Executive summary

FontSense is a prototype machine-learning system that predicts the broad category of a typeface from a cropped text image. It is intended as a fast first guess for graphic designers and design students when the original font information is missing.

## 2. Problem statement

Explain why broad typeface-category recognition is useful, who experiences the problem, and why the project does not attempt exact font-family identification.

## 3. ML formulation

- Input: cropped PNG/JPEG with one word or short Latin-script line
- Output: five class probabilities and one predicted category
- Task: supervised multiclass image classification
- Primary metric: macro F1

## 4. Data

Document the exact number of usable families and images per category. Explain Google Fonts source, license recording, family-level splitting, generated text phrases, and augmentation settings.

## 5. Leakage prevention

State that the same family never appears in multiple splits. Explain why image-level random splitting would create unrealistically strong results.

## 6. EDA and data-quality findings

Insert:

- families per category and split
- images per category and split
- example image grid
- image dimensions and corruption checks
- class balance
- label ambiguity and rendering failures

## 7. Modeling

### Majority baseline
Report result and explain its role as a sanity check.

### HOG + Logistic Regression
Explain HOG edge-direction features, experiment parameters, validation results, speed, and model size.

### Small CNN
Explain architecture, augmentation, optimizer, early stopping, validation results, speed, and model size.

## 8. Model selection

Use a comparison table. Select the final model using validation macro F1 plus reproducibility, inference time, and model size—not test performance.

## 9. Final test evaluation

Report the untouched held-out-family results exactly as produced by `reports/*_test_metrics.json` and the classification report. Include the confusion matrix.

## 10. Error analysis

Discuss recurring confusions, difficult font families, blur/rotation effects, low-confidence examples, and cases where a human should ignore the output.

## 11. Responsible AI and limitations

- No personal data is used.
- Font licenses and sources are recorded.
- Categories are conventions and may be subjective.
- The system is not an exact font identifier or licensing authority.
- Results are not validated for non-Latin scripts, mixed fonts, logos, curved text, or heavily edited photographs.
- Low-confidence results require human review.

## 12. Reproducibility

Document installation, dataset generation, training, evaluation, and demo commands. State the clean-Colab test result and repository commit used for submission.

## 13. AI assistance disclosure

AI tools were used for planning, code scaffolding, debugging, and documentation support. All generated components were reviewed, tested, adapted, and must be explainable by the student. Final experimental results were produced by running the submitted repository.

## 14. Conclusion and next steps

Summarize whether the project met its functional goals. Possible next steps: more independent families, real screenshot benchmark, confidence calibration, non-Latin datasets, Grad-CAM, or an Adobe/browser integration.
