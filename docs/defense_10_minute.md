# FontSense — 10-minute defense plan

## 0:00–1:00 — Problem and user

“My name is Rustamov Alixan. As a graphic designer, I often receive flattened images where the original font is unknown. Before searching for a similar font, I first need to understand its broad category. FontSense predicts serif, sans serif, display, handwriting, or monospace from a cropped text image.”

Clarify that the system does not identify the exact font.

## 1:00–2:00 — Input, output, and value

Show one example crop and the five probability outputs. Explain the uncertainty warning and intended use as a first guess.

## 2:00–3:15 — Dataset

Explain Google Fonts, open-source licenses, generated images, and variations. Show family and image counts.

## 3:15–4:15 — The key leakage decision

Show a diagram:

- Wrong: Roboto images randomly spread across train and test
- Correct: every Roboto image stays in one split

Explain that family-level splitting tests generalization to unseen typefaces.

## 4:15–5:30 — Models and MLflow

Explain:

1. Majority baseline
2. HOG + Logistic Regression
3. Small CNN

Show the MLflow comparison or exported run table. Explain why you changed one major factor per experiment.

## 5:30–6:45 — Final results

Show the final comparison table, test macro F1, accuracy, per-class results, inference time, and model size. Clearly separate validation model selection from final test evaluation.

## 6:45–7:45 — Error analysis

Show the confusion matrix and 2–3 mistakes. Explain why display/handwriting or serif/display may overlap and how generated data differs from real designs.

## 7:45–8:45 — Live demo

Upload one normal unseen crop, show probabilities, then show one difficult or invalid example. Keep a screenshot or recording as backup.

## 8:45–9:30 — Responsible AI and limitations

Mention licensing, no personal data, subjective labels, Latin-only scope, low-confidence handling, and human oversight.

## 9:30–10:00 — Conclusion

“FontSense demonstrates an end-to-end ML pipeline: legal data acquisition, leakage-safe splitting, two model families, tracked experiments, evaluation on unseen font families, error analysis, and a reproducible demo. The main limitation is the gap between generated text and complex real-world artwork.”
