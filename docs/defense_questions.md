# Likely defense questions and strong answers

## Why macro F1 instead of only accuracy?
Every category should matter equally. Accuracy can look good while a smaller or harder class performs poorly. Macro F1 calculates F1 per class and averages them equally.

## What is the target?
One of five Google Fonts-style broad categories: serif, sans serif, display, handwriting, or monospace.

## What is data leakage here?
If images from the same font family appear in training and test, the model can memorize that family. The split is therefore made by family before images are generated.

## Why use HOG?
Typeface categories depend heavily on edge, stroke, and shape patterns. HOG provides a simple, fast, interpretable feature baseline.

## Why compare a CNN?
The CNN learns visual features directly and can capture patterns that hand-designed HOG features may miss.

## Why not identify the exact font?
Exact font identification requires a much larger and different problem definition. The capstone intentionally focuses on a realistic five-class task.

## Why is an API not the main model?
The project trains its own HOG/Logistic Regression and CNN models. Any Gradio or Flask layer is only an interface.

## Why might generated-data performance be misleading?
Generated text is cleaner and more controlled than photos, logos, perspective distortion, texture, and mixed-font designs. A small real-world set is used only as a separate practical check, not as proof of broad production validity.

## How did you select the final model?
Using validation macro F1 plus inference time, model size, stability, and reproducibility. The test set was not used for model selection.

## What would you improve with more time?
Add more independent families, build a labeled real-screenshot benchmark, calibrate confidence, support more scripts, and explore transfer learning or visual explanations.
