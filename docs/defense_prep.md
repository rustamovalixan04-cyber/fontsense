# FontSense defense preparation guide

This guide uses the saved FontSense evidence and adds no new experiment. The
teacher removed EXTC1 from the required final scope. FontSense proceeded
directly to final technical packaging and verification.

## A. 30-second summary

“FontSense is my computer-vision capstone project. It accepts a cropped image of
Latin text and predicts one of five broad font categories: display,
handwriting, monospace, sans serif, or serif. I compared a majority baseline,
HOG with Logistic Regression, and a small CNN. The CNN was selected using
validation macro F1, then evaluated once on 600 images from 15 unseen font
families. It achieved 86.67% accuracy and 0.8653 macro F1. The tool is a first
guess for a broad category, not an exact font-family identifier.”

## B. Full 5–10 minute presentation script

### 1. Introduction

Hello, my name is Rustamov Alixan, and my capstone project is FontSense. It is a
computer-vision system that looks at a cropped image containing Latin text and
predicts a broad font category. The five possible categories are display,
handwriting, monospace, sans serif, and serif.

### 2. Problem

Designers often receive a screenshot, poster, or flattened image without the
original editable file. The exact font information may be missing. Searching
through thousands of fonts immediately is difficult, so identifying the broad
category can be a useful first step. FontSense does not claim to identify the
exact family or style. It gives a category prediction, five probabilities, and
an uncertainty status.

### 3. Why I chose the project

I chose this problem because it connects graphic design with machine learning.
It is visual, practical, and small enough for an AI/ML Fundamentals capstone,
but it still includes important real ML decisions such as data licensing,
leakage prevention, model comparison, validation-based selection, error
analysis, and deployment through a simple interface.

### 4. Dataset

I used font files and official category metadata from Google Fonts. The audit
recorded the family, category, source, licence, font file, Latin support, and
validation status. I selected 90 usable families: 18 in each category. I then
generated 40 readable text images per family, giving 3,600 images in total.
The source images are 224 by 96 pixels. They contain short English words or
phrases with controlled variation in size, position, background, contrast,
spacing, rotation, blur, compression, scale, and translation. These effects
were scheduled independently of the category so that an effect could not reveal
the answer.

### 5. Family-level split

The most important data decision was to split by font family before image
generation. There are 60 training families, 15 validation families, and 15 test
families. If I randomly split individual images, the same font family could
appear in both training and test data. The model might then memorize the shapes
of that family. Keeping each family in only one split makes the final test a
stronger check on previously unseen families. The audit confirmed zero family
overlap.

### 6. Development process

I first audited the fonts and licences, froze the family split, generated the
dataset, and ran EDA and image-quality checks. All 3,600 manifest images existed
and opened successfully. The checks found no blank, corrupted, missing, or
exact-duplicate images. I then trained and compared models using training data
for fitting and validation data for experiment comparison. I used MLflow to
record parameters, metrics, runtimes, and artifacts. Only after the final CNN,
preprocessing, class order, checkpoint, and threshold were frozen did I perform
the one-time held-out test evaluation.

### 7. Models compared

The first model was a majority-class baseline. It always predicted the same
training category and achieved 20% validation accuracy and 0.0667 macro F1.
This was only a sanity check.

The second approach used HOG features with multinomial Logistic Regression.
HOG describes local edge and stroke directions, which are relevant to letter
shapes. The best HOG experiment achieved 69.5% validation accuracy and 0.6933
macro F1.

The third approach was a small convolutional neural network. Unlike HOG, the
CNN learned visual features directly from image pixels. The selected CNN
achieved 83.5% validation accuracy and 0.8331 validation macro F1, so it was
selected without looking at test results.

### 8. Selected CNN

The selected Reference small CNN uses grayscale images resized to 112 by 48,
normalised with mean 0.5 and standard deviation 0.5. It starts with 16 filters,
uses four convolutional layers, batch normalisation, ReLU activations, pooling,
adaptive average pooling, dropout of 0.25, and a final linear layer for five
classes. Training used a learning rate of 0.001, fixed seed 42, mild
training-only augmentation, and early stopping. The saved checkpoint came from
epoch 14 based on validation macro F1.

### 9. Final result

The final test contained 600 images from 15 unseen font families, with 120
images and three families per category. The CNN achieved 86.67% accuracy and
0.8653 macro F1: 520 predictions were correct and 80 were incorrect. The saved
model is 255,505 bytes, and recorded CPU inference averaged 7.66 milliseconds
per image. Sans serif was the weakest category, with 65.83% recall. Common
errors included sans serif predicted as monospace and serif predicted as
monospace.

### 10. Confidence threshold

The confidence threshold is 0.60. It was selected from validation predictions,
not test results. If the highest probability is at least 0.60, the app marks
the result accepted. Otherwise it shows an uncertain first guess. The threshold
does not change the probabilities or predicted category. It only changes the
status. Confidence is not a guarantee: the final error analysis includes some
confident mistakes.

### 11. Tools

The main tools were Python, PyTorch and torchvision for the CNN, Pillow and
NumPy for images, scikit-image for HOG, scikit-learn for Logistic Regression and
metrics, pandas for manifests, MLflow for experiment tracking, Gradio for the
interface, pytest for automated checks, Google Colab for the reproducible demo,
and Git and GitHub for version control and CI.

### 12. How inference works

When a user uploads an image, the app first checks that the input is a readable,
non-blank image of a safe size. It applies EXIF orientation correction, converts
the image to grayscale, resizes it to 112 by 48, converts it to a tensor, and
normalises it. The CNN extracts learned stroke and shape features and outputs
five scores. Softmax converts them into five probabilities. The largest
probability gives the predicted category, and the 0.60 threshold gives the
accepted or uncertain status.

### 13. Advantages

The project protects against family leakage, compares simple and learned
models, evaluates on unseen families, uses a small fast CPU model, records
uncertainty, and keeps reproducible evidence. The app and Colab notebook reuse
the exact frozen model and preprocessing from the final evaluation. FontSense
also has a verified one-folder Windows standalone build that packages the exact
frozen final CNN.

### 14. Limitations

The biggest limitation is the synthetic-to-real gap. Clean rendered text is not
the same as a phone photo, poster, textured design, curved logo, or compressed
screenshot. Unfamiliar fonts can fail, and a mistake can still have high
confidence. Sans serif was the weakest class. The current scope is Latin text
and broad categories only; it does not recognise an exact family, weight,
style, or licence.

### 15. Demo transition

Now I will show the project showcase briefly and then open the Gradio app. I
will upload one readable crop, explain the category, confidence, five
probabilities, and status, and then show how an invalid or uncertain input is
handled. This demo illustrates the interface; one example is not performance
evidence.

### 16. Future work

Future work could add a labelled benchmark of real screenshots and photographs,
more independent families, better confidence calibration, more writing
systems, and a separate exact-family recognition task.

### 17. Conclusion

FontSense demonstrates a complete educational ML workflow: legal font audit,
reproducible synthetic data, leakage-safe family splitting, EDA, baseline and
CNN comparisons, validation-based selection, one-time held-out evaluation,
error analysis, uncertainty handling, tests, and a working demo. Its final
result is encouraging for unseen Google Fonts families, but it should be used
as an honest first guess rather than a production font identifier.

## C. Data leakage explanation

**Why did you split by font family?**

If generated images were randomly split by image, the same font family could
appear in training and testing. The words or effects could differ, but the
family-specific letter shapes would remain very similar. The model might
memorise those shapes and produce an unrealistically strong test result.
FontSense therefore assigns every image from one family to exactly one split.
The 15 test families are unseen during fitting and selection, which gives a
stronger generalisation check.

## D. Live demo flow

Use this 1–2 minute sequence:

1. Open the existing project showcase and point out the problem, five classes,
   and final result. Do not spend more than 20 seconds there.
2. Open the existing Gradio app with `python app.py` and confirm that **Final
   CNN** is selected.
3. Upload one prepared, readable Latin-text crop that is not blank or heavily
   distorted.
4. Press **Predict** and say that the category is the model's broad first guess,
   not an exact family name.
5. Point to the confidence percentage. Explain that it is the largest of the
   five probabilities.
6. Point to all five probabilities and explain that they sum to approximately
   one.
7. Explain that 60% or higher is accepted and below 60% is uncertain. The
   threshold does not change the probabilities.
8. If time allows, show a prepared blank/invalid crop or a known uncertain
   example. Explain the error message or warning without judging model quality
   from that one example.
9. Return to the conclusion: the app is useful as a first guess, with human
   review still required.

Keep the Colab notebook open as a backup. A Colab share link is temporary and
stops when the runtime closes.

## E. Tools

| Tool | Purpose |
|---|---|
| Python | Main language for data, training, evaluation, tests, and inference |
| PyTorch | Defines, trains, saves, and runs the small CNN |
| torchvision | Provides deterministic image transforms and training augmentation |
| Pillow | Opens, validates, renders, and converts images |
| NumPy | Pixel arrays, numerical checks, and probability handling |
| pandas | Reads and validates manifests, splits, predictions, and reports |
| scikit-image / HOG | Extracts classical edge and stroke-direction features |
| scikit-learn | Logistic Regression, metrics, reports, and confusion matrices |
| MLflow | Records meaningful experiment parameters, metrics, runtimes, and artifacts |
| Gradio | Provides the local and Colab upload interface |
| Google Colab | Runs the reproducible CPU demo without the generated dataset |
| Git | Tracks focused source and documentation history |
| GitHub | Hosts the public repository and runs GitHub Actions CI |
| pytest | Runs the automated project test suite |

## F. How the CNN works

```text
uploaded image
→ input validation
→ EXIF correction and grayscale
→ resize to 112×48
→ normalize with mean 0.5 and standard deviation 0.5
→ convolution layers
→ learned stroke and shape features
→ five probabilities
→ predicted broad category
→ accepted or uncertain status at threshold 0.60
```

**What is a CNN?** A convolutional neural network is a model designed to learn
visual patterns from pixels. Its filters first learn small features such as
edges and strokes, and deeper layers combine them into more useful shapes. The
final layer produces a score for each of the five font categories.

## G. Model progression

The progression was deliberately simple:

1. **Majority baseline:** proves that a real model must beat a trivial 20%
   accuracy rule on this balanced five-class problem.
2. **HOG + Logistic Regression:** tests whether hand-designed edge and stroke
   features can solve much of the problem with a small classical model.
3. **Small CNN:** tests whether learned spatial features improve on HOG while
   staying understandable and small enough for Colab CPU inference.

The CNN was selected because it had the highest **validation macro F1**:
0.8331, compared with 0.6933 for HOG and 0.0667 for the majority baseline.
The test set was not used for that choice. It was opened only after the model,
preprocessing, class order, checkpoint, and threshold had been frozen.

## H. Metric explanations

- **Accuracy:** the fraction of all predictions that are correct.
- **Precision:** for one predicted class, the fraction of those predictions that
  truly belong to that class.
- **Recall:** for one true class, the fraction that the model successfully
  finds.
- **F1:** the harmonic mean of precision and recall for one class.
- **Macro F1:** calculate F1 separately for every class, then average the five
  values equally.
- **Confusion matrix:** a table showing every true class against every predicted
  class, so repeated error directions are visible.

**Why macro F1?** Every font category matters equally. Macro F1 gives the weak
classes the same weight as the strong classes and balances precision with
recall. Accuracy is still reported, but it can hide a class with poor recall,
such as sans serif in the final test.

## I. Advantages

1. **Leakage protection:** complete font families, not individual images, define
   the train, validation, and test splits.
2. **Meaningful comparison:** the project progresses from a trivial baseline to
   HOG + Logistic Regression and then a CNN.
3. **Unseen-family evaluation:** the final 600 images come from 15 families that
   were not used for fitting, model selection, early stopping, or threshold
   selection.
4. **Small and fast:** the CNN is 255,505 bytes and recorded 7.66 ms per image
   during the final CPU test.
5. **Honest uncertainty and reproducibility:** the app applies the frozen 0.60
   threshold, verifies the checkpoint hash, and reuses saved preprocessing and
   class order.

These advantages support an educational prototype. They do not make FontSense
production-ready.

## J. Limitations

- The training and assessed test images are synthetic renders, so real photos,
  screenshots, textures, perspective, and layouts can differ.
- Unfamiliar designs can fail, including with high confidence.
- Sans serif was the weakest evaluated category, with 65.83% recall.
- Serif, sans serif, and monospace can share geometric or stroke features.
- The validated scope is Latin text only.
- The output is one of five broad categories, not an exact family or style.
- It is not an OCR system or a font-licensing authority.

**Is 86.67% good enough?** It is a strong capstone result on the defined held-out
synthetic benchmark because the families were unseen and macro F1 was 0.8653.
It is not proof that the model is reliable on every real design. A separate,
representative real-world benchmark and stronger calibration would be needed
before production use.

## K. Development timeline

1. Defined the five-class educational scope.
2. Audited official Google Fonts metadata, files, Latin support, and licences.
3. Selected 90 usable families, balanced across five categories.
4. Froze a 60/15/15 family-level train/validation/test split.
5. Generated and validated 3,600 readable images with seed 42.
6. Completed EDA, balance checks, duplicate checks, and leakage checks.
7. Established the majority-class sanity baseline.
8. Ran meaningful HOG + Logistic Regression experiments with MLflow.
9. Ran three small CNN experiments with MLflow.
10. Selected the Reference small CNN using validation macro F1.
11. Froze the checkpoint, preprocessing, class order, seed, and threshold.
12. Evaluated once on the complete held-out test split.
13. Connected the frozen CNN to Gradio.
14. Added and manually checked the Colab demo route.
15. Added automated tests and GitHub Actions CI.
16. Completed the M8C3 Data Gate evidence and manual CV decision.
17. Completed final technical documentation and defense preparation.

## L. 30 defense questions

### 1. Why did you choose FontSense?

**SHORT ANSWER:** It connects my graphic-design interest with a manageable
computer-vision classification problem.

**LONGER ANSWER IF ASKED:** Designers often see flattened artwork without font
metadata. A broad category can narrow the search space. The task also allowed me
to demonstrate data licensing, leakage prevention, baseline comparison, CNN
training, evaluation, and deployment within a realistic capstone scope.

### 2. Does FontSense find the exact font?

**SHORT ANSWER:** No. It predicts one broad category, not an exact family or
style.

**LONGER ANSWER IF ASKED:** Exact-family identification is a different and much
larger problem requiring many more labelled families and styles. FontSense
returns display, handwriting, monospace, sans serif, or serif as an educational
first guess.

### 3. What are the five target categories?

**SHORT ANSWER:** Display, handwriting, monospace, sans serif, and serif.

**LONGER ANSWER IF ASKED:** The frozen class order is exactly that order. The
CNN and app both verify it against the saved evaluation contract before
inference.

### 4. Where did the data come from, and was it legal to use?

**SHORT ANSWER:** Font files and category metadata came from official Google
Fonts sources, with each family's source and licence recorded.

**LONGER ANSWER IF ASKED:** The audit checked Latin support, opened and rendered
the selected files, and recorded family, category, source, licence, file path,
and validation status. The project code is MIT licensed, while every font keeps
its own recorded open-source licence.

### 5. Why did you generate synthetic images?

**SHORT ANSWER:** Font files provide known labels and make a balanced,
reproducible dataset possible.

**LONGER ANSWER IF ASKED:** I rendered short Latin phrases with controlled
variation in size, position, contrast, background, spacing, rotation, blur,
compression, and scale. Seed 42 and a frozen configuration make the process
reproducible. The trade-off is the synthetic-to-real gap.

### 6. Why did you split by font family?

**SHORT ANSWER:** To stop the same family's shapes appearing in both fitting and
evaluation data.

**LONGER ANSWER IF ASKED:** An image-level split could let the model memorise a
family even when the phrase or effect changes. Family splitting keeps every
family in one split and tests generalisation to completely unseen families.

### 7. What are the split numbers?

**SHORT ANSWER:** 60 training families, 15 validation families, and 15 test
families.

**LONGER ANSWER IF ASKED:** There are 90 families total and 40 images per
family. That gives 2,400 training images, 600 validation images, and 600 test
images. Each split is balanced across all five categories.

### 8. What is the purpose of train, validation, and test data?

**SHORT ANSWER:** Train fits the model, validation selects it, and test measures
the frozen final choice once.

**LONGER ANSWER IF ASKED:** Training data updated model weights. Validation data
compared HOG and CNN experiments, controlled early stopping, and selected the
0.60 threshold. Test data did none of those jobs and was used only for the final
held-out result.

### 9. What other leakage controls did you use?

**SHORT ANSWER:** Metadata was excluded from features, augmentation stayed in
training, and the frozen split and manifest hashes were checked.

**LONGER ANSWER IF ASKED:** The models use image pixels or HOG values only.
Family names, file paths, split labels, phrases, source fonts, and effect
metadata are not features. Validation and test preprocessing is deterministic,
and tests assert zero overlap and no test access during selection.

### 10. Where and when was augmentation used?

**SHORT ANSWER:** Mild random affine and sharpness augmentation was used only on
training images.

**LONGER ANSWER IF ASKED:** Validation, test, and app inference use deterministic
preprocessing with no random augmentation. This keeps comparison and evaluation
stable and prevents validation/test behavior from influencing fitting.

### 11. Why grayscale?

**SHORT ANSWER:** Font category mainly depends on stroke and shape, not colour.

**LONGER ANSWER IF ASKED:** Grayscale reduces the input to one channel, making
the model smaller and cheaper while preserving letter structure and contrast.
It also discourages the model from depending on colours that do not define the
font category.

### 12. Why resize to 112×48?

**SHORT ANSWER:** It preserves a short text-line shape while keeping the CNN
small enough for CPU and Colab use.

**LONGER ANSWER IF ASKED:** The source images are 224×96, so 112×48 is the same
aspect ratio at half the width and height. The size was part of validation
experiments and was frozen before the final test.

### 13. What does the majority baseline prove?

**SHORT ANSWER:** It gives a trivial sanity-check result that useful models must
beat.

**LONGER ANSWER IF ASKED:** The balanced five-class validation set gives the
always-one-class rule 20% accuracy and 0.0667 macro F1. HOG and the CNN clearly
outperformed it.

### 14. What is HOG?

**SHORT ANSWER:** HOG summarises local edge directions and stroke shapes in an
image.

**LONGER ANSWER IF ASKED:** It divides a grayscale image into cells, builds
orientation histograms, and normalises blocks. That makes it a sensible
classical feature for font strokes, but it cannot learn features directly from
the task as a CNN can.

### 15. What does multinomial Logistic Regression do?

**SHORT ANSWER:** It maps the HOG feature vector to probabilities for all five
categories.

**LONGER ANSWER IF ASKED:** It learns linear decision boundaries between the
five classes. It is simple, fast, and interpretable enough to provide a useful
baseline against the CNN.

### 16. What is a CNN?

**SHORT ANSWER:** A CNN learns visual filters that combine edges and strokes
into higher-level letter-shape features.

**LONGER ANSWER IF ASKED:** Convolution layers scan small regions with shared
filters. Pooling reduces spatial size, and deeper layers combine simple
patterns. The final linear layer produces five class scores, which softmax
converts to probabilities.

### 17. Why use the progression majority → HOG → CNN?

**SHORT ANSWER:** It shows improvement from a trivial rule to classical vision
and then learned visual features.

**LONGER ANSWER IF ASKED:** Each stage answers a useful question. Majority checks
the metric pipeline, HOG checks whether fixed stroke features are enough, and
the CNN checks whether task-learned spatial features add value. Their validation
macro F1 values were 0.0667, 0.6933, and 0.8331.

### 18. How was the final model selected without using the test set?

**SHORT ANSWER:** The Reference small CNN had the highest validation macro F1.

**LONGER ANSWER IF ASKED:** Training used only training families; validation
controlled experiment comparison and early stopping. The selected epoch,
checkpoint, preprocessing, class order, and threshold were recorded before any
test image was loaded.

### 19. How did you control overfitting?

**SHORT ANSWER:** I used unseen validation families, a small network, dropout,
training-only mild augmentation, and early stopping.

**LONGER ANSWER IF ASKED:** Family-level validation is stricter than seeing new
images of known families. Learning curves and validation macro F1 were tracked,
and the best checkpoint was saved at epoch 14 instead of simply using the last
epoch.

### 20. What does 86.67% accuracy mean?

**SHORT ANSWER:** The final CNN correctly classified 520 of 600 held-out images.

**LONGER ANSWER IF ASKED:** It describes this specific balanced synthetic test
set from 15 unseen families. It does not mean 86.67% on every screenshot, photo,
font family, language, or future real-world distribution.

### 21. What are precision and recall?

**SHORT ANSWER:** Precision asks whether predictions of a class are correct;
recall asks how many true examples of that class were found.

**LONGER ANSWER IF ASKED:** A class can have high precision but lower recall.
For example, sans serif precision was 0.919 but recall was 0.658, meaning the
model was usually right when it predicted sans serif but missed many true sans
serif images.

### 22. Why is macro F1 the primary metric?

**SHORT ANSWER:** It balances precision and recall and gives every class equal
importance.

**LONGER ANSWER IF ASKED:** Macro F1 calculates one F1 per category and averages
them equally. Even though this dataset is balanced, it still reveals weak class
behavior better than accuracy alone. The final value was 0.8653.

### 23. Was class imbalance a problem?

**SHORT ANSWER:** No; the saved dataset and every split were exactly balanced by
category.

**LONGER ANSWER IF ASKED:** There are 720 images and 18 families per category in
the full dataset. Train, validation, and test also have equal category counts.
Macro F1 was still used because class difficulty can differ even when counts are
equal.

### 24. Why is the uncertainty threshold 0.60?

**SHORT ANSWER:** Validation analysis chose the lowest candidate with at least
90% accepted-prediction accuracy and at least 50% coverage.

**LONGER ANSWER IF ASKED:** At 0.60, the validation data met that rule. The
threshold was frozen before test evaluation. It changes only accepted versus
uncertain status, not the probabilities or predicted class.

### 25. Does an accepted or high-confidence prediction guarantee correctness?

**SHORT ANSWER:** No. Confidence is the model's probability estimate, not a
guarantee.

**LONGER ANSWER IF ASKED:** On the final test, accepted predictions were 92.51%
accurate, so some were still wrong. The report contains 34 confident mistakes.
Users should treat the output as a first guess and inspect difficult inputs.

### 26. Why can the model be confidently wrong?

**SHORT ANSWER:** A new font can contain shapes that strongly resemble the wrong
category based on patterns learned from training families.

**LONGER ANSWER IF ASKED:** Softmax can be sharp even when an example is outside
the training distribution. Synthetic training cannot cover every real design.
The strongest saved confident mistake was a serif image predicted as monospace
with 94.81% confidence.

### 27. What was the weakest class and a common confusion?

**SHORT ANSWER:** Sans serif was weakest; sans serif → monospace occurred 24
times.

**LONGER ANSWER IF ASKED:** Sans serif recall was 65.83%. Clean geometric sans
serif letters can have regular widths and strokes that resemble monospace.
Serif → monospace was also common, with 20 errors.

### 28. How do you know the test result is trustworthy?

**SHORT ANSWER:** The model contract was frozen first, test access was recorded,
hashes stayed unchanged, and all 600 test images were evaluated once.

**LONGER ANSWER IF ASKED:** Saved freeze and receipt files show that test data was
not used for training, validation, early stopping, model selection, or threshold
selection. The final evaluation checked 120 images and three unseen families per
category, with zero overlap and no post-test tuning.

### 29. What did MLflow and the Data Gate add?

**SHORT ANSWER:** MLflow made experiments comparable; the Data Gate checked that
the data and boundaries were trustworthy enough for modelling.

**LONGER ANSWER IF ASKED:** MLflow recorded real parameter changes, metrics,
runtimes, sizes, and artifacts. The unchanged teacher Data Gate self-check gave
19 PASS, 1 WARN, and 1 MANUAL, so it suggested YELLOW. Full manual CV evidence
then passed and supported the final human GREEN decision.

### 30. What are the main real-world and AI-assistance limitations?

**SHORT ANSWER:** Synthetic images do not represent every real design, and AI
helped build the project, but the saved results came from executed artifacts.

**LONGER ANSWER IF ASKED:** Real screenshots, photos, unfamiliar fonts, non-Latin
scripts, mixed fonts, and heavy edits can fail. ChatGPT/Codex assisted with
planning, implementation, debugging, testing, documentation, and review. I
reviewed and ran the work, I am responsible for understanding it, and no model
metric was invented to replace an executed result.

## M. AI assistance

“I used ChatGPT/Codex to assist with planning, code scaffolding and
implementation, debugging, tests, documentation, and repository review. The
dataset counts, validation comparisons, and final metrics come from executed
and saved project artifacts, not invented text. I reviewed and ran the work and
remain responsible for understanding, testing, presenting, and defending it.
AI assistance did not replace real experimental evidence or human review.”

## N. Rapid-fire cheat sheet

| Item | Answer |
|---|---|
| Project | FontSense |
| Classes | 5 |
| Families | 90 |
| Images | 3,600 |
| Family split | 60 train / 15 validation / 15 test |
| Final test | 600 images / 15 unseen families |
| Best model | Reference small CNN |
| Accuracy | 86.67% |
| Macro F1 | 0.8653 |
| Input | 112×48 grayscale |
| Threshold | 0.60 |
| Model size | 255,505 bytes |
| Inference | 7.66 ms/image |
| Weakest class | Sans serif |
| Biggest limitation | Synthetic-to-real gap |

- **Why family split?** To stop the same family's shapes leaking into training
  and test data.
- **Why CNN?** It learned better task-specific visual features and had the
  highest validation macro F1.
- **Why macro F1?** It balances precision and recall while weighting all five
  categories equally.
- **Why threshold?** Validation data selected 0.60 to warn on lower-confidence
  guesses without changing the model output.
- **Biggest limitation?** Synthetic rendered text differs from real screenshots,
  photos, and complex designs.

## O. Things not to say

| Wrong statement | Correct wording |
|---|---|
| “The model finds the exact font.” | “The model predicts one of five broad font categories.” |
| “The automatic Data Gate was Green.” | “The automatic self-check suggested YELLOW; manual CV evidence supported the final human GREEN decision.” |
| “86.67% means it works on all images.” | “86.67% is the result on the defined 600-image held-out synthetic test.” |
| “The test set selected the model.” | “Validation macro F1 selected the model; test data was used once afterward.” |
| “Confidence guarantees correctness.” | “Confidence is a model estimate; confident mistakes still exist.” |
| “All 90 families were training data.” | “60 families trained the model, 15 validated it, and 15 were held out for test.” |
| “The dataset contains real screenshots.” | “The assessed dataset contains controlled synthetic renders from Google Fonts.” |
| “FontSense is production ready.” | “FontSense is an educational prototype and broad-category first-guess tool.” |
| “The Windows build is one self-contained EXE.” | “It is a verified one-folder build; its support files must remain beside the EXE.” |
| “The Windows executable is signed.” | “The distribution is unsigned, so SmartScreen may warn.” |

## P. Final 5-minute checklist

- [ ] Existing site/showcase open and key facts checked
- [ ] Local Gradio running with **Final CNN** selected
- [ ] Windows `FontSense.exe` double-click and browser opening checked
- [ ] One valid cropped Latin-text demo image ready
- [ ] Optional blank, invalid, or uncertain example ready
- [ ] GitHub README open
- [ ] Colab notebook open as backup
- [ ] Internet connection checked
- [ ] Notifications muted
- [ ] Laptop connected to power
- [ ] Important numbers reviewed: 90 families, 3,600 images, 60/15/15 split,
      0.8653 macro F1, 86.67% accuracy, 0.60 threshold
