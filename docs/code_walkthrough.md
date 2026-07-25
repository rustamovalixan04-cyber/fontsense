# Code walkthrough for the defense

## `font_audit.py`

Checks whether candidate fonts exist and can render Latin text. The local-system path is only for a proof run.

## `google_fonts.py`

Reads the curated catalog, downloads each family’s official `METADATA.pb`, verifies its category and Latin subset, selects a usable TTF/OTF file, records its license and source URL, and creates the final font manifest.

## `split.py`

Assigns whole font families to train, validation, or test. `assert_no_family_leakage` stops the pipeline if one family appears in more than one split.

## `generate_dataset.py`

Renders short phrases into 224×96 images with controlled changes such as size, position, background, blur, noise, compression, and small rotation. It writes `families.csv` and `manifest.csv`.

## `features.py`

Converts images to grayscale and extracts HOG features. HOG measures local edge and stroke-direction patterns.

## `train_hog.py`

Trains a majority-class sanity baseline and several HOG + Logistic Regression configurations. It chooses the model using validation macro F1, refits it on train plus validation, and saves the pipeline. The test set is not used for selection.

## `cnn_model.py` and `train_cnn.py`

Define and train a compact convolutional neural network in PyTorch. The training loop records loss, accuracy, macro F1, early stopping, and experiment parameters. The best validation checkpoint is saved.

## `evaluate.py`

Loads a saved model and evaluates it only on test families. It creates metrics JSON, a per-class report, a confusion matrix, predictions, and error examples.

## `inference.py`

Provides one reusable prediction interface for HOG and CNN artifacts. It validates image size, returns probabilities, measures inference time, and gives a warning below the confidence threshold.

## `app.py`

Creates the Gradio upload interface. This is only the interface; the actual ML models are trained inside the project.

## What you must be able to explain

- Why the family-level split is essential.
- Why macro F1 is primary.
- What the majority baseline proves.
- What HOG represents.
- How the CNN differs from HOG.
- Why validation selects the model and test data is used once at the end.
- Why low confidence requires human review.
