# FontSense — Typeface Category Recognition

**Student:** Rustamov Alixan  
**Field:** Graphic Design / Computer Vision  
**Task:** Multiclass image classification

FontSense accepts a cropped image containing Latin-script text and predicts one of five broad typeface categories:

- `serif`
- `sans_serif`
- `display`
- `handwriting`
- `monospace`

The output includes the predicted category, class probabilities, and an uncertainty warning for low-confidence inputs.

> Core project rule: font families are split before image generation. No family may appear in more than one of train, validation, or test. This prevents the model from memorizing a typeface family.

## Repository status

This package contains the complete project architecture, Google Fonts audit/downloader, leakage-safe dataset generator, EDA workflow, HOG + Logistic Regression baseline, PyTorch CNN, MLflow integration, final evaluation, Gradio demo, tests, documentation, and defense material.

A small **proof-of-concept artifact** is included so the inference flow can be tested immediately. The final assessed metrics must be generated from the approved Google Fonts dataset by running the pipeline in Colab.

## Quick start — run the included proof model

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL shown in the terminal and upload a cropped text image.

## Recommended Colab flow

After creating the GitHub repository, open each notebook in order:

1. `notebooks/01_font_audit.ipynb`
2. `notebooks/02_dataset_generation.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_baseline_mlflow.ipynb`
5. `notebooks/05_cnn_mlflow.ipynb`
6. `notebooks/06_final_evaluation.ipynb`
7. `demo.ipynb`

Every notebook begins with a setup section. Replace the placeholder repository URL after uploading this project to GitHub.

## One-command local pipeline

### Small proof pipeline using locally installed fonts

```bash
python scripts/run_pipeline.py --source system --images-per-family 20 --cnn-epochs 8
```

### Final Google Fonts pipeline

```bash
python scripts/download_google_fonts.py --max-per-category 35
python -m fontsense.generate_full_dataset --verify-reproducibility
python -m fontsense.train_hog --manifest reports/dataset/full_manifest.csv
python -m fontsense.train_cnn --manifest reports/dataset/full_manifest.csv --epochs 15
python -m fontsense.evaluate --manifest reports/dataset/full_manifest.csv --model hog
python -m fontsense.evaluate --manifest reports/dataset/full_manifest.csv --model cnn
```

When running modules locally, either install the package or set the source path:

```bash
pip install -e .
```

## Project structure

```text
FontSense_Capstone_Project/
├── README.md
├── app.py
├── demo.ipynb
├── requirements.txt
├── pyproject.toml
├── config/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/fontsense/
├── tests/
├── artifacts/
├── reports/
└── submission/
```

## Models

### 1. Majority-class baseline
A sanity check that always predicts the most frequent class.

### 2. HOG + multinomial Logistic Regression
HOG summarizes edge and stroke direction patterns. Logistic Regression produces probabilities for all five classes and is fast, explainable, and reproducible.

### 3. Small CNN
A compact PyTorch convolutional network learns visual stroke patterns directly from grayscale images.

### Optional
Transfer learning with MobileNetV2 or EfficientNetB0 may be added only after the required pipeline is complete.

## Evaluation

Primary metric: **macro F1**. Supporting outputs:

- accuracy
- per-class precision, recall, and F1
- confusion matrix
- inference time
- model size
- low-confidence and invalid-input behavior
- error examples grouped by true and predicted class

The test set is used only for final evaluation.

## Academic integrity

AI tools may assist with planning, code, debugging, and documentation, but Rustamov Alixan remains responsible for testing, understanding, explaining, and defending every submitted component. Meaningful AI assistance should be acknowledged in the final report and defense.

## Important limitation

The model predicts broad typeface categories, not exact font families. It is trained mainly on generated Latin-script text crops and may be unreliable on logos, photographs, curved text, extreme effects, mixed fonts, or non-Latin scripts.
