# FontSense Windows standalone build

## Purpose

This build lets an evaluator run the frozen FontSense CNN on Windows x64 by
double-clicking `FontSense.exe`. The evaluator does not need Python, the project
virtual environment, Google Colab, or a terminal command. It is a one-folder
distribution, so the complete `FontSense` folder must be kept together.

## Verified build environment

| Component | Verified version |
|---|---|
| Operating system | Windows 11 Pro x64, version 10.0.26200 |
| Python | 3.12.13 |
| PyInstaller | 6.22.1 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| Gradio | 6.20.0 |
| Pillow | 12.3.0 |
| NumPy | 2.5.1 |

CUDA is not required. The package contains the CPU inference runtime.

## Rebuild

Create and populate the normal project `.venv`, then install the pinned build
tool:

```powershell
.\.venv\Scripts\python.exe -m pip install -r packaging\windows\requirements-build.txt
```

Build the distribution from the repository root:

```powershell
.\scripts\build_windows.ps1
```

The reproducible PyInstaller configuration is
`packaging/windows/FontSense.spec`. The output is:

```text
dist/FontSense/FontSense.exe
```

## Launch and diagnostics

Normal evaluator launch:

```text
dist\FontSense\FontSense.exe
```

The console stays open, the server binds only to `127.0.0.1`, and Gradio asks
the default browser to open the local interface. If the browser does not open,
use the local URL printed in the console. Close the console or press Ctrl+C to
stop FontSense.

Developer diagnostics:

```powershell
.\dist\FontSense\FontSense.exe --self-test
.\dist\FontSense\FontSense.exe --predict-json data\sample\serif__DejaVu_Serif_0000.png
.\dist\FontSense\FontSense.exe --no-browser --port 7867
```

## Frozen runtime contract

- Checkpoint: `artifacts/cnn/cnn_model.pt`
- SHA-256: `c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`
- Classes: `display`, `handwriting`, `monospace`, `sans_serif`, `serif`
- Preprocessing: grayscale, 112 × 48, mean 0.5, standard deviation 0.5
- Confidence threshold: 0.60

The package includes the checkpoint and
`reports/final_evaluation/pre_test_freeze.json` with their relative structure
preserved under the PyInstaller runtime folder.

## Verification result

Run the complete package verifier:

```powershell
.\.venv\Scripts\python.exe scripts\verify_windows_package.py
```

The final clean build passed:

- package content audit;
- bundled checkpoint hash check;
- executable `--self-test`;
- five fixed source/package comparisons;
- predeclared maximum probability tolerance of `1e-5`;
- global observed maximum probability difference of `0.0`;
- packaged local Gradio server startup;
- HTTP 200 response from the local interface.
- PNG and JPEG prediction through the packaged Gradio controls;
- Reset plus blank, too-small, missing, and corrupted-input handling;
- two normal launch/stop cycles;
- an outside-repository portable-folder self-test and live prediction;
- student visual confirmation of browser opening and a live prediction.

The evidence is saved in
`reports/windows_package_equivalence.json`. These checks verify packaging
equivalence only; they are not a new model evaluation and do not use the formal
held-out test dataset.

## Size and release archive

- One-folder distribution: 842,016,029 bytes (803.01 MiB)
- Distribution files: 13,663
- Executable size: 65,501,032 bytes (62.47 MiB)
- Executable SHA-256: `9723c3f591f187575a0ff03c55f2b493b85f40c4c1e3a1ca1266238613c61eb4`
- Local release ZIP: `FontSense_Windows_x64_FINAL.zip`
- ZIP size: 317,811,916 bytes (303.09 MiB)
- ZIP SHA-256: `d4c2ec69e5c182f696cbfa7f7847bc6f16ab3a95ec104de2737b0d860f77c619`

The ZIP contains the complete `FontSense/` folder, not only the executable.
`dist/`, `build/`, and the ZIP are ignored local release artifacts and are not
committed to Git.

## Intentionally not packaged

The standalone runtime does not include:

- the optional HOG comparison model or its joblib artifacts;
- generated train, validation, or test images;
- full dataset manifests or the frozen family split;
- MLflow runs;
- notebooks;
- training scripts as explicit runtime resources;
- project reports other than the required freeze record;
- credentials or secrets.

## Windows warning

The executable is not digitally signed. Windows SmartScreen may show an
unknown-publisher warning when the folder is copied to another computer. This
is a distribution limitation, not a model or checkpoint failure.
