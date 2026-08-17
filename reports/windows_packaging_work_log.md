# Windows packaging work log

## Scope control

- No dataset was regenerated.
- No model was retrained or tuned.
- The frozen split, final evaluation, confidence threshold, and assessed checkpoint were not changed.
- The untracked `docs/feature_spec.md` file was confirmed to be the obsolete EXTC1 Windows-launch draft, recorded with SHA-256 `a933a1189d9a699a0d16e5a2fbf86788efd25d1fbd2cc88dd987df02de7d450a`, and removed before packaging. It was never part of Git history.

## Frozen source model

- Checkpoint: `artifacts/cnn/cnn_model.pt`
- SHA-256: `c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f`
- Class order: `display`, `handwriting`, `monospace`, `sans_serif`, `serif`
- Input: 112 x 48 grayscale, normalized with mean 0.5 and standard deviation 0.5
- Confidence threshold: 0.60

## Build environment

- Platform: Windows 11 Pro x64, version 10.0.26200
- Python: 3.12.13
- PyInstaller: 6.22.1
- PyTorch: 2.13.0+cpu
- torchvision: 0.28.0+cpu
- Gradio: 6.20.0
- Pillow: 12.3.0
- NumPy: 2.5.1
- CUDA available: no

## Build and fixes

The release was built with `scripts/build_windows.ps1` as a console-enabled PyInstaller one-folder application. Clean-build smoke testing exposed three missing Gradio runtime data files in sequence: `safehttpx/version.txt`, `groovy/version.txt`, and `gradio/blocks_events.py`. The specification was updated only to bundle these required dependency resources, then the package was rebuilt cleanly.

## Verification result

- Packaged checkpoint hash: matches the frozen source checkpoint.
- Packaged self-test: PASS.
- Source/package comparison: PASS for five fixed images.
- Maximum probability difference: 0.0.
- Predeclared tolerance: 0.00001.
- Packaged Gradio HTTP smoke test: PASS.
- Packaged PNG/JPEG, repeated prediction, Reset, and invalid-input checks: PASS.
- Outside-repository portable-folder self-test, launch, and prediction: PASS.
- Normal launch after shutdown: PASS twice.
- Student browser and live-prediction visual confirmation: PASS.
- Full project test suite: 79 passed.
- Manual Data Gate verification: GREEN.
- Detailed machine-readable evidence: `reports/windows_package_equivalence.json`.

## Release artifact

- Local archive: `FontSense_Windows_x64_FINAL.zip`
- Archive structure: complete `FontSense/` one-folder distribution
- Executable size: 65,501,032 bytes (62.47 MiB)
- Executable SHA-256: `9723c3f591f187575a0ff03c55f2b493b85f40c4c1e3a1ca1266238613c61eb4`
- Distribution size: 842,016,029 bytes (803.01 MiB)
- Archive size: 317,811,916 bytes (303.09 MiB)
- Archive SHA-256: `d4c2ec69e5c182f696cbfa7f7847bc6f16ab3a95ec104de2737b0d860f77c619`
- Git policy: the generated archive and unpacked build remain ignored; reproducible build instructions and verification evidence are committed.
