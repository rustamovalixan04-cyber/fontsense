# V2 training and product reliability pass

The improvement work uses the existing 20,000-image V2 dataset. There are 200
font families, with 40 families and 4,000 images in each category. Only 14,000
training images are used for fitting. The 3,000 validation images compare models;
the remaining 3,000 test images stay reserved. The complete data audit is
`reports/v2/data/dataset_validation_summary.json`.

Increasing the number of renders from the same families does not by itself
demonstrate better recognition of unfamiliar fonts. The saved split and manifest
remain frozen so that improvements can be compared honestly.

## Training

Candidate C is repeated because its interrupted attempt saved scores but no
loadable weights. The old history remains in
`reports/v2/cnn/candidate_c/previous_attempts/`; the old interruption record is
retained. Every new best checkpoint is written atomically, and every completed
epoch is saved to CSV. Creating a file named `STOP` in a candidate's report
directory requests a stop after the current epoch and allows the trainer to
finish its reports using the saved best weights.

The additional lower-learning-rate refinement starts from Candidate B's verified
checkpoint. It uses learning rate 0.0001, unchanged width 24 and dropout 0.28,
the same preprocessing and training-only augmentation, at most 12 epochs, and
early stopping after four epochs without sufficient improvement. It writes a
separate checkpoint rather than changing Candidate B. Its justification and
source checkpoint hash are in `config/v2/cnn_refinement.json`.

The completed C restart reached validation macro F1 0.760591 and the refinement
reached 0.788264. Neither beat Candidate B's saved 0.795823, so B remains selected.
The current comparison is `reports/v2/cnn/validation_report.md`. These validation
scores cannot be compared directly to the V1 test score because the image and
family sets differ.

The refinement finished eight epochs and saved its checkpoint and validation
outputs, but a relative-output-path error prevented the summary write. That bug
is fixed and covered by a regression test. The summary was recovered using:

```powershell
.\.venv\Scripts\python.exe scripts/recover_v2_completed_report.py --run-id 777d06080e3a48eaae6b1841055ace81
```

The recovery verifies the saved checkpoint, predictions, history, frozen data
hashes, and FINISHED MLflow run. It does not repeat training or inference on
dataset images. MLflow now has explicit `best_validation_macro_f1` and
`best_validation_accuracy` metrics so best-checkpoint results are distinguished
from the last epoch's learning-curve values.

Commands from the repository root:

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -u -m fontsense.v2_train C --device cpu

$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -u -m fontsense.v2_train D --config config/v2/cnn_refinement.json --artifact-dir artifacts/v2/refinement --report-dir reports/v2/refinement --device cpu

.\.venv\Scripts\python.exe -m fontsense.v2_train select --refinement-report-dir reports/v2/refinement
```

CPU thread counts can slightly change floating-point arithmetic and subsequent
training trajectories. Compare the saved scores for each actual run, and preserve
historical scores as separate evidence.

## User-facing fixes

- Transparent text images are composited on white before conversion to RGB.
- Image dimensions and actual file formats are checked before decoding pixels.
- Streaming uploads obey the 4 MB server limit even without a size header.
- Invalid model probabilities and inconsistent predictions produce clear errors.
- CPU inference runs outside the asynchronous HTTP request loop.
- A small shared runtime loads the frozen model without constructing Gradio for
  every new API process. Local Gradio and Colab still use the same model contract.
- Blank-image checks use a 256-bin histogram instead of large float arrays.
- Resetting or replacing an image cancels its request and prevents stale results.
- Invalid uploads are not automatically submitted a second time.
- Large images are resized in the browser before upload; a preparation timeout
  prevents a decoder from leaving the interface stuck indefinitely.
- Pytest discovers the project's `tests/` directory instead of bundled dependency
  tests inside the copied Windows release.
- MLflow preserves genuine training import errors instead of masking them with a
  context-manager error.

These changes preserve the V1 checkpoint, its assessed final results, grayscale
112-by-48 preprocessing, class order, and 0.60 confidence threshold. A new V2
candidate must complete its own selection and frozen evaluation before being
advertised as a tested replacement for V1.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/web_client.test.cjs
.\.venv\Scripts\python.exe -m uvicorn api.index:app --host 127.0.0.1 --port 8001
# In a second terminal:
.\.venv\Scripts\python.exe scripts/verify_app_smoke.py
```

The HTTP smoke report is `reports/v2/product_verification.json`. Its generated
text images check technical behavior only. Local warmed timing excludes public
network delays and Vercel cold starts. The Windows browser-control tool blocked
visual verification because it could not reliably determine the browser URL;
no successful visual inspection is claimed.

## Remaining accuracy limits

FontSense predicts five broad categories, not an exact font family. It expects a
clear crop of Latin text. Mixed fonts, full pages, unfamiliar designs, photographs,
and inputs without text can produce confident mistakes. A confidence threshold
is a warning rule rather than a guarantee of correctness.

Additional useful data would include independently labelled, permitted real
screenshots and photographs, with font-family grouping preserved. Synthetic
renders alone cannot establish real-world accuracy. Such a collection should
use a new dataset version and must not change the existing frozen split or reuse
held-out families for fitting. The stateless web API processes images in memory.
Gradio can use a temporary upload cache. Neither interface automatically adds
uploaded images to training.

AI assistance: Codex helped inspect code, implement and test fixes, run the
documented experiments, and prepare these records. The student remains
responsible for reviewing and explaining the work and manually checking the demo.
