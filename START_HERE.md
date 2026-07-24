# Start here — FontSense

## What is already finished

- Your edited project brief is included in `docs/`.
- The repository structure is ready for GitHub.
- Google Fonts downloading and license/category auditing are implemented.
- Dataset generation, family-level splitting, EDA, HOG baseline, CNN, MLflow hooks, evaluation, error analysis, Gradio demo, tests, final-report outline, and defense notes are implemented.
- A small system-font proof run was completed to confirm that the whole code path works. Its results are stored under `artifacts/proof/` and `reports/proof/`; they are **not final assessed results**.

## What you do now

1. Send `docs/FontSense_Project_Brief_Rustamov_Alixan.docx` to your mentor before the brief deadline.
2. Create an empty GitHub repository named `fontsense-capstone`.
3. Upload the contents of this folder and make the first commit.
4. Replace `PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE` in the notebooks after the repository exists.
5. Once the idea is approved, run the notebooks in order, starting with `01_font_audit.ipynb`.

## Do not present the proof metrics

The proof run uses locally installed fonts with approximate labels. It exists only to verify that the software works. Your final tables and screenshots must come from the Google Fonts run.

## Minimum final run

Aim for approximately 15–25 usable independent font families per category and 30–40 rendered images per family. Reduce the amount only when Colab time or storage becomes a real blocker, and document the reason.

## Before submission

- Run the final demo from a fresh Colab runtime.
- Confirm no family overlap between splits.
- Confirm mentors can access the repository.
- Fill in `submission/submission_details.md`.
- Freeze the repository before submitting to LMS.
- Do not add assessed work after the deadline.
