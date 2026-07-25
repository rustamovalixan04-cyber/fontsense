# FontSense Capstone Project Instructions

## Student and project context

I am Rustamov Alixan, a beginner AI/ML student. I am still learning Python, machine learning, Git, experiment tracking, and software development.

This repository is my final AI/ML Fundamentals capstone project: **FontSense**, a computer-vision system that classifies a cropped Latin-script text image into one of five categories:

- serif
- sans serif
- display
- handwriting
- monospace

The system should return the predicted category, probabilities for all five classes, and a low-confidence warning when the input is unclear.

I must understand, test, explain, and defend every important part of the finished project.

## How to work with me

1. Inspect the repository and relevant files before changing anything.
2. Explain the next important step before making changes.
3. Use simple language suitable for a beginner.
4. Work in small, clear task blocks instead of rebuilding the whole project at once.
5. Do not redesign the project or make large structural changes without asking first.
6. Explain every new file, important function, library, model, metric, and technical decision.
7. Prefer simple, working, reproducible solutions over unnecessary complexity.
8. Keep the project appropriate for an AI/ML Fundamentals course and runnable in Google Colab.
9. Preserve useful existing work unless there is a clear reason to replace it.
10. Never fabricate results, metrics, screenshots, experiments, dataset statistics, or successful test output.

## Required workflow for every task

For every meaningful task, follow this exact sequence:

### 1. PLAN
Before editing:
- explain what the task is;
- explain why it is needed;
- list the files likely to be changed;
- mention any important risk or decision.

### 2. DO
Make only the changes needed for the current task. Do not mix unrelated work into the same task.

### 3. CHECK
Run the relevant verification, such as:
- automated tests;
- notebook cells;
- import checks;
- linting or formatting;
- dataset validation;
- training smoke tests;
- inference tests;
- Git status.

Do not claim that something works unless it was actually checked.

### 4. REVIEW
After checking:
- summarize exactly what changed;
- show which files were changed;
- explain the result in simple language;
- mention any remaining error, warning, limitation, or risk.

### 5. COMMIT AND PUSH
After every completed task:
- commit the work automatically only when its checks pass;
- keep each commit focused on one coherent task;
- never commit broken, incomplete, or unverified work;
- never make empty commits;
- never backdate commits or fabricate development history;
- do not amend or rewrite earlier commits unless I explicitly ask and understand the reason;
- push the new commit to the connected GitHub repository automatically after the commit succeeds;
- push only to the current normal branch;
- never force-push;
- if commit or push fails, stop and show the real error instead of pretending it worked.

Before committing, show the proposed commit message and the files to be included. Then commit and push automatically unless I explicitly tell you not to.

## Commit message rules

Commit messages must sound natural, specific, and written by a normal student developer.

Describe what changed, not who or what produced it.

Good examples:
- `Add Google Fonts metadata audit`
- `Create family-level dataset split`
- `Fix image rendering for variable fonts`
- `Add class balance charts to EDA`
- `Train HOG logistic regression baseline`
- `Record baseline runs with MLflow`
- `Add CNN validation metrics`
- `Handle low-confidence predictions`
- `Document final model limitations`

Avoid robotic, vague, or agent-style messages:
- `AI agent completed task`
- `Implement comprehensive enhancements`
- `Update multiple files`
- `Automated repository improvements`
- `Perform requested changes`
- `Misc fixes`
- `Final update`

Do not create misleading commit messages to hide AI assistance. AI assistance must be acknowledged honestly in the project documentation where required. Commit messages should simply describe the real code or documentation change.

## Git safety rules

1. Check `git status` before and after each task.
2. Do not delete or overwrite unrelated user work.
3. Do not create a new branch unless I ask.
4. Do not force-push.
5. Do not reset, rebase, amend, squash, or rewrite history without explicit permission.
6. Do not commit secrets, tokens, passwords, downloaded font archives, generated datasets, cache folders, virtual environments, or unnecessary large artifacts.
7. Respect `.gitignore`.
8. Keep the working tree clean after a successful commit.
9. After each successful task commit, push it to the connected GitHub repository automatically unless I explicitly tell you not to.
10. If the repository is not connected, authentication is missing, or the push is rejected, stop and explain the exact problem.

## Project-specific technical rules

### Dataset
1. Use legally usable open-source fonts, mainly from Google Fonts.
2. Record font family, category, source, license, file path, and validation status.
3. Focus on Latin-script words and short text lines.
4. Keep generated data reproducible by using saved configuration and random seeds.
5. Inspect generated images instead of trusting the generator automatically.
6. Keep the final assessed dataset separate from any small proof-of-concept dataset.

### Leakage prevention

The most important rule is:

**No font family may appear in more than one of training, validation, and test.**

Split by font family before image generation.

Never use these as model inputs:
- font family name;
- source folder;
- file path;
- category metadata;
- split label;
- any value that directly reveals the answer.

Keep the final test families untouched until the final model has been selected.

### Required models
The core comparison should include:
1. majority-class baseline;
2. HOG features with multinomial Logistic Regression;
3. a small CNN.

Transfer learning is optional and must not replace the required approaches unless approved.

### Evaluation
Use validation data for model selection. Use the test set only for final evaluation.

Primary metric:
- macro F1

Also report:
- accuracy;
- per-class precision, recall, and F1;
- confusion matrix;
- inference time;
- model size;
- low-confidence behavior;
- meaningful error examples.

Report weak or disappointing results honestly and explain them.

### MLflow
Record meaningful experiments with clear names and parameters.

Do not create fake or duplicate runs merely to increase the experiment count.

Each run should represent a real change, such as:
- HOG settings;
- Logistic Regression regularization;
- image size;
- CNN architecture;
- learning rate;
- augmentation;
- number of epochs.

### Demo
The final demo should:
- accept a PNG or JPEG text crop;
- apply the same preprocessing used during training;
- show one predicted category;
- show probabilities for all five categories;
- warn when confidence is low;
- reject invalid or unreadable inputs without crashing.

### Reproducibility
The project should run from a fresh environment using documented instructions.

Do not rely on:
- hidden local files;
- manually created notebook state;
- personal absolute file paths;
- uncommitted code;
- unavailable private data.

## Course route

Follow this order unless I explicitly approve a change:

1. Scope and project-brief approval.
2. Repository setup and environment verification.
3. Font source audit and license documentation.
4. Family-level split and dataset generation.
5. Data audit and EDA.
6. Majority and HOG baselines.
7. MLflow experiments and model selection.
8. CNN training and validation.
9. Final untouched test evaluation.
10. Error analysis and limitations.
11. Inference demo.
12. Reproducibility and clean-run testing.
13. README and final report.
14. Submission and defense preparation.

Do not skip ahead to final training before the setup, data checks, and leakage-safe split are working.

## Communication style

After every task, report:

1. **What was done**
2. **Why it was done**
3. **Files changed**
4. **Checks run**
5. **Result**
6. **Remaining risk**
7. **Commit message used**
8. **Push result**

Keep explanations short and simple unless I ask for more detail.

When code fails, show the real error and explain it. Do not hide errors or claim success prematurely.

## Academic integrity

AI tools may assist with planning, coding, debugging, testing, and documentation, but I remain responsible for the final work.

Do not help create fake history, fake authorship, fake results, or misleading evidence.

Make sure I can explain:
- the project problem;
- how the data was created;
- why the split is leakage-safe;
- what HOG does;
- what Logistic Regression does;
- how the CNN works;
- why macro F1 is used;
- how the final model was selected;
- where the model fails;
- how AI assistance was used.

If I request unnecessary complexity or something I will not be able to defend, warn me and recommend a simpler option.

## Starting instruction

When first reading this file:

1. inspect the repository;
2. check the current Git branch and status;
3. identify the current project stage;
4. inspect existing tests and setup files;
5. explain the next smallest useful task;
6. do not begin that task until the plan is clear.
