# EXTC0 no-partner Peer QA review

## Review metadata

- Reviewer: Rustamov Alixan
- Owner: N/A — teacher-provided no-partner demo route
- Round: No-partner demo route
- Date: 2026-08-10
- Reviewed project: Student Support Risk — Peer QA Demo
- Source: Teacher-provided `EXTC0_Demo_Peer_QA_Repository.zip`; no repository URL was supplied
- Integrity: The demo was extracted outside FontSense, reviewed read-only, and not edited, committed, or submitted for an owner.

## Project understanding

- Problem and user: Estimate a simple support priority for a student-support mentor from attendance and missing assignments (`README.md`, lines 5–10).
- Input and output: A comma-separated `attendance_rate,missing_assignments` string produces `LOW`, `MEDIUM`, or `HIGH` plus a score (`README.md`, lines 9–10; `app.py`, lines 3–4 and 26–28).
- Selected logic and reported result: The demo uses a transparent hand-written scoring rule (`README.md`, line 11; `app.py`, lines 10–22). `docs/evaluation.md` reports synthetic placeholder values of 0.71 baseline accuracy and 0.78 final macro F1. They are not real validation evidence.
- Main limitation: There is no real dataset, trained model, exact split procedure, or real-world validation. Input validation is also absent (`data/README.md`; `docs/evaluation.md`; `docs/limitations.md`).

## Blind repository navigation

| Item | Status | Evidence or required question |
|---|---|---|
| Problem statement | Found | `README.md`, lines 5–11 |
| Intended user | Found | `README.md`, line 8 |
| Input and output | Found | `README.md`, lines 9–10; `app.py`, lines 3–4 and 26–28 |
| Data source and structure | Partial | `data/README.md` honestly says there is no real dataset, but no source, schema, split, or leakage evidence exists. |
| Pipeline and preprocessing | Partial | Parsing and scoring are visible in `app.py`, lines 10–22; a documented reusable pipeline is not present. |
| Results and evaluation | Partial | `docs/evaluation.md` contains placeholder metrics but no calculation or exact held-out procedure. |
| Demo entry point | Partial | `README.md` points to missing `demo.py`; `app.py` is the working entry point. |
| Setup and run instructions | Partial | No third-party packages are needed, but the filename is wrong and the Python version is not documented. |
| Limitations | Found | `README.md`, lines 29–30; `docs/limitations.md` |

No hidden absolute paths, credentials, external services, or model artifacts were found. `requirements.txt` contains no third-party dependency. The project root and `app.py` reveal the working command, but a literal README-only run fails.

## Executed checks

Environment: Windows, Python 3.14.6, demo root as the working directory, and `PYTHONDONTWRITEBYTECODE=1`.

| Case | Command and input | Expected | Actual evidence | Result |
|---|---|---|---|---|
| README command | `python demo.py "0.82,2"` | Demo produces a prediction. | Exit 2: Python cannot open `demo.py` because the file does not exist. | Fail |
| Normal | `python app.py "0.82,2"` | A valid support-priority result. | Exit 0: `support_priority=LOW score=0.186` | Pass |
| Invalid | `python app.py ""` | A controlled validation message. | Exit 1: `ValueError: not enough values to unpack (expected 2, got 1)` at `app.py:11`. | Fail |
| Edge | `python app.py "1.20,-1"` | Reject impossible attendance and assignment values. | Exit 0: `support_priority=LOW score=-0.170`; impossible values are accepted. | Fail |

## Evaluation evidence check

| Evaluation item | Status | Evidence |
|---|---|---|
| Baseline | Partial | `docs/evaluation.md` states accuracy 0.71 but does not define or reproduce it. |
| Meaningful model/configuration comparison | Missing | No compared configuration or experiment evidence is present. |
| Metric choice and justification | Partial | Accuracy and macro F1 are listed, but their choice is not justified and the values are not derived. |
| Unseen-data evaluation | Partial | “Held out” is stated, but the split procedure and data do not exist. |
| Error analysis or failure cases | Partial | Limitations describe invalid behavior, but no empirical or class-level analysis is present. |
| Final-model justification | Partial | Explainability is stated as the reason, without comparative evidence. |
| Honest limitations and conclusion | Present | `docs/limitations.md` and `README.md` clearly describe the demo as synthetic and non-validating. |

## Defense questions

1. What input bounds and validation behavior are intended, and what should happen for empty or impossible values?
2. What exactly does “held out” mean, and how were 0.71 accuracy and 0.78 macro F1 calculated without a real dataset?
3. Why were the 0.25 and 0.50 thresholds chosen, and what error costs matter to the intended mentor?

Answer evidence and answer quality: **STUDENT ANSWER REQUIRED**. No project owner was present, so answers were not invented or graded.

## Structured feedback

- Works well: The project is unusually clear and honest about its teaching-only scope. The scoring rule is short, transparent, and easy to trace, and the limitations are explicit.
- Blocker: The literal README command names a missing `demo.py`, so a first-time reviewer cannot run the demo as instructed.
- Important issue: Empty input crashes, while impossible values such as attendance 1.20 and missing assignments -1 are accepted and can produce a negative score.
- Nice to have: Document the supported Python version, valid input ranges, expected output, and a few executable tests.
- Likely defense question: What evidence supports the thresholds and synthetic metrics, and how should false-high versus false-low support priorities be handled?

## Owner triage, next action, and networking

- Owner triage: N/A — teacher-provided no-partner demo route. No owner decision or deadline was fabricated.
- Main risk: The documented command does not run, and invalid values are not controlled.
- Controlled next action: **STUDENT CONFIRMATION REQUIRED**. A reasonable owner action would be to correct the one README filename and add input-bound checks, but this review did not edit the demo.
- Verification evidence: Re-run the four commands above and add tests for empty and out-of-range inputs.
- Mentor guidance: **STUDENT CONFIRMATION REQUIRED** — no mentor statement was supplied.
- My project: FontSense.
- I can help with: **STUDENT INPUT REQUIRED**.
- I need help with: **STUDENT INPUT REQUIRED**.
- Peer connection: **STUDENT INPUT REQUIRED** — no peer connection or reviewer was fabricated.

## Data Gate decision

**YELLOW.** The small demo is mostly reviewable, its true scope is honest, and the working `app.py` command produces a result. One repairable demo blocker remains—the README points to a missing file—and input validation plus evaluation evidence are incomplete.

Required next step: the demo owner should fix the documented entry point, define valid input behavior, and distinguish placeholder evaluation claims from reproducible evidence before treating the project as submission-ready.

## Evidence verification

- The original ZIP, teacher documents, and pairing-message SHA-256 hashes matched their pre-review values after the review.
- Every extracted demo file matched its pre-review SHA-256 and remained read-only.
- The filled DOCX retained one section and 21 tables, preserved all section properties and table grids, contained no blank underline fields, and changed only `word/document.xml` inside the copied template package.
- Visual DOCX rendering was attempted but was unavailable because LibreOffice is not installed on this computer; structural validation passed.
- FontSense’s full automated suite passed: 69 tests, with 12 existing NumPy/joblib deprecation warnings.
