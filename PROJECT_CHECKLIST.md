# FontSense project checklist

## Approval and scope
- [ ] Send the edited brief to mentors
- [ ] Record mentor approval or required revisions
- [ ] Freeze the five-class scope

## Repository
- [x] GitHub-ready structure created
- [x] README, requirements, .gitignore, tests, and documentation created
- [ ] Create GitHub repository and push initial commit
- [ ] Add repository URL to notebooks and LMS template

## Data
- [x] Google Fonts audit/downloader implemented
- [x] System-font proof audit implemented
- [x] Family-level splitting implemented and tested
- [x] Synthetic dataset generator implemented
- [ ] Run final Google Fonts audit
- [ ] Confirm at least three independent families per category; preferably 20–35
- [ ] Review license/source manifest
- [ ] Run final dataset generation

## Modeling
- [x] Majority baseline implemented
- [x] HOG + Logistic Regression experiments implemented
- [x] MLflow hooks implemented
- [x] Compact PyTorch CNN implemented
- [ ] Run final HOG experiments
- [ ] Run final CNN experiments
- [ ] Select final model using validation only

## Evaluation
- [x] Test evaluation script implemented
- [x] Classification report, confusion matrix, predictions, errors implemented
- [ ] Run final untouched test evaluation
- [ ] Write honest error analysis
- [ ] Choose uncertainty threshold using validation data

## Demo and delivery
- [x] Gradio application implemented
- [x] Colab demo notebook created
- [ ] Test normal, low-confidence, and invalid inputs
- [ ] Run clean-Colab reproduction test
- [ ] Ask another person to follow README without verbal help

## Submission and defense
- [x] Final report structure created
- [x] 10-minute defense structure created
- [x] Likely Q&A created
- [ ] Insert final results and screenshots
- [ ] Complete LMS submission template
- [ ] Grant mentor repository access
- [ ] Freeze repository before deadline
- [ ] Do not commit assessed work after submission
