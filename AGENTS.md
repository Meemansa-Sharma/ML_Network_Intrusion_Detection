
# Project Instructions — ML Network Intrusion Detection

## 1. Project identity

Project title: Machine Learning-Based Network Intrusion Detection and Analytics System.

Purpose: Build a reproducible, educational cybersecurity analytics project that uses machine learning to classify network-flow records as normal or potentially malicious using the public UNSW-NB15 dataset.

This is an offline dataset-based research prototype, not a live intrusion prevention system.

## 2. Work style

- Work incrementally. Explain the plan before making significant changes.
- Inspect existing files before editing them.
- Do not overwrite working files or create duplicate files unnecessarily.
- Implement one logical component at a time.
- Explain important design decisions in beginner-friendly language.
- After each component, run appropriate tests and report their actual results.
- Never claim that code, tests, downloads, or model evaluations succeeded unless they were actually verified.
- Ask before introducing large dependencies, external services, paid APIs, or major scope changes.

## 3. Technical requirements

- Python 3.13 and the project's `.venv` environment.
- pandas and NumPy for data processing.
- scikit-learn for preprocessing, model training, and evaluation.
- Matplotlib and Seaborn for visualisations.
- Jupyter for exploratory analysis and the final notebook.
- joblib for saving and loading trained models.
- pytest for automated tests.

Use the environment interpreter directly on Windows:
`.venv\Scripts\python.exe`

Do not require PowerShell script activation.

## 4. Dataset and data integrity

Use the official UNSW-NB15 dataset:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

- Prefer the official predefined training and testing CSV files.
- Document the exact files used, their source, and their expected schema.
- Do not fabricate observations or silently substitute a different dataset.
- Do not commit large raw dataset files to the source repository.
- Create data/README.md with download and placement instructions.
- Validate file existence, columns, target values, missing values, and data types before training.
- Never fit preprocessing on the test set.
- Preserve the provided train/test split for final evaluation.
- Make randomness reproducible wherever applicable.

## 5. Classification and leakage prevention

Primary objective: binary classification using the dataset's `label` target, after verifying the actual schema and label meaning.

- Exclude `label` from model features.
- Exclude `attack_cat` from binary-model inputs because it reveals attack-category information derived from the outcome.
- Investigate identifiers and categorical columns before choosing features.
- Do not assume every column exists; inspect the downloaded files first.
- Fit imputers, encoders, scalers, and other learned preprocessing only on training data, preferably through scikit-learn Pipelines and ColumnTransformers.
- Keep the test set untouched during model selection.
- Document any excluded features and the reasons for exclusion.

Multiclass attack-category classification is an optional extension, not a reason to compromise the primary binary evaluation.

## 6. Modelling and evaluation

Build a simple baseline first, then compare suitable models such as Logistic Regression and Random Forest. Add more models only when they provide meaningful value.

Report appropriate metrics:
- Precision, recall, and F1-score
- Confusion matrix
- Per-class metrics
- Accuracy, with limitations explained
- Training and prediction time where feasible

Treat false positives and false negatives as important cybersecurity outcomes. Explain that high accuracy alone does not prove a detector is effective.

Generate clear, labelled charts and explain what each chart reveals. Include feature-importance analysis only when the chosen model supports a defensible interpretation.

## 7. Code quality and security

- Use clear names, functions, docstrings, and comments where helpful.
- Avoid monolithic scripts and duplicated logic.
- Handle missing files, invalid schemas, and unexpected target values with useful error messages.
- Never hard-code secrets, API keys, personal paths, or credentials.
- Do not add live scanning, unauthorised traffic collection, or offensive network actions.
- Write tests for preprocessing, feature exclusion, target validation, and relevant edge cases.
- Keep requirements.txt aligned with actual project dependencies.

## 8. Expected deliverables

Prepare these final deliverables:
1. A working Jupyter notebook (`.ipynb`) with explanations, analysis, models, metrics, and charts.
2. `requirements.txt` with necessary dependencies.
3. A clear `README.md` with overview, dataset source, setup, execution instructions, and limitations.
4. `Meemansa_ProjectReport.docx` with methodology, experiments, results, visualisations, discussion, and references.
5. Supporting source code and tests where appropriate.

The notebook must run from top to bottom in a clean environment after the dataset is placed as documented.

## 9. Reporting and honesty

- Never invent experimental results, dataset statistics, citations, or performance numbers.
- Distinguish measured results from expected outcomes.
- Record limitations, class imbalance, possible dataset bias, and generalisation concerns.
- Do not claim real-world detection capability based solely on benchmark performance.
- Keep the implementation understandable enough for the student to explain and defend during a project review.
