# Machine Learning-Based Network Intrusion Detection and Analytics System

A reproducible, educational cybersecurity analytics project that uses machine learning
to classify network-flow records as normal or potentially malicious using the public
**UNSW-NB15** dataset.

> **This is an offline dataset-based research prototype, not a live intrusion prevention system.**
 
This was made for a cybersecurity bootcamp of IBM BharatCares with the help of IBM BOB AI.
---

## Dataset

**Source:** UNSW-NB15 — https://research.unsw.edu.au/projects/unsw-nb15-dataset

**Citation:**
> Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion
> detection systems (UNSW-NB15 network data set). *2015 Military Communications and
> Information Systems Conference (MilCIS)*, 1–6. https://doi.org/10.1109/MilCIS.2015.7348942

See [`data/README.md`](data/README.md) for download instructions, expected file placement,
and schema documentation.

---

## Project structure

```
ML_Network_Intrusion_Detection/
├── app.py                              ← Streamlit dashboard (frontend)
├── data/
│   ├── README.md                       ← download + placement instructions
│   ├── UNSW_NB15_training-set.csv      ← place here (not committed)
│   └── UNSW_NB15_testing-set.csv       ← place here (not committed)
├── models/                             ← fitted Pipeline .joblib files
│   ├── DummyClassifier.joblib
│   ├── LogisticRegression.joblib
│   └── RandomForest.joblib             ← ~101 MB; loaded by dashboard, never retrained
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  ← Phase 2 EDA
├── results/
│   ├── figures/                        ← saved chart PNGs (displayed in dashboard)
│   └── metrics/                        ← saved JSON/CSV metric files (read by dashboard)
├── scripts/
│   ├── run_training.py                 ← trains all models + saves artifacts
│   ├── run_interpret.py                ← feature importance + error analysis
│   └── generate_report.py             ← regenerates the .docx report
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  ← load + validate CSV files
│   ├── eda.py                          ← reusable EDA helpers
│   ├── evaluate.py                     ← metrics, confusion matrices, plots
│   ├── features.py                     ← column exclusion constants and rationale
│   ├── interpret.py                    ← feature importance + error analysis helpers
│   ├── preprocessing.py               ← leakage-safe preprocessing pipeline
│   └── train.py                        ← model pipeline builder + serialisation
├── tests/
│   ├── test_data_loader.py
│   ├── test_eda.py
│   ├── test_evaluate.py
│   ├── test_preprocessing.py
│   └── test_train.py
├── requirements.txt
└── README.md
```

---

## Streamlit Dashboard

The project includes an interactive web dashboard ([`app.py`](app.py)) built with
[Streamlit](https://streamlit.io/). It displays all saved results — metrics, charts,
and feature-importance data — and lets you run batch predictions using the pre-trained
Random Forest pipeline, all without retraining anything.

### Launch the dashboard

```powershell
# Install dependencies first (only needed once)
.venv\Scripts\python.exe -m pip install streamlit pillow

# Run the dashboard
.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Dashboard pages

| Page | What it shows |
|---|---|
| **Overview** | Project summary, dataset stats, best-model metrics, key charts |
| **Model Comparison** | Full metrics table, confusion matrices, and metric glossary for all three models |
| **Dataset Explorer** | Class/attack-category distributions, EDA charts, 500-row data preview |
| **Feature Importance & Errors** | RF feature importance chart + table, error-analysis plots and protocol breakdown |
| **Prediction Lab** | Upload a CSV of network-flow records to get batch predictions; download results |
| **Project Information** | Dataset background, metric definitions, limitations, usage guide |

> **Important:** The dashboard never calls `.fit()` or retrains any model.
> It reads `.joblib` pipeline files and pre-computed `results/` files only.
> This is an offline educational demonstration, not a live network monitor.

---

## Setup

### 1. Prerequisites

- Python 3.13
- The `.venv` virtual environment (already created in this repo)

All required packages are pre-installed in `.venv`.  To recreate from scratch:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Place the dataset files

Download the two CSV files from the UNSW-NB15 project page and place them in `data/`:

```
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

Full instructions are in [`data/README.md`](data/README.md).

### 3. Verify the dataset loads correctly

```powershell
.venv\Scripts\python.exe -m src.data_loader
```

Expected output: row counts, column counts, label distributions, and identifier candidates
for both splits.

---

## Running the EDA notebook

```powershell
# Interactive (JupyterLab)
.venv\Scripts\jupyter.exe lab notebooks/01_exploratory_data_analysis.ipynb

# Non-interactive top-to-bottom execution
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=300 `
    notebooks/01_exploratory_data_analysis.ipynb
```

### EDA outputs

| File | Description |
|---|---|
| `results/figures/class_distribution_both_splits.png` | Normal vs attack bar chart for both splits |
| `results/figures/numeric_distributions_train.png` | Histograms for all 39 numeric features |
| `results/figures/categorical_frequencies_train.png` | Top-N bar charts for `proto`, `service`, `state` |
| `results/figures/correlation_heatmap_train.png` | Pearson correlation matrix (features only) |
| `results/figures/attack_category_distribution_train.png` | Attack-category breakdown — train |
| `results/figures/attack_category_distribution_test.png` | Attack-category breakdown — test |
| `results/metrics/eda_summary.json` | Machine-readable summary of all EDA findings |

---

## Preprocessing

### Columns excluded from model features

| Column | Reason |
|---|---|
| `label` | Binary prediction target — not a feature |
| `attack_cat` | Encodes the attack category derived from the outcome; using it would leak the label |
| `id` | Sequential row identifier; ranges overlap between splits (both start at 1); no traffic-flow meaning |

These exclusions are enforced in [`src/features.py`](src/features.py) and applied by
[`src/preprocessing.py`](src/preprocessing.py) before any transformation.

### Numeric features (39 columns)

All numeric columns except `id` and `label` are processed through:

1. **`SimpleImputer(strategy="median")`** — fills any unexpected NaN with the training median.
   Median is preferred over mean because most numeric features (byte counts, timing values)
   are heavily right-skewed.
2. **`StandardScaler`** — zero mean, unit variance. Required for Logistic Regression;
   does not hurt Random Forest.

### Categorical features (`proto`, `service`, `state`)

Each categorical column gets its own sub-pipeline:

1. **`SimpleImputer(strategy="most_frequent")`** — fills any unexpected NaN.
2. **`OneHotEncoder(handle_unknown="ignore")`** — produces a binary column per category.
   `handle_unknown="ignore"` is **required**: EDA confirmed that `state` has two values
   in the test set (`"ACC"`, `"CLO"`) not seen in training — without this flag, test
   transformation would crash.

### High-cardinality handling for `proto`

`proto` has **133 unique training values**; the top-10 cover 92.9% of rows.
`OneHotEncoder(max_categories=20)` is applied to `proto` only, which bundles rare
protocols into an implicit infrequent-category bucket.

**Trade-off:** protocols that appear only in rare attack types will be merged into the
infrequent bucket, potentially reducing discrimination on those rare sub-categories.
For the binary task (normal vs attack) this is an acceptable compromise.
`service` (13 values) and `state` (9 values) are encoded without a cap.

### Train-only fitting — leakage prevention

```python
# Correct: fit only on training data
X_train, y_train = split_features_target(train_df)
X_test,  y_test  = split_features_target(test_df)

preprocessor = build_preprocessor(X_train)
fit_preprocessor(preprocessor, X_train)   # ← training data ONLY

X_train_t = transform(preprocessor, X_train)
X_test_t  = transform(preprocessor, X_test)   # ← uses train statistics
```

Fitting on test data would leak test distribution information into imputer statistics
and scaler parameters, producing optimistic evaluation results that would not generalise.

---

## Model training and evaluation

### Train all models and generate artifacts

```powershell
.venv\Scripts\python.exe scripts/run_training.py
```

This script (runtime ~20 seconds total):
1. Loads and validates both CSV splits.
2. Trains DummyClassifier, Logistic Regression, and Random Forest — each as a
   complete `Pipeline(preprocessor + classifier)` fitted on the training split only.
3. Evaluates all models on the **held-out official test split**.
4. Saves results to the locations below.

### Artifacts generated

| Path | Contents |
|---|---|
| `models/DummyClassifier.joblib` | Fitted baseline pipeline |
| `models/LogisticRegression.joblib` | Fitted LR pipeline |
| `models/RandomForest.joblib` | Fitted RF pipeline (~96 MB) |
| `results/metrics/metrics_<Model>.json` | Full per-model metrics |
| `results/metrics/model_comparison.json` | Slim comparison summary |
| `results/metrics/model_comparison.csv` | Comparison table (CSV) |
| `results/figures/confusion_matrix_<Model>.png` | Per-model confusion matrix |
| `results/figures/roc_curves.png` | Overlay ROC curves |
| `results/figures/model_comparison.png` | Grouped bar chart — key metrics |

### Measured results (UNSW-NB15 official test set, 82,332 rows)

| Model | Accuracy | Attack Recall | Attack Precision | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| DummyClassifier | 0.5506 | 1.0000 | 0.5506 | 0.3551 | 0.5000 |
| LogisticRegression | 0.8089 | 0.9724 | 0.7527 | 0.7948 | 0.9551 |
| RandomForest | **0.8726** | **0.9858** | **0.8194** | **0.8665** | **0.9789** |

**Reading the confusion matrices (test set):**

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| DummyClassifier | 0 | 37,000 | 0 | 45,332 |
| LogisticRegression | 22,519 | 14,481 | 1,253 | 44,079 |
| RandomForest | 27,150 | 9,850 | 642 | 44,690 |

- **FN (False Negatives)** = attacks predicted as normal = **missed attacks**.
  Random Forest misses 642 attacks vs Logistic Regression's 1,253.
- **FP (False Positives)** = normal predicted as attack = **false alarms**.
  Random Forest produces 9,850 false alarms vs Logistic Regression's 14,481.
- The Dummy baseline never misses an attack (FN=0) because it always predicts "attack",
  but it generates 37,000 false alarms — useless in practice.

> **Important:** These results are from the UNSW-NB15 benchmark dataset, which has
> an unusually high fraction of attack traffic (~68% train, ~55% test). Performance
> on real network traffic — where normal traffic typically dominates — will differ.
> These numbers are not a guarantee of real-world detection capability.

### Loading a saved model for prediction

```python
from src.train import load_pipeline
from src.preprocessing import split_features_target

pipe = load_pipeline("models/RandomForest.joblib")
X_test, y_test = split_features_target(test_df)
y_pred = pipe.predict(X_test)
```

---

## Feature importance and error analysis (Phase 5)

```powershell
.venv\Scripts\python.exe scripts/run_interpret.py
```

Loads the saved Random Forest pipeline, cross-checks stored metrics (no retraining),
extracts feature importances, and runs error analysis on the test split.

| Artifact | Description |
|---|---|
| `results/figures/feature_importance_rf.png` | Top-20 feature importances (impurity-based) |
| `results/figures/error_distributions_rf.png` | FP vs FN feature distributions |
| `results/metrics/interpretation_summary.json` | Feature importances + error counts |

**Top features (measured):** `sttl` (14.1%), `ct_state_ttl` (9.5%), `dload` (6.8%).

**Limitation:** Random Forest impurity importance (MDI) can overvalue high-cardinality features
and distribute importance across correlated ones. These values describe model split behaviour,
not causal relationships with attack activity.

---

## Final deliverables

| File | Description |
|---|---|
| `app.py` | Streamlit web dashboard (frontend) |
| `notebooks/01_exploratory_data_analysis.ipynb` | Phase 2 EDA notebook |
| `notebooks/02_model_training_and_evaluation.ipynb` | Phase 4–5 results notebook |
| `requirements.txt` | Project dependencies (includes `streamlit`, `pillow`) |
| `Meemansa_ProjectReport.docx` | Full project report |
| `README.md` | This file |

---

## Generating the project report

```powershell
.venv\Scripts\python.exe scripts/generate_report.py
```

Regenerates `Meemansa_ProjectReport.docx` from saved JSON metrics. All values are measured
results — none are fabricated.

---

## Running the test suite

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

Tests use **synthetic DataFrames** only — no real dataset files are needed.
208 tests across 7 test files; no real dataset files required.

---

## Limitations

- This is an offline benchmark evaluation. Performance on UNSW-NB15 does not imply
  real-world detection capability.
- The dataset contains an unusually high proportion of attack traffic (~68% in training),
  which is the inverse of most production environments. Evaluation metrics must be
  interpreted accordingly.
- `id` is excluded from all model features because it is a sequential row identifier with
  overlapping ranges in train and test splits.
- `attack_cat` is excluded from all binary model inputs because it reveals the attack
  category — information derived from the outcome.
- The test set is kept untouched throughout model selection and is only used for final
  evaluation.
"# ML_Network_Intrusion_Detection" 
