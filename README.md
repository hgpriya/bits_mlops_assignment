# Heart Disease Risk Prediction - End-to-End MLOps

Predicts the presence of heart disease from patient health data (UCI Heart
Disease, Cleveland subset) and serves the model as a monitored, containerised API.

> Status: **Steps 1-3 complete** (EDA, model development, experiment tracking). Later steps are added incrementally.

## Project layout
```
data/                 download script, raw/ and processed/ data
src/heart_mlops/      reusable package (config, data cleaning, ... )
scripts/              runnable pipeline steps (EDA, training, ...)
notebooks/            exploratory notebooks
tests/                unit tests (pytest)
reports/              figures, EDA summary, final report
k8s/  monitoring/     deployment & monitoring config (later steps)
screenshots/          evidence for the report
.github/workflows/    CI/CD pipeline (later step)
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1 - Data acquisition & EDA
```bash
python data/download_data.py   # -> data/raw/heart_cleveland_raw.csv
python scripts/run_eda.py      # -> data/processed/heart_clean.csv, reports/figures/, reports/eda_summary.md
```

### Cleaning decisions
| Issue | Treatment | Why |
|---|---|---|
| `?` markers in `ca`, `thal` | Read as missing; filled with most frequent value | Few rows affected; both are categorical |
| Target `num` is 0-4 severity | `target = 1 if num > 0 else 0` | Assignment asks for presence/absence |
| Duplicate rows | Dropped | Avoid leakage between CV folds |
| Implausible values (e.g. BP < 70) | Set to missing, then imputed | Treat as entry errors, keep the row |
| Model-time imputation | Re-done inside the training pipeline | Prevents leakage from the test folds |

## Steps 2 & 3 - Model development and experiment tracking
```bash
python scripts/train.py                                         # ~3-6 min
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # open http://localhost:5000
```
Outputs: `models/model.joblib`, `models/model_card.json`, `reports/model_comparison.md`,
`reports/figures/model_*.png`, and an MLflow experiment `heart-disease-risk`.

### Feature engineering
| Feature | Treatment |
|---|---|
| age, trestbps, chol, thalach, oldpeak, ca | median impute -> standard scale |
| sex, fbs, exang | most-frequent impute (already 0/1) |
| cp, restecg, slope, thal | most-frequent impute -> one-hot encode |
| **hr_pct_of_max** (new) | thalach / (220 - age): heart rate reached vs. age-predicted maximum |
| **bp_stage2** (new) | 1 if resting BP >= 140 mm Hg |

All of this lives in one scikit-learn `Pipeline` together with the classifier, so the
saved model takes raw patient fields and applies identical preprocessing at serving time.

### Model selection
* Hold-out: stratified 80/20 split; the test set is never used for tuning or selection.
* Candidates: Logistic Regression, Random Forest, Gradient Boosting.
* Tuning: `GridSearchCV`, stratified 5-fold, optimising ROC-AUC.
* Metrics: accuracy, precision, recall, F1, ROC-AUC (CV mean +/- std and test set).
* Winner = highest mean CV ROC-AUC, then refit on all data and saved.

### What MLflow records
Parent run per training session -> nested run per model family (best params, CV + test
metrics, confusion matrix, feature-importance plot, full grid results CSV) -> nested run
per grid trial. The parent also stores ROC/CV comparison plots, the model card and the
final model in MLflow format (with signature and input example).
