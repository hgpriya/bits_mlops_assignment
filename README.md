# Heart Disease Risk Prediction - End-to-End MLOps

Predicts the presence of heart disease from patient health data (UCI Heart
Disease, Cleveland subset) and serves the model as a monitored, containerised API.

> Status: **Step 1 (Data acquisition & EDA) complete.** Later steps are added incrementally.

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
