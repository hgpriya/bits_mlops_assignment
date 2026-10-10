"""Data loading and cleaning for the UCI Heart Disease (Cleveland) dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from heart_mlops import config

logger = logging.getLogger(__name__)


def load_raw(path: Path | str = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV produced by data/download_data.py.

    The UCI file uses '?' for missing values; these become NaN here.
    """
    df = pd.read_csv(path, na_values=["?"])
    missing_cols = set(config.COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing columns: {sorted(missing_cols)}")
    return df[config.COLUMNS]


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column (non-zero only)."""
    counts = df.isna().sum()
    report = pd.DataFrame({"missing": counts.astype(int), "pct": (counts / len(df) * 100).round(2)})
    return report[report["missing"] > 0].sort_values("missing", ascending=False)


def binarise_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary `target`: 0 = no disease, 1 = disease (num 1-4)."""
    out = df.copy()
    out[config.TARGET] = (out[config.RAW_TARGET] > 0).astype(int)
    return out.drop(columns=[config.RAW_TARGET])


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Set physiologically implausible values to NaN so they get imputed."""
    out = df.copy()
    for col, (low, high) in config.VALID_RANGES.items():
        bad = ~out[col].between(low, high) & out[col].notna()
        if bad.any():
            logger.warning("%s: %d out-of-range value(s) set to NaN", col, int(bad.sum()))
            out.loc[bad, col] = np.nan
    return out


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Simple, documented imputation for EDA / the cleaned CSV.

    * numeric -> median (robust to skew, e.g. cholesterol)
    * binary / categorical (incl. `ca`, `thal`) -> most frequent value

    Note: for modelling, imputation is re-done INSIDE the sklearn pipeline
    (fitted on training folds only) to avoid data leakage. This function only
    produces the cleaned, human-inspectable dataset.
    """
    out = df.copy()
    for col in config.NUMERIC_FEATURES:
        if col == "ca":
            continue
        out[col] = out[col].fillna(out[col].median())
    for col in ["ca"] + config.BINARY_FEATURES + config.CATEGORICAL_FEATURES:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast discrete columns to int so the CSV is clean and unambiguous."""
    out = df.copy()
    int_cols = ["age", "trestbps", "chol", "thalach", "ca"] + \
        config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET]
    for col in int_cols:
        out[col] = out[col].round().astype(int)
    out["oldpeak"] = out["oldpeak"].astype(float)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: dedupe -> range check -> target -> impute -> dtypes."""
    n0 = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    if len(out) < n0:
        logger.info("Dropped %d duplicate row(s)", n0 - len(out))
    out = flag_out_of_range(out)
    out = binarise_target(out)
    out = impute(out)
    out = enforce_dtypes(out)
    assert not out.isna().any().any(), "NaNs remain after cleaning"
    return out


def save_clean(df: pd.DataFrame, path: Path | str = config.CLEAN_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
