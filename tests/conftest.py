"""Shared fixtures: a small synthetic dataset with the real UCI schema.

Tests never need the network or the real dataset, so CI stays fast and
deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart_mlops import config
from heart_mlops.features import build_model_pipeline


def make_patients(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.choice([0, 1, 2], n),
        "thalach": rng.integers(80, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    # target depends on a few features so models have signal to learn
    score = (df["cp"] == 4) * 1.2 + df["exang"] * 1.0 + df["oldpeak"] * 0.4 \
        + (df["thal"] == 7) * 1.0 - (df["thalach"] - 140) / 40
    df[config.TARGET] = (score + rng.normal(0, 0.5, n) > 1.3).astype(int)
    return df


@pytest.fixture
def patients() -> pd.DataFrame:
    return make_patients()


@pytest.fixture
def raw_frame(patients) -> pd.DataFrame:
    """Looks like the raw UCI file: 0-4 `num`, '?' style gaps, a duplicate."""
    raw = patients.drop(columns=[config.TARGET]).copy()
    raw["num"] = np.where(patients[config.TARGET] == 1, 2, 0)
    raw = raw.astype({"ca": float, "thal": float})
    raw.loc[0, "ca"] = np.nan
    raw.loc[1, "thal"] = np.nan
    raw.loc[2, "trestbps"] = 10            # implausible -> should be imputed
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)  # duplicate row
    return raw[config.COLUMNS]


@pytest.fixture
def fitted_model(patients):
    model = build_model_pipeline(LogisticRegression(max_iter=1000))
    return model.fit(patients[config.ALL_FEATURES], patients[config.TARGET])


@pytest.fixture
def sample_patient() -> dict:
    return {"age": 58, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270, "fbs": 0,
            "restecg": 2, "thalach": 111, "exang": 1, "oldpeak": 0.8, "slope": 1,
            "ca": 0, "thal": 7}
