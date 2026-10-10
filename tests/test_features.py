"""Unit tests for feature engineering and the preprocessing pipeline."""
import numpy as np
import pytest

from heart_mlops import config
from heart_mlops.features import (
    ENGINEERED_FEATURES, ClinicalFeatures, build_preprocessor, output_feature_names,
)


def test_clinical_features_adds_expected_columns(patients):
    X = patients[config.ALL_FEATURES]
    out = ClinicalFeatures().fit(X).transform(X)
    for col in ENGINEERED_FEATURES:
        assert col in out.columns
    row = X.iloc[0]
    assert out.loc[0, "hr_pct_of_max"] == pytest.approx(row["thalach"] / (220 - row["age"]))
    assert out.loc[0, "bp_stage2"] == float(row["trestbps"] >= 140)


def test_clinical_features_keeps_missing_bp_missing(patients):
    X = patients[config.ALL_FEATURES].astype({"trestbps": float})
    X.loc[0, "trestbps"] = np.nan
    out = ClinicalFeatures().fit(X).transform(X)
    assert np.isnan(out.loc[0, "bp_stage2"])


def test_preprocessor_output_is_numeric_and_complete(patients):
    X = patients[config.ALL_FEATURES].astype({"chol": float, "thal": float})
    X.loc[0, "chol"] = np.nan
    X.loc[1, "thal"] = np.nan
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert not np.isnan(Xt).any()


def test_preprocessor_ignores_unseen_category(patients):
    X = patients[config.ALL_FEATURES]
    pre = build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["thal"] = 5                                  # never seen in training
    assert pre.transform(unseen).shape[1] == pre.transform(X.head(1)).shape[1]


def test_numeric_features_are_scaled(patients):
    X = patients[config.ALL_FEATURES]
    Xt = build_preprocessor().fit_transform(X)
    n_numeric = len(config.NUMERIC_FEATURES) + 1         # + hr_pct_of_max
    means = Xt[:, :n_numeric].mean(axis=0)
    assert np.allclose(means, 0, atol=1e-6)


def test_output_feature_names_include_engineered(fitted_model):
    names = output_feature_names(fitted_model)
    assert any("hr_pct_of_max" in n for n in names)
    assert any(n.startswith("cat__cp_") for n in names)
