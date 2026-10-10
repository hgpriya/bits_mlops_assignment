"""Feature engineering + preprocessing pipeline.

Everything that transforms raw patient fields into model inputs lives in ONE
sklearn Pipeline, so training and serving apply identical logic (no skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops import config

ENGINEERED_FEATURES = ["hr_pct_of_max", "bp_stage2"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds domain-driven features on top of the raw UCI fields.

    * hr_pct_of_max - max heart rate achieved as a fraction of the
      age-predicted maximum (220 - age). Low values indicate poor exercise
      tolerance, a known cardiac risk marker, and normalise `thalach` by age.
    * bp_stage2 - resting systolic BP >= 140 mm Hg (stage-2 hypertension flag).
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        age_max = (220 - X["age"]).clip(lower=1)
        X["hr_pct_of_max"] = X["thalach"] / age_max
        X["bp_stage2"] = (X["trestbps"] >= 140).astype(float)
        # keep NaN where an input was missing so the imputer handles it
        X.loc[X["trestbps"].isna(), "bp_stage2"] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + ENGINEERED_FEATURES, dtype=object)


def build_preprocessor() -> Pipeline:
    """Raw patient DataFrame -> numeric matrix ready for any classifier."""
    numeric = config.NUMERIC_FEATURES + ["hr_pct_of_max"]
    binary = config.BINARY_FEATURES + ["bp_stage2"]
    categorical = config.CATEGORICAL_FEATURES

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("bin", binary_pipe, binary),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("engineer", ClinicalFeatures()),
        ("columns", columns),
    ])


def build_model_pipeline(classifier) -> Pipeline:
    """Full pipeline: feature engineering -> preprocessing -> classifier."""
    pre = build_preprocessor()
    return Pipeline(pre.steps + [("clf", classifier)])


def output_feature_names(model: Pipeline) -> list[str]:
    """Names of the columns the classifier actually sees."""
    return list(model.named_steps["columns"].get_feature_names_out())
