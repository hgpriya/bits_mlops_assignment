"""Unit tests for training helpers and inference."""
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from heart_mlops import config
from heart_mlops.features import build_model_pipeline
from heart_mlops.predict import load_model, predict_records, risk_band
from heart_mlops.train import candidate_models, clean_params, holdout_metrics, load_training_data


def test_candidate_models_cover_required_families():
    models = candidate_models()
    assert {"logistic_regression", "random_forest"} <= set(models)
    for _, (est, grid) in models.items():
        assert hasattr(est, "predict_proba")
        assert all(k.startswith("clf__") for k in grid)


def test_holdout_metrics_are_valid_probabilities(fitted_model, patients):
    m = holdout_metrics(fitted_model, patients[config.ALL_FEATURES], patients[config.TARGET])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert m["roc_auc"] > 0.7                          # synthetic data has real signal


def test_clean_params_strips_prefix_and_none():
    assert clean_params({"clf__C": 1.0, "clf__class_weight": None}) == {
        "C": 1.0, "class_weight": "None"}


def test_load_training_data_splits_features_and_target(tmp_path, patients):
    path = tmp_path / "clean.csv"
    patients.to_csv(path, index=False)
    X, y = load_training_data(path)
    assert list(X.columns) == config.ALL_FEATURES
    assert y.name == config.TARGET


def test_pipeline_works_with_tree_model(patients):
    model = build_model_pipeline(RandomForestClassifier(n_estimators=20, random_state=0))
    model.fit(patients[config.ALL_FEATURES], patients[config.TARGET])
    proba = model.predict_proba(patients[config.ALL_FEATURES].head(5))
    assert proba.shape == (5, 2)


def test_saved_model_round_trip(tmp_path, fitted_model, sample_patient):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, path)
    out = predict_records(load_model(path), [sample_patient])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert out["risk_band"] in {"low", "moderate", "high"}


def test_predict_records_rejects_missing_fields(fitted_model, sample_patient):
    bad = {k: v for k, v in sample_patient.items() if k != "chol"}
    with pytest.raises(ValueError, match="chol"):
        predict_records(fitted_model, [bad])


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")


@pytest.mark.parametrize("p,band", [(0.1, "low"), (0.45, "moderate"), (0.9, "high")])
def test_risk_band(p, band):
    assert risk_band(p) == band
