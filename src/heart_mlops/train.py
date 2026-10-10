"""Model training, tuning, evaluation and MLflow experiment tracking.

Flow
1. Stratified 80/20 split -> hold-out test set never used for tuning/selection.
2. Each candidate: GridSearchCV (stratified 5-fold, ROC-AUC). Every grid
   candidate is logged as a nested MLflow run.
3. Best config per model: 5-fold CV metrics (accuracy, precision, recall,
   F1, ROC-AUC) + hold-out test metrics + plots, logged to MLflow.
4. Winner = highest mean CV ROC-AUC. It is refit on ALL data and saved to
   models/model.joblib + models/model_card.json and logged as an MLflow model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from heart_mlops import config, plots
from heart_mlops.features import build_model_pipeline

log = logging.getLogger("train")

SCORING = {m: m for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}


def candidate_models() -> dict[str, tuple[object, dict]]:
    """Model name -> (estimator, hyper-parameter grid)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=config.SEED),
            {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.SEED, n_jobs=1),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 4, 6],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", 0.5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=config.SEED),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.8, 1.0],
            },
        ),
    }


def load_training_data(path: Path | str = config.CLEAN_CSV) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df[config.ALL_FEATURES], df[config.TARGET]


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holdout_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def clean_params(params: dict) -> dict:
    return {k.replace("clf__", ""): ("None" if v is None else v) for k, v in params.items()}


def _reviewed_skops_types(model) -> list[str]:
    """Types in our own pipeline that skops does not trust by default.

    Newer MLflow saves sklearn models with the pickle-free *skops* format, which
    refuses non-builtin types (our ClinicalFeatures transformer, numpy.dtype,
    tree internals). We built this model ourselves, so we list those types
    explicitly as reviewed/trusted.
    """
    try:
        import skops.io as sio
        return sorted(sio.get_untrusted_types(data=sio.dumps(model)))
    except Exception:  # skops not installed / older MLflow -> nothing to add
        return []


def log_sklearn_model(model, X_example: pd.DataFrame) -> None:
    """Log the pipeline in MLflow format (works across MLflow 2.x and 3.x)."""
    signature = infer_signature(X_example, model.predict_proba(X_example))
    base = dict(signature=signature, input_example=X_example.head(3))
    trusted = _reviewed_skops_types(model)
    if trusted:
        log.info("Trusting reviewed model types for skops: %s", trusted)

    attempts = []
    if trusted:
        attempts.append({"name": "model", "skops_trusted_types": trusted})
    attempts += [{"name": "model"}, {"artifact_path": "model"}]
    for i, extra in enumerate(attempts):
        try:
            mlflow.sklearn.log_model(model, **base, **extra)
            mlflow.set_tag("mlflow_model_logged", "true")
            return
        except TypeError:  # argument not supported by this MLflow version
            if i == len(attempts) - 1:
                raise


def tune_and_evaluate(name, estimator, grid, X_train, y_train, X_test, y_test, cv, fig_dir):
    """Grid-search one model family, log everything, return best pipeline + metrics."""
    search = GridSearchCV(build_model_pipeline(estimator), grid,
                          scoring=config.SELECTION_METRIC, cv=cv, n_jobs=-1,
                          refit=True, return_train_score=True)

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tags({"model_family": name, "stage": "best_of_family"})
        search.fit(X_train, y_train)

        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-trial-{i:02d}", nested=True):
                mlflow.set_tags({"model_family": name, "stage": "grid_trial"})
                mlflow.log_params(clean_params(params))
                mlflow.log_metrics({
                    "cv_roc_auc_mean": float(res["mean_test_score"][i]),
                    "cv_roc_auc_std": float(res["std_test_score"][i]),
                    "train_roc_auc_mean": float(res["mean_train_score"][i]),
                    "fit_time_s": float(res["mean_fit_time"][i]),
                })

        best = search.best_estimator_
        mlflow.log_params(clean_params(search.best_params_))
        mlflow.log_param("grid_size", len(res["params"]))

        scores = cross_validate(clone(best), X_train, y_train, cv=cv,
                                scoring=SCORING, n_jobs=-1)
        cv_metrics = {}
        for m in SCORING:
            cv_metrics[f"cv_{m}_mean"] = float(np.mean(scores[f"test_{m}"]))
            cv_metrics[f"cv_{m}_std"] = float(np.std(scores[f"test_{m}"]))
        test_metrics = {f"test_{k}": float(v)
                        for k, v in holdout_metrics(best, X_test, y_test).items()}
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        for p in (
            plots.plot_confusion(best, X_test, y_test, f"{name} - test set",
                                 fig_dir / f"model_cm_{name}.png"),
            plots.plot_feature_importance(best, f"{name} - top features",
                                          fig_dir / f"model_features_{name}.png"),
        ):
            mlflow.log_artifact(str(p), "plots")

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"grid_results_{name}.csv"
            pd.DataFrame(res).drop(columns=["params"]).to_csv(p, index=False)
            mlflow.log_artifact(str(p), "grid_search")

        log.info("%-20s CV ROC-AUC=%.3f  test ROC-AUC=%.3f  best=%s",
                 name, cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"],
                 clean_params(search.best_params_))
    return best, {**cv_metrics, **test_metrics,
                  "best_params": clean_params(search.best_params_)}


def write_comparison_report(summary: pd.DataFrame, winner: str) -> Path:
    table = summary.drop(columns=["best_params"]).astype(float).round(3)
    lines = [
        "# Model comparison (auto-generated)", "",
        f"Selected model: **{winner}** (highest mean {config.CV_FOLDS}-fold CV ROC-AUC)", "",
        "| model | " + " | ".join(table.columns) + " |",
        "|---|" + "---|" * len(table.columns),
        *["| " + n + " | " + " | ".join(str(v) for v in r) + " |"
          for n, r in zip(table.index, table.values)],
        "", "## Best hyper-parameters",
        *[f"* **{n}**: `{summary.loc[n, 'best_params']}`" for n in summary.index],
    ]
    path = config.PROJECT_ROOT / "reports" / "model_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    log.info("MLflow tracking URI: %s", tracking_uri)

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    log.info("train=%d  test=%d  positive rate=%.2f", len(X_train), len(X_test), y.mean())

    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = f"training-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags({"stage": "training_session",
                         "data_sha256": file_sha256(config.CLEAN_CSV)[:16]})
        mlflow.log_params({
            "n_rows": len(X), "n_train": len(X_train), "n_test": len(X_test),
            "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
            "seed": config.SEED, "selection_metric": config.SELECTION_METRIC,
        })

        fitted, results = {}, {}
        for name, (est, grid) in candidate_models().items():
            fitted[name], results[name] = tune_and_evaluate(
                name, est, grid, X_train, y_train, X_test, y_test, cv, fig_dir)

        summary = pd.DataFrame(results).T
        for p in (
            plots.plot_roc_comparison(fitted, X_test, y_test,
                                      fig_dir / "model_roc_comparison.png"),
            plots.plot_cv_comparison(summary, list(SCORING), config.CV_FOLDS,
                                     fig_dir / "model_cv_comparison.png"),
        ):
            mlflow.log_artifact(str(p), "plots")

        # Select on CV score (not test) so the test set stays an honest check.
        winner = summary[f"cv_{config.SELECTION_METRIC}_mean"].astype(float).idxmax()
        log.info("Selected model: %s", winner)

        final_model = clone(fitted[winner]).fit(X, y)  # refit on all data
        joblib.dump(final_model, config.MODEL_PATH)

        card = {
            "model_name": winner,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": parent.info.run_id,
            "selection_metric": f"cv_{config.SELECTION_METRIC}_mean",
            "best_params": results[winner]["best_params"],
            "metrics": {k: round(float(v), 4) for k, v in results[winner].items()
                        if k != "best_params"},
            "input_features": config.ALL_FEATURES,
            "target": "1 = heart disease present, 0 = absent",
            "decision_threshold": 0.5,
            "training_rows": int(len(X)),
            "data_sha256": file_sha256(config.CLEAN_CSV),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        }
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, default=str))

        mlflow.set_tag("selected_model", winner)
        mlflow.log_metrics({f"selected_{k}": v for k, v in card["metrics"].items()})
        mlflow.log_artifact(str(config.MODEL_CARD_PATH))
        mlflow.log_artifact(str(write_comparison_report(summary, winner)))
        log_sklearn_model(final_model, X_train)
        log.info("Saved %s and %s", config.MODEL_PATH, config.MODEL_CARD_PATH)
    return card
