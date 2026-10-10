"""Evaluation plots logged to MLflow and used in the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from heart_mlops.features import output_feature_names  # noqa: E402


def plot_confusion(model, X, y, title: str, path: Path) -> Path:
    cm = confusion_matrix(y, model.predict(X))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(model, title: str, path: Path, top: int = 15) -> Path:
    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in output_feature_names(model)]
    if hasattr(clf, "coef_"):
        values, xlabel = clf.coef_[0], "Coefficient (standardised; + = higher risk)"
    else:
        values, xlabel = clf.feature_importances_, "Impurity-based importance"
    s = pd.Series(values, index=names)
    s = s.reindex(s.abs().sort_values().index).tail(top)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(s.index, s.values,
            color=["#e76f51" if v > 0 else "#2a9d8f" for v in s.values])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_comparison(fitted: dict, X, y, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - hold-out test set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cv_comparison(summary: pd.DataFrame, metrics: list[str], folds: int, path: Path) -> Path:
    x = np.arange(len(metrics))
    width = 0.8 / len(summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, row) in enumerate(summary.iterrows()):
        means = [float(row[f"cv_{m}_mean"]) for m in metrics]
        stds = [float(row[f"cv_{m}_std"]) for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=name)
    ax.set_xticks(x + width * (len(summary) - 1) / 2, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean +/- std over folds)")
    ax.set_title(f"{folds}-fold cross-validation comparison")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
