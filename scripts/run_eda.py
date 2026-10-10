"""Step 1: clean the raw data and produce EDA figures + a summary.

Usage (from the project root, after data/download_data.py):
    python scripts/run_eda.py

Outputs:
    data/processed/heart_clean.csv
    reports/figures/*.png
    reports/eda_summary.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - works in CI / Docker
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart_mlops import config, data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("eda")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Consistent visual identity for every figure in the report
PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
CLASS_NAMES = {0: "No disease", 1: "Disease"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.relative_to(config.PROJECT_ROOT))
    return path


def plot_missing(raw: pd.DataFrame) -> None:
    rep = data.missing_value_report(raw)
    fig, ax = plt.subplots(figsize=(6, 3))
    if rep.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(rep.index, rep["missing"], color="#264653")
        for i, (m, p) in enumerate(zip(rep["missing"], rep["pct"])):
            ax.text(m + 0.1, i, f"{m} ({p}%)", va="center")
        ax.set_xlabel("Missing rows")
    ax.set_title("Missing values in raw data")
    _save(fig, "01_missing_values.png")


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2,
                f"{v} ({v / counts.sum():.0%})", ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Class balance of target")
    _save(fig, "02_class_balance.png")


def plot_numeric_histograms(df: pd.DataFrame) -> None:
    cols = [c for c in config.NUMERIC_FEATURES if c != "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, ax=ax,
                     palette=PALETTE, element="step", stat="density",
                     common_norm=False)
        ax.set_xlabel(config.FEATURE_LABELS[col])
        ax.get_legend().set_title("")
        for t, lbl in zip(ax.get_legend().texts, CLASS_NAMES.values()):
            t.set_text(lbl)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of continuous features by outcome", fontsize=14)
    _save(fig, "03_numeric_histograms.png")


def plot_categorical_rates(df: pd.DataFrame) -> None:
    cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]
    base = df[config.TARGET].mean()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        g = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        ax.bar(g.index.astype(str), g["mean"], color="#e76f51")
        ax.axhline(base, ls="--", color="grey", lw=1)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(config.FEATURE_LABELS[col], fontsize=10)
        ax.set_ylabel("Disease rate")
    fig.suptitle("Disease rate per category (dashed = overall rate)", fontsize=14)
    _save(fig, "04_categorical_disease_rate.png")


def plot_correlation(df: pd.DataFrame) -> pd.Series:
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=True, fmt=".2f",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Feature correlation matrix (Pearson)")
    _save(fig, "05_correlation_heatmap.png")
    return corr[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False)


def plot_age_band_risk(df: pd.DataFrame) -> None:
    bands = pd.cut(df["age"], bins=[0, 40, 50, 60, 70, 120],
                   labels=["<40", "40-49", "50-59", "60-69", "70+"])
    g = df.groupby([bands, "sex"], observed=False)[config.TARGET].mean().unstack()
    g.columns = ["Female" if c == 0 else "Male" for c in g.columns]
    fig, ax = plt.subplots(figsize=(7, 4))
    g.plot(kind="bar", ax=ax, color=["#8ab17d", "#264653"], rot=0)
    ax.set_xlabel("Age band")
    ax.set_ylabel("Disease rate")
    ax.set_title("Disease rate by age band and sex")
    _save(fig, "06_age_sex_risk.png")


def plot_outliers(df: pd.DataFrame) -> None:
    cols = [c for c in config.NUMERIC_FEATURES if c != "ca"]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET,
                    palette=PALETTE, ax=ax, legend=False)
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_xlabel("Disease")
        ax.set_title(config.FEATURE_LABELS[col], fontsize=9)
        ax.set_ylabel("")
    fig.suptitle("Spread and outliers of continuous features", fontsize=13)
    _save(fig, "07_boxplots.png")


def _md_table(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> Markdown table (avoids an extra dependency)."""
    header = "| " + " | ".join([""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(r[0])] + [str(v) for v in r[1:]]) + " |"
            for r in df.itertuples()]
    return "\n".join([header, sep, *rows])


def write_summary(raw: pd.DataFrame, clean: pd.DataFrame, target_corr: pd.Series) -> None:
    miss = data.missing_value_report(raw)
    bal = clean[config.TARGET].value_counts(normalize=True).sort_index()
    lines = [
        "# EDA summary (auto-generated)",
        "",
        f"* Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns",
        f"* Clean shape: {clean.shape[0]} rows x {clean.shape[1]} columns",
        f"* Duplicates removed: {len(raw) - len(raw.drop_duplicates())}",
        f"* Class balance: no disease {bal.get(0, 0):.1%} / disease {bal.get(1, 0):.1%}",
        "",
        "## Missing values (raw)",
        _md_table(miss) if not miss.empty else "None",
        "",
        "## Correlation with target (sorted by strength)",
        _md_table(target_corr.round(3).to_frame("corr")),
        "",
        "## Descriptive statistics (clean)",
        _md_table(clean.describe().T.round(2)),
    ]
    path = config.PROJECT_ROOT / "reports" / "eda_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("saved %s", path.relative_to(config.PROJECT_ROOT))


def main() -> None:
    raw = data.load_raw()
    log.info("raw data: %s", raw.shape)
    clean = data.clean(raw)
    out = data.save_clean(clean)
    log.info("clean data: %s -> %s", clean.shape, out.relative_to(config.PROJECT_ROOT))

    plot_missing(raw)
    plot_class_balance(clean)
    plot_numeric_histograms(clean)
    plot_categorical_rates(clean)
    target_corr = plot_correlation(clean)
    plot_age_band_risk(clean)
    plot_outliers(clean)
    write_summary(raw, clean, target_corr)
    log.info("EDA complete")


if __name__ == "__main__":
    main()
