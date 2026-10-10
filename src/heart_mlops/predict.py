"""Inference helpers shared by the CLI script, the API and the tests.

The saved artifact is the FULL sklearn pipeline (feature engineering +
preprocessing + classifier), so callers pass raw patient fields only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd

from heart_mlops import config

# Probability cut-offs for a human-friendly risk band.
RISK_BANDS = ((0.30, "low"), (0.60, "moderate"), (1.01, "high"))


def load_model(path: Path | str = config.MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python scripts/train.py` first.")
    return joblib.load(path)


def load_model_card(path: Path | str = config.MODEL_CARD_PATH) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def risk_band(probability: float) -> str:
    for upper, label in RISK_BANDS:
        if probability < upper:
            return label
    return "high"


def predict_records(model, records: Iterable[dict], threshold: float = 0.5) -> list[dict]:
    """Score raw patient records -> prediction, probability, confidence, band."""
    frame = pd.DataFrame(list(records))
    missing = [c for c in config.ALL_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")
    proba = model.predict_proba(frame[config.ALL_FEATURES])[:, 1]
    results = []
    for p in proba:
        p = float(p)
        pred = int(p >= threshold)
        results.append({
            "prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "probability_disease": round(p, 4),
            "confidence": round(p if pred else 1 - p, 4),
            "risk_band": risk_band(p),
        })
    return results
