"""Steps 2 & 3: feature engineering, model training/tuning, MLflow tracking.

Usage (from the project root, after scripts/run_eda.py):
    python scripts/train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart_mlops.train import main  # noqa: E402

if __name__ == "__main__":
    main()
