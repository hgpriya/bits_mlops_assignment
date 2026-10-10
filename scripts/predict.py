"""Batch / single inference from the command line.

Examples (from the project root):
    python scripts/predict.py --input sample_request.json
    python scripts/predict.py --input data/processed/heart_clean.csv --limit 5
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart_mlops import config  # noqa: E402
from heart_mlops.predict import load_model, predict_records  # noqa: E402


def read_records(path: Path, limit: int | None) -> list[dict]:
    if path.suffix.lower() == ".csv":
        records = pd.read_csv(path)[config.ALL_FEATURES].to_dict("records")
    else:
        data = json.loads(path.read_text())
        records = data["patients"] if isinstance(data, dict) and "patients" in data else data
        records = records if isinstance(records, list) else [records]
    return records[:limit] if limit else records


def main() -> None:
    parser = argparse.ArgumentParser(description="Heart disease risk inference")
    parser.add_argument("--input", required=True, type=Path, help="JSON or CSV file")
    parser.add_argument("--model", default=config.MODEL_PATH, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    records = read_records(args.input, args.limit)
    for rec, res in zip(records, predict_records(model, records)):
        print(json.dumps({"input": rec, "output": res}))


if __name__ == "__main__":
    main()
