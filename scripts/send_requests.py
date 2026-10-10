"""Generate realistic traffic against the API (for the monitoring demo).

Sends real patient rows from the cleaned dataset, plus ~5% deliberately
invalid requests so the error-rate panel has something to show.

Usage:
    python scripts/send_requests.py --url http://localhost:8000 --n 300 --rps 5
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--rps", type=float, default=5.0)
    parser.add_argument("--error-rate", type=float, default=0.05)
    args = parser.parse_args()

    rows = pd.read_csv(ROOT / "data" / "processed" / "heart_clean.csv")[FEATURES]
    records = rows.to_dict("records")
    ok = failed = 0
    for i in range(args.n):
        rec = dict(random.choice(records))
        if random.random() < args.error_rate:
            rec["age"] = 999                     # invalid on purpose -> 422
        try:
            r = requests.post(f"{args.url}/predict", json=rec, timeout=5)
            ok += r.status_code == 200
            failed += r.status_code != 200
        except requests.RequestException:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"sent {i + 1}/{args.n}  ok={ok}  non-200={failed}")
        time.sleep(1 / args.rps)
    print(f"done: ok={ok} non-200={failed}")


if __name__ == "__main__":
    main()
