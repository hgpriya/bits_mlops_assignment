"""Download the UCI Heart Disease (Cleveland) dataset.

Usage (from the project root):
    python data/download_data.py

Writes data/raw/heart_cleveland_raw.csv with proper column headers.
Source: https://archive.ics.uci.edu/dataset/45/heart+disease
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart_mlops import config  # noqa: E402


def download(url: str, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def main() -> None:
    url = config.UCI_BASE_URL + config.RAW_FILE_NAME
    print(f"Downloading {url}")
    content = download(url)
    df = pd.read_csv(io.BytesIO(content), header=None, names=config.COLUMNS)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_CSV, index=False)

    sha = hashlib.sha256(content).hexdigest()[:16]
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {config.RAW_CSV}")
    print(f"Source checksum (sha256, first 16): {sha}")


if __name__ == "__main__":
    main()
