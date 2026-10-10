"""Smoke-test a running API: /health, /predict, bad input, /metrics.

Usage:
    python scripts/smoke_test.py                       # http://localhost:8000
    python scripts/smoke_test.py --url http://<ip>     # e.g. Kubernetes LoadBalancer
Exits non-zero on any failure, so CI can use it as a gate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def wait_for_health(url: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return r.json()
            last_err = f"status {r.status_code}: {r.text}"
        except requests.RequestException as exc:
            last_err = str(exc)
        time.sleep(2)
    raise RuntimeError(f"API not healthy after {timeout}s ({last_err})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    sample = json.loads((ROOT / "sample_request.json").read_text())

    print("1) health   ->", wait_for_health(url, args.timeout))

    r = requests.post(f"{url}/predict", json=sample, timeout=10)
    r.raise_for_status()
    body = r.json()
    assert body["prediction"] in (0, 1), body
    assert 0.5 <= body["confidence"] <= 1.0, body
    print("2) predict  ->", json.dumps(body))

    bad = {**sample, "age": 500}
    r = requests.post(f"{url}/predict", json=bad, timeout=10)
    assert r.status_code == 422, f"expected 422, got {r.status_code}"
    print("3) invalid input correctly rejected with 422")

    metrics = requests.get(f"{url}/metrics", timeout=10).text
    assert "api_requests_total" in metrics
    print("4) metrics  -> api_requests_total present")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - print a clear failure for CI logs
        print(f"SMOKE TEST FAILED: {exc}")
        sys.exit(1)
