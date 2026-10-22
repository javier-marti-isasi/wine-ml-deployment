"""Smoke test HTTP sobre API local, Docker o Azure.

python -m scripts.smoke_test --base-url http://localhost:8000
"""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = args.base_url.rstrip("/")
    with urlopen(f"{url}/health", timeout=30) as response:
        assert response.status == 200
        assert json.load(response)["status"] == "ok"

    payload = (ROOT / "example-request.json").read_bytes()
    request = Request(
        f"{url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        assert response.status == 200
        result = json.load(response)
        assert 0 <= result["probability_good"] <= 1
        print("Smoke test OK:", result)


if __name__ == "__main__":
    main()
