"""Command line entry point.

Datasets are fetched, never committed. Each fetcher writes into data/raw and
then stops — conversion to Parquet happens in notebook 00 so it is visible in
the narrative rather than hidden in a script.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from churnval.config import PATHS

RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already present: {dest.name}")
        return dest
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def fetch_retail() -> None:
    """Online Retail II from the UCI repository. Small, no credentials."""
    target = PATHS.raw / "online_retail_ii"
    archive = _download(RETAIL_URL, target.with_suffix(".zip"))
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    print(f"extracted to {target}")


def fetch_kkbox() -> None:
    """KKBox competition data. Requires accepting the competition rules first.

    Deliberately not automated end to end: the rules must be accepted by a
    human on the competition page, and a script that appears to do it for you
    would be misleading.
    """
    print(
        "KKBox data requires a Kaggle account and acceptance of the competition\n"
        "rules at https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules\n\n"
        "Once accepted, with the Kaggle CLI installed and %USERPROFILE%\\.kaggle\\\n"
        "kaggle.json in place, run:\n\n"
        "  kaggle competitions download -c kkbox-churn-prediction-challenge "
        f"-f transactions.csv.7z -p \"{PATHS.raw}\"\n"
        "  kaggle competitions download -c kkbox-churn-prediction-challenge "
        f"-f members_v3.csv.7z -p \"{PATHS.raw}\"\n\n"
        "Skip user_logs unless you specifically need it — it is tens of GB and\n"
        "the core argument does not use it."
    )


FETCHERS = {"retail": fetch_retail, "kkbox": fetch_kkbox}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="churnval")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a dataset into data/raw")
    fetch.add_argument("dataset", choices=sorted(FETCHERS))

    args = parser.parse_args(argv)
    PATHS.ensure()

    if args.command == "fetch":
        FETCHERS[args.dataset]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
