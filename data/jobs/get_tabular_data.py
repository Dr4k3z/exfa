from argparse import ArgumentParser, Namespace
from datetime import date as Date
from pathlib import Path

import pandas as pd

from exfa import LsegDataProvider

UNIVERSE_PATH = Path("data/EUROSTOXX.csv")
OUTPUT_PATH = Path("data/lseg/equity_classification.parquet")


def get_assets(path: Path = UNIVERSE_PATH) -> list[str]:
    universe = pd.read_csv(path)
    return universe["RIC"].dropna().astype(str).drop_duplicates().tolist()


def parse_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--start-date", type=lambda date: Date.fromisoformat(date) if date is not None else None
    )
    arg_parser.add_argument(
        "--end-date", type=lambda date: Date.fromisoformat(date) if date is not None else None
    )
    return arg_parser.parse_args()


def main() -> None:
    assets = get_assets()

    with LsegDataProvider() as lseg:
        classifications = lseg.get_trbc(assets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    classifications.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(classifications):,} classifications -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
