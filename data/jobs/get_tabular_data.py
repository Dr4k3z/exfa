from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

from exfa import LsegDataProvider
from exfa.provider import PostgresProvider
from exfa.utils.logger import logger

UNIVERSE_PATH = Path("data/EUROSTOXX.csv")


def get_assets(path: Path = UNIVERSE_PATH) -> list[str]:
    universe = pd.read_csv(path)
    return universe["RIC"].dropna().astype(str).drop_duplicates().tolist()


def parse_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--assets", default=None, type=str)
    return arg_parser.parse_args()


def main(assets: list[str] | None = None) -> None:
    if assets is None:
        assets = get_assets()

    with LsegDataProvider(verbose=True) as lseg:
        classifications = lseg.get_trbc(assets)

    with PostgresProvider() as postgres:
        result = postgres.insert_dataframe_into_table(
            "lseg",
            table="equity_classification",
            df=classifications,
            conflict_columns="asset",
        )

    logger.info(f"Saved {result.inserted:,} rows to lseg.equity_classification")
    if result.rejected_rows:
        raise RuntimeError(f"Rejected classification rows: {result.rejected_rows[:5]}")


if __name__ == "__main__":
    args = parse_args()
    main(args.assets.split(",") if args.assets is not None else None)
