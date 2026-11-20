from argparse import ArgumentParser, Namespace
from pathlib import Path

import polars as pl

from exfa import LsegDataProvider, PolarsAdapter
from exfa.config import normalize_lseg_assets, tables_registry
from exfa.provider import PostgresProvider
from exfa.utils.logger import logger

UNIVERSE_PATH = Path("data/EUROSTOXX.csv")


def get_assets(path: Path = UNIVERSE_PATH) -> list[str]:
    universe = pl.read_csv(path, columns=["RIC"], schema_overrides={"RIC": pl.String})
    return normalize_lseg_assets(universe["RIC"].drop_nulls().to_list())


def parse_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--assets", default=None, type=str)
    return arg_parser.parse_args()


def main(assets: list[str] | None = None) -> None:
    assets = get_assets() if assets is None else normalize_lseg_assets(assets)

    with LsegDataProvider(adapter=PolarsAdapter(), verbose=True) as lseg:
        classifications = lseg.get_trbc(assets)

    table = tables_registry.EquityClassification_Table
    with PostgresProvider(adapter=PolarsAdapter()) as postgres:
        result = postgres.insert_dataframe_into_table(
            table,
            df=classifications,
            conflict_columns="asset",
        )

    logger.info(f"Saved {result.inserted:,} rows to {table.qualified_name}")
    if result.rejected_rows:
        raise RuntimeError(f"Rejected classification rows: {result.rejected_rows[:5]}")


if __name__ == "__main__":
    args = parse_args()
    main(args.assets.split(",") if args.assets is not None else None)
