from argparse import ArgumentParser, Namespace
from datetime import date as Date
from pathlib import Path

import polars as pl

from exfa import LsegDataProvider, PolarsAdapter, PriceField
from exfa.config import PostgresTable, normalize_lseg_assets, tables_registry
from exfa.provider import PostgresProvider
from exfa.utils import get_last_business_date, logger

PARTIAL_OBSERVATION_COLUMNS = (
    "bid",
    "ask",
    "shares_outstanding",
    "market_cap",
    "revenue",
    "book_value",
    "operating_cash_flow",
)


def get_assets(
    file_path: Path = Path("data/EUROSTOXX.csv"),
) -> list[str]:
    universe = pl.read_csv(file_path, columns=["RIC"], schema_overrides={"RIC": pl.String})
    return normalize_lseg_assets(universe["RIC"].drop_nulls().to_list())


def pull_timeseries_data(
    lseg: LsegDataProvider[pl.DataFrame],
    assets: list[str],
    start_date: Date,
    end_date: Date,
) -> dict[PostgresTable, pl.DataFrame]:
    """
    Pull all currently supported time-series datasets required by
    the risk-factor model.
    """

    assets = normalize_lseg_assets(assets)
    market_data = lseg.get_prices(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
        fields=[
            PriceField.CLOSE,
            PriceField.HIGH,
            PriceField.LOW,
            PriceField.BID,
            PriceField.ASK,
            PriceField.VWAP,
            PriceField.VOLUME,
        ],
        allow_missing_quotes=True,
    )

    shares_outstanding = lseg.get_shares_outstanding(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
    )

    shares_outstanding = shares_outstanding.with_columns(
        pl.col("shares_outstanding").cast(pl.Int64)
    )

    value_fundamentals = lseg.get_value_fundamentals(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
        currency="EUR",
    )

    return {
        tables_registry.RefPrices_Table: market_data,
        tables_registry.ValueFundamentals_Table: shares_outstanding.join(
            value_fundamentals, on=["date", "asset"], how="full", coalesce=True, validate="1:1"
        ),
    }


def save_datasets(datasets: dict[PostgresTable, pl.DataFrame]) -> None:
    """
    Write each historical dataset to PostgreSQL using its date and asset keys.
    """

    failures = []
    with PostgresProvider(adapter=PolarsAdapter(), verbose=True) as postgres:
        for table, dataset in datasets.items():
            records = dataset.with_columns(pl.col("date").cast(pl.Date))
            result = postgres.insert_dataframe_into_table(
                table,
                df=records,
                conflict_columns=("date", "asset"),
                preserve_null_columns=PARTIAL_OBSERVATION_COLUMNS,
            )
            logger.info(f"Saved {result.inserted:,} rows to {table.qualified_name}")
            if result.rejected_rows:
                failures.append(
                    f"{table.qualified_name}: {len(result.rejected_rows):,} rejected rows "
                    f"(first five: {result.rejected_rows[:5]})"
                )

    if failures:
        raise RuntimeError("\n".join(failures))


def parse_args() -> Namespace:
    arg_parser = ArgumentParser()

    last_business_date = get_last_business_date()
    arg_parser.add_argument(
        "--start-date",
        default=last_business_date,
        type=Date.fromisoformat,
    )
    arg_parser.add_argument(
        "--end-date",
        default=last_business_date,
        type=Date.fromisoformat,
    )
    arg_parser.add_argument("--assets", default=None, type=str)
    args = arg_parser.parse_args()
    if args.start_date > args.end_date:
        arg_parser.error("--start-date must be on or before --end-date")
    return args


if __name__ == "__main__":
    args = parse_args()

    start_date = args.start_date
    end_date = args.end_date

    if args.assets is not None:
        assets = args.assets.split(",")
    else:
        assets = get_assets()

    with LsegDataProvider(adapter=PolarsAdapter(), verbose=True) as lseg:
        datasets = pull_timeseries_data(
            lseg=lseg,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
        )

    save_datasets(datasets)
