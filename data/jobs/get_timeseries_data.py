from datetime import date as Date
from pathlib import Path

import pandas as pd

from exfa import LsegDataProvider, PriceField
from exfa.provider.postgres_provider import PostgresProvider


def get_assets(
    file_path: Path = Path("data/EUROSTOXX.csv"),
) -> list[str]:
    df = pd.read_csv(file_path)

    return df["RIC"].dropna().astype(str).drop_duplicates().tolist()


def pull_timeseries_data(
    lseg: LsegDataProvider,
    assets: list[str],
    start_date: Date,
    end_date: Date,
) -> dict[str, pd.DataFrame]:
    """
    Pull all currently supported time-series datasets required by
    the risk-factor model.
    """

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
    )

    shares_outstanding = lseg.get_shares_outstanding(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
    )

    shares_outstanding = shares_outstanding.astype({"shares_outstanding": "Int64"})

    value_fundamentals = lseg.get_value_fundamentals(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
        currency="EUR",
    )

    return {
        "market_data": market_data,
        "shares_outstanding": shares_outstanding,
        "value_fundamentals": value_fundamentals,
    }


def build_equity_panel(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    market_data = datasets["market_data"]
    shares = datasets["shares_outstanding"]
    fundamentals = datasets["value_fundamentals"]

    return market_data.join(shares, how="left", validate="one_to_one").join(
        fundamentals, how="left", validate="one_to_one"
    )


def save_datasets(datasets: dict[str, pd.DataFrame]) -> None:
    """
    Write each historical dataset to PostgreSQL using its date and asset keys.
    """

    failures = []
    with PostgresProvider() as postgres:
        for table, dataset in datasets.items():
            records = dataset.reset_index()
            records["date"] = records["date"].dt.date
            result = postgres.insert_dataframe_into_table(
                "lseg",
                table=table,
                df=records,
            )
            print(f"Saved {result.inserted:,} rows to lseg.{table}")
            if result.rejected_rows:
                failures.append(
                    f"lseg.{table}: {len(result.rejected_rows):,} rejected rows "
                    f"(first five: {result.rejected_rows[:5]})"
                )

    if failures:
        raise RuntimeError("\n".join(failures))


def main(
    assets: list[str],
    start_date: Date,
    end_date: Date,
) -> None:

    with LsegDataProvider() as lseg:
        datasets = pull_timeseries_data(
            lseg=lseg,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
        )

    datasets["equity_panel"] = build_equity_panel(datasets)
    save_datasets(datasets)


if __name__ == "__main__":
    assets = get_assets()

    start_date = Date(2022, 1, 1)
    end_date = Date(2026, 7, 31)

    main(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
    )
