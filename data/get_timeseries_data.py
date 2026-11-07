from datetime import date as Date
from pathlib import Path

import pandas as pd

from exfa import LsegDataProvider, PriceField

DATA_DIR = Path("data/lseg")


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


def save_datasets(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path = DATA_DIR,
) -> None:
    """
    Persist individual normalized datasets as parquet.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, df in datasets.items():
        path = output_dir / f"{name}.parquet"

        df.to_parquet(
            path,
            index=True,
        )

        print(f"Saved {name}: {len(df):,} rows -> {path}")


def main(
    assets: list[str],
    start_date: Date,
    end_date: Date,
) -> None:

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LsegDataProvider() as lseg:
        datasets = pull_timeseries_data(
            lseg=lseg,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
        )

    save_datasets(datasets)

    panel = build_equity_panel(datasets)

    panel_path = DATA_DIR / "equity_panel.parquet"

    panel.to_parquet(
        panel_path,
        index=True,
    )

    print(f"Saved equity panel: {len(panel):,} rows -> {panel_path}")


if __name__ == "__main__":
    assets = get_assets()

    start_date = Date(2022, 1, 1)
    end_date = Date(2026, 7, 31)

    main(
        assets=assets,
        start_date=start_date,
        end_date=end_date,
    )
