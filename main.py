from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exfa import RiskFactorModel
from exfa.factors import CountryFactor, InterceptFactor, MomentumFactor, SectorFactor


def read_data_from_parquet(
    market_data_path: Path = Path("data/lseg/equity_panel.parquet"),
    sm_data_path: Path = Path("data/lseg/equity_classification.parquet"),
) -> tuple[pd.DataFrame, ...]:
    mkt_df = pd.read_parquet(market_data_path)[
        [
            "vwap",
            "volume",
            "shares_outstanding",
            "market_cap",
        ]
    ].rename(columns={"vwap": "price"})

    mkt_df = mkt_df.reset_index()
    mkt_df["date"] = pd.to_datetime(mkt_df["date"]).dt.date

    sm_df = pd.read_parquet(sm_data_path)[
        ["asset", "country_of_headquarters", "trbc_economic_sector_name"]
    ].rename(
        columns={"country_of_headquarters": "country", "trbc_economic_sector_name": "industry"}
    )

    return (mkt_df, sm_df)


if __name__ == "__main__":
    mkt_df, sm_df = read_data_from_parquet()
    mkt_df = mkt_df.merge(sm_df, left_on="asset", right_on="asset")
    mkt_df = mkt_df.set_index(["date", "asset"]).sort_index()

    returns = mkt_df["price"].groupby(level="asset").transform(lambda x: np.log(x).diff())

    rfm = RiskFactorModel(
        [
            MomentumFactor(
                name="mom_21d", window=21, impute=True, winsorize=True, standardize=True
            ),
            MomentumFactor(
                name="mom_126d", window=126, impute=True, winsorize=True, standardize=True
            ),
            MomentumFactor(
                name="mom_252d", window=252, impute=True, winsorize=True, standardize=True
            ),
            CountryFactor(column="country"),
            SectorFactor(sector_col="industry"),
            InterceptFactor(),
        ]
    )

    result = rfm.fit(mkt_df, returns)

    print(result.vif.describe())

    result.diagnostics["r2"].plot()
    plt.show()
