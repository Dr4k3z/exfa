from pathlib import Path

import numpy as np
import pandas as pd

from exfa import FundamentalFactorModel
from exfa.diagnostics import RiskFactorDiagnostics
from exfa.models.fundamental_model.factors import (
    CountryFactor,
    InterceptFactor,
    MomentumFactor,
    SectorFactor,
)


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

    factors = [
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
    # Factor construction uses labeled data; model estimation uses NumPy only.
    blocks = [factor.compute(mkt_df) for factor in factors]
    loadings_panel = pd.concat(blocks, axis=1)
    if loadings_panel.columns.has_duplicates:
        raise ValueError("Factor loading names must be unique")
    return_panel = returns.unstack("date").sort_index().sort_index(axis=1)
    assets, dates = return_panel.index, return_panel.columns
    loadings = np.stack(
        [
            loadings_panel[column]
            .unstack("date")
            .reindex(index=assets, columns=dates)
            .to_numpy()
            for column in loadings_panel.columns
        ],
        axis=1,
    )
    returns_array = return_panel.to_numpy()
    valid_times = np.isfinite(returns_array).all(axis=0) & np.isfinite(loadings).all(
        axis=(0, 1)
    )
    categorical_groups = []
    offset = 0
    for factor, block in zip(factors, blocks, strict=True):
        if factor.requires_constraint:
            categorical_groups.append(tuple(range(offset, offset + block.shape[1])))
        offset += block.shape[1]
    rfm = FundamentalFactorModel().fit(
        returns=returns_array[:, valid_times],
        loadings=loadings[:, :, valid_times],
        categorical_groups=categorical_groups,
    )
    rfm_diagnostics = RiskFactorDiagnostics(rfm.covariance, rfm.loadings)
    print(rfm_diagnostics.portfolio_risk(rfm.returns))
