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
from exfa.provider import PostgresProvider


def read_data_from_postgres(
    provider: PostgresProvider[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = provider.get_reference_prices()[["vwap", "volume"]]
    fundamentals = provider.get_value_fundamentals()[["shares_outstanding", "market_cap"]]
    mkt_df = prices.join(fundamentals, how="left", validate="one_to_one").rename(
        columns={"vwap": "price"}
    )
    mkt_df["price"] = mkt_df["price"].astype(np.float64)

    mkt_df = mkt_df.reset_index()
    mkt_df["date"] = pd.to_datetime(mkt_df["date"]).dt.date

    sm_df = provider.get_equity_classifications()[
        ["asset", "country_of_headquarters", "trbc_economic_sector_name"]
    ].rename(
        columns={"country_of_headquarters": "country", "trbc_economic_sector_name": "industry"}
    )

    return (mkt_df, sm_df)


if __name__ == "__main__":
    with PostgresProvider() as provider:
        mkt_df, sm_df = read_data_from_postgres(provider)
    mkt_df = mkt_df.merge(sm_df, left_on="asset", right_on="asset")
    mkt_df = mkt_df.set_index(["date", "asset"]).sort_index()

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
    history = max(factor.window for factor in factors if isinstance(factor, MomentumFactor))
    prices = mkt_df["price"].unstack("date").sort_index().sort_index(axis=1)
    price_values = prices.to_numpy()
    valid_prices = np.isfinite(price_values) & (price_values > 0)
    eligible_assets = valid_prices.sum(axis=1) >= history + 2
    # Sparse dates can appear in only a few listings. Use the dates shared by
    # a majority of assets with enough history for the longest factor window.
    keep_dates = valid_prices[eligible_assets].sum(axis=0) > eligible_assets.sum() / 2
    keep_assets = eligible_assets & valid_prices[:, keep_dates].all(axis=1)
    assets = prices.index[keep_assets]
    dates = prices.columns[keep_dates]
    if len(assets) == 0 or len(dates) < history + 2:
        raise ValueError(
            f"The PostgreSQL price history has {len(assets)} complete assets across "
            f"{len(dates)} shared dates. The model needs at least one asset and "
            f"{history + 2} shared dates."
        )

    market_data = {
        column: mkt_df[column]
        .unstack("date")
        .reindex(index=assets, columns=dates)
        .to_numpy()
        for column in ("price", "country", "industry")
    }
    # Each difference is labeled by its ending date, one column after its start.
    returns_array = np.diff(np.log(market_data["price"]), axis=1)[:, history - 1 :]
    rfm = FundamentalFactorModel(factors).fit(returns_array, market_data)
    rfm_diagnostics = RiskFactorDiagnostics(rfm.covariance, rfm.loadings)
    print(rfm_diagnostics.portfolio_risk(rfm.returns))
