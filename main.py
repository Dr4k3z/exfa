import numpy as np
import polars as pl

from exfa import FundamentalFactorModel, PolarsAdapter
from exfa.config import tables_registry
from exfa.diagnostics import RiskFactorDiagnostics
from exfa.models.fundamental_model.factors import (
    CountryFactor,
    InterceptFactor,
    MomentumFactor,
    SectorFactor,
)
from exfa.provider import PostgresProvider


def read_data_from_postgres(provider: PostgresProvider[pl.DataFrame]) -> pl.DataFrame:
    query = """
        SELECT prices.date, prices.asset, prices.vwap::double precision AS price,
               classifications.country_of_headquarters AS country,
               classifications.trbc_economic_sector_name AS industry
        FROM {} AS prices
        JOIN {} AS classifications USING (asset)
    """
    return provider.read_query(
        query,
        tables=(tables_registry.RefPrices_Table, tables_registry.EquityClassification_Table),
    )


def select_model_sessions(covered_sessions: np.ndarray) -> slice:
    """
    Select the longest uninterrupted run of covered sessions, preferring the latest tie.
    """

    edges = np.diff(covered_sessions.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    if not len(starts):
        return slice(0, 0)
    lengths = stops - starts
    longest = np.flatnonzero(lengths == lengths.max())[-1]
    return slice(starts[longest], stops[longest])


if __name__ == "__main__":
    with PostgresProvider(adapter=PolarsAdapter()) as provider:
        mkt_df = read_data_from_postgres(provider)
    sm_df = mkt_df.select("asset", "country", "industry").unique()

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
    prices = mkt_df.pivot(on="date", index="asset", values="price", sort_columns=True).sort(
        "asset"
    )
    price_values = prices.drop("asset").to_numpy()
    valid_prices = np.isfinite(price_values) & (price_values > 0)
    eligible_assets = valid_prices.sum(axis=1) >= history + 2
    # Keep one contiguous run of observed sessions so a coverage gap cannot
    # turn a multi-session move into a daily return.
    covered_sessions = valid_prices[eligible_assets].sum(axis=0) > eligible_assets.sum() / 2
    sessions = select_model_sessions(covered_sessions)
    keep_assets = eligible_assets & valid_prices[:, sessions].all(axis=1)
    aligned_prices = price_values[keep_assets, sessions]
    asset_count, date_count = aligned_prices.shape
    if asset_count == 0 or date_count < history + 2:
        raise ValueError(
            f"The PostgreSQL price history has {asset_count} complete assets across "
            f"{date_count} shared dates. The model needs at least one asset and "
            f"{history + 2} shared dates."
        )

    classifications = (
        prices.select("asset")
        .filter(pl.Series(keep_assets))
        .join(sm_df, on="asset", how="left", validate="1:1", maintain_order="left")
    )
    market_data = {"price": aligned_prices}
    for column in ("country", "industry"):
        labels = classifications[column].to_numpy()
        market_data[column] = np.broadcast_to(labels[:, None], aligned_prices.shape)
    # Each difference is labeled by its ending date, one column after its start.
    returns_array = np.diff(np.log(market_data["price"]), axis=1)[:, history - 1 :]
    rfm = FundamentalFactorModel(factors).fit(returns_array, market_data)
    rfm_diagnostics = RiskFactorDiagnostics(rfm.covariance, rfm.loadings)
    print(rfm_diagnostics.portfolio_risk(rfm.returns))
