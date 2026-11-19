import numpy as np
import polars as pl

from exfa import FundamentalFactorModel, PolarsAdapter
from exfa.diagnostics import RiskFactorDiagnostics
from exfa.models.fundamental_model.factors import (
    CountryFactor,
    InterceptFactor,
    MomentumFactor,
    SectorFactor,
)
from exfa.provider import PostgresProvider


def read_data_from_postgres(
    provider: PostgresProvider[pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    keys = ["date", "asset"]
    prices = provider.get_reference_prices().select(*keys, "vwap", "volume")
    fundamentals = provider.get_value_fundamentals().select(
        *keys, "shares_outstanding", "market_cap"
    )
    mkt_df = (
        prices.join(fundamentals, on=keys, how="left", validate="1:1")
        .with_columns(
            pl.col("vwap").cast(pl.Float64).alias("price"),
            pl.col("date").cast(pl.Date),
        )
        .drop("vwap")
    )
    sm_df = provider.get_equity_classifications().select(
        "asset",
        pl.col("country_of_headquarters").alias("country"),
        pl.col("trbc_economic_sector_name").alias("industry"),
    )
    return mkt_df, sm_df


if __name__ == "__main__":
    with PostgresProvider(adapter=PolarsAdapter()) as provider:
        mkt_df, sm_df = read_data_from_postgres(provider)
    mkt_df = mkt_df.join(sm_df.select("asset"), on="asset", how="semi")

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
    # Sparse dates can appear in only a few listings. Use the dates shared by
    # a majority of assets with enough history for the longest factor window.
    keep_dates = valid_prices[eligible_assets].sum(axis=0) > eligible_assets.sum() / 2
    keep_assets = eligible_assets & valid_prices[:, keep_dates].all(axis=1)
    aligned_prices = price_values[keep_assets][:, keep_dates]
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
