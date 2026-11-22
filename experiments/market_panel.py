from datetime import date as Date

import numpy as np
import pandas as pd

from exfa.config import tables_registry
from exfa.provider import PostgresProvider


def prepare_price_panel(
    prices: pd.DataFrame, max_fill_sessions: int
) -> tuple[pd.DataFrame, int, int]:
    if max_fill_sessions < 0:
        raise ValueError("The forward-fill limit must be nonnegative.")
    prices = prices.astype(float)
    valid = np.isfinite(prices) & (prices > 0)
    clean_prices = prices.where(valid)
    if max_fill_sessions:
        clean_prices = clean_prices.ffill(axis=1, limit=max_fill_sessions)
    complete = clean_prices.notna().to_numpy().all(axis=1)
    fitted_prices = clean_prices.iloc[complete]
    valid_observations = valid.to_numpy()
    recovered_assets = int(complete.sum() - valid_observations.all(axis=1).sum())
    filled_observations = int((~valid_observations[complete]).sum())
    return fitted_prices, recovered_assets, filled_observations


def read_classified_prices(
    start_date: Date, end_date: Date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with PostgresProvider() as provider:
        reference_prices = provider.get_reference_prices(
            start_date=start_date, end_date=end_date
        )
        classifications = provider.read_query(
            """
            SELECT asset, country_of_headquarters AS country,
                   trbc_economic_sector_name AS industry
            FROM {}
            """,
            tables=(tables_registry.EquityClassification_Table,),
            index=("asset",),
        )
    return reference_prices["vwap"].unstack("date").sort_index(axis=1), classifications


def prepare_classified_panel(
    prices: pd.DataFrame, classifications: pd.DataFrame, max_fill_sessions: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if classifications.index.has_duplicates:
        raise ValueError("Need one country and sector classification per asset.")
    prepared, recovered, filled = prepare_price_panel(prices, max_fill_sessions)
    labels = classifications.reindex(prepared.index)
    classified = labels.notna().to_numpy().all(axis=1)
    print(
        f"Price coverage: {len(prepared)}/{len(prices)} assets, "
        f"{recovered} recovered by filling {filled} VWAP observations; "
        f"{int(classified.sum())} assets have country and sector labels."
    )
    return prepared.iloc[classified], labels.iloc[classified]
