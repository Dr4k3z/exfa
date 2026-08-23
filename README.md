# ExFa

## TODO:
1. Separate the Regression and Diagnostic logic from the `RiskFactorModel` class and make independent modules
2. Define a proper way to combine factors together, e.g.

    ```
    data["MOMENTUM"] = (
        0.5 * data["mom_12m_1m_z"]
        + 0.5 * data["mom_6m_1m_z"]
    )

    data["SHORT_TERM"] = data["mom_1m_z"]
    ```

    For this it could be helpful to move to Polars (see point 4)

3. Abstraction `DataProvider` layer that can support different data sources and standardizes the way data is fetched, manipulated and merged.
4. Move to Polars, possibly using `pl.Expr` for the factors computation
5. Fix `RuntimeWarning: invalid value encountered in scalar divide`

## LSEG ingestion

`LsegDataProvider` is an extraction client for scheduled ingestion jobs. It
opens one LSEG session for the duration of its context. Historical methods
return sorted pandas dataframes with a unique `(date, asset)` MultiIndex.

```python
from datetime import date

from exfa import LsegDataProvider, PriceField


with LsegDataProvider() as lseg:
    prices = lseg.get_prices(
        assets=[".STOXX50E", "ENI.MI"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        fields=[PriceField.CLOSE, PriceField.VOLUME],
    )

    market_cap = lseg.get_timeseries_field(
        assets=[".STOXX50E", "ENI.MI"],
        field="TR.CompanyMarketCap",
        name="market_cap",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

panel = prices.join(market_cap, how="left")
prices.to_parquet("prices.parquet", index=True)
```

By default, session configuration is read from `LSEG_APP_KEY`,
`LSEG_SESSION_NAME`, and `LSEG_CONFIG_NAME`. Pass an `LsegConfig` explicitly
when a job should not depend on environment variables. The lower-level
`get_data()` method is available for LSEG fields not covered by a convenience
method.
