# ExFa

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
