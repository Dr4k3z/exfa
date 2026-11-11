# ExFa

## TODO:
- Define a proper way to combine factors together, e.g.

    ```
    data["MOMENTUM"] = (
        0.5 * data["mom_12m_1m_z"]
        + 0.5 * data["mom_6m_1m_z"]
    )

    data["SHORT_TERM"] = data["mom_1m_z"]
    ```

    For this it could be helpful to move to Polars (see point 4)

- Abstraction `DataProvider` layer that can support different data sources and standardizes the way data is fetched, manipulated and merged.

- Move to Polars, possibly using `pl.Expr` for the factors computation

- Support for Statical factors (PCA)

- Factor pushout

- Risk Variance Decomposition

- Make it faster (moving to Polars might help)

- Make it more stable over time (fitting a proper model may help)

## LSEG ingestion

`LsegDataProvider` is an extraction client for scheduled ingestion jobs. It
opens one LSEG session for the duration of its context. Choose an output adapter
when constructing it. The default `PandasAdapter` preserves historical output as
sorted pandas dataframes with a unique `(date, asset)` MultiIndex. A
`PolarsAdapter` produces Polars dataframes with explicit `date` and `asset`
columns. Every data-fetching method uses the same selected adapter.

```python
from datetime import date as Date

from exfa import LsegDataProvider, PriceField


with LsegDataProvider() as lseg:
    prices = lseg.get_prices(
        assets=[".STOXX50E", "ENI.MI"],
        start_date=Date(2024, 1, 1),
        end_date=Date(2024, 12, 31),
        fields=[PriceField.CLOSE, PriceField.VOLUME],
    )

    market_cap = lseg.get_timeseries_field(
        assets=[".STOXX50E", "ENI.MI"],
        field="TR.CompanyMarketCap",
        name="market_cap",
        start_date=Date(2024, 1, 1),
        end_date=Date(2024, 12, 31),
    )

panel = prices.join(market_cap, how="left")
prices.to_parquet("prices.parquet", index=True)
```

By default, session configuration is read from `LSEG_APP_KEY`,
`LSEG_SESSION_NAME`, and `LSEG_CONFIG_NAME`. Pass an `LsegConfig` explicitly
when a job should not depend on environment variables. The lower-level
`get_data()` method is available for LSEG fields not covered by a convenience
method.

### Choose an output adapter

```python
from datetime import date as Date
from exfa import LsegDataProvider, PandasAdapter, PolarsAdapter

with LsegDataProvider(adapter=PolarsAdapter()) as lseg:
    classifications = lseg.get_trbc(["ENI.MI", "VOD.L"])
    prices = lseg.get_prices(
        ["ENI.MI", "VOD.L"], Date(2024, 1, 1), Date(2024, 12, 31),
    )
    market_cap = lseg.get_market_cap(
        ["ENI.MI", "VOD.L"], Date(2024, 1, 1), Date(2024, 12, 31),
    )

panel = prices.join(market_cap, on=["date", "asset"], how="left", validate="1:1")
panel.write_parquet("prices.parquet")

# Explicit pandas selection is equivalent to the default:
pandas_provider = LsegDataProvider(adapter=PandasAdapter())
```

The provider uses the SDK content APIs and their decoded JSON
`response.data.raw`, rather than `ld.get_data()` or `response.data.df`.
It validates LSEG headers and rows, normalizes field names and historical keys,
and outer-joins independently observed fields using ordinary Python values.
Only then does the adapter build its dataframe. The Polars path never converts
through pandas; the LSEG SDK itself still depends on pandas.

Historical timestamps are normalized to UTC without timezone metadata. Exact
repeated observations are deduplicated; conflicting `(date, asset)` observations
and duplicate output names raise `LsegSchemaError`. Rows missing a date or asset,
and rows with no observed values, are omitted. Invalid JSON table shapes raise
`LsegSchemaError`; failed or partially errored responses raise `LsegProviderError`
instead of silently returning partial data. No filling across dates occurs.

`get_data()` now uses `fundamental_and_reference.Definition` and retains the
response's column titles. It targets the fundamental/reference content API;
the convenience SDK function's additional mixed pricing/streaming routing is
not reproduced. Use `get_prices()` for historical pricing. The adapter is a
keyword-only constructor argument, and its output type propagates through all
request return annotations, including convenience and combined-field methods.

Custom adapters implement `Adapter[T].shape_raw_data(raw_data, *, index=())`.
`raw_data` is a column mapping (`TableData` in `exfa.provider.adapters`) of decoded,
normalized Python values, not a JSON string or an SDK dataframe. For historical
results, `index` is `("date", "asset")`; reference results use `()`.
Adapters are called once per public request and may return any type `T`.
They should preserve column values, row order, and keys. For example:

```python
from collections.abc import Sequence
from exfa.provider.adapters import TableData

class RecordsAdapter:
    def shape_raw_data(
        self, raw_data: TableData, *, index: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        return [dict(zip(raw_data, row, strict=True))
                for row in zip(*raw_data.values(), strict=True)]

with LsegDataProvider(adapter=RecordsAdapter()) as lseg:
    rows = lseg.get_trbc(["ENI.MI"])  # list[dict[str, object]]
```

## Daily fundamental regression diagnostics

The NumPy diagnostic report evaluates the stored daily residuals and historical
loadings, without refitting or substituting the last date's loadings:

```python
from exfa import FundamentalFactorModel
from exfa.diagnostics import compute_model_regression_report
from exfa.models.fundamental_model.factors import CountryFactor, InterceptFactor, MomentumFactor

model = FundamentalFactorModel([
    InterceptFactor(), CountryFactor(), MomentumFactor("momentum", window=20),
])
model.fit(returns=returns, market_data=market_data)
# With two countries: intercept, country DE, country US, momentum.
report = compute_model_regression_report(model, continuous_columns=[3])
print(report.r2)                  # (times,), centered across assets each day
print(report.vif)                 # (selected continuous columns, times)
print(report.constraint_error)    # (times,), maximum absolute constraint error
print(report.fitted_return_corr)  # (times,)
print(report.n_obs, report.n_dropped, report.dropped_assets)
```

`DailyRegressionReport` is a frozen dataclass with named metric attributes,
rather than a tuple. Observation counts are properties derived from `num_assets`
and `dropped_assets`; they are not stored separately. Use attribute access rather
than tuple unpacking or positional report construction.

Specify the continuous factor indices explicitly: exclude intercepts and complete
categorical groups. An empty selection (the default) produces an empty VIF matrix.
VIF follows the previous pipeline's uncentered auxiliary regressions; constant
columns and insufficient observations yield NaN, and collinearity yields infinity.
R² and correlation are NaN when the required cross-sectional variation is absent.
There is no clipping of negative R² values.

`compute_daily_regression_report(returns, residuals, loadings_history,
factor_returns, continuous_columns=..., constraints=...)` also accepts stored
arrays directly. Returns and residuals use `(assets, times)`, historical loadings
use `(assets, factors, times)`, and factor returns use `(factors, times)`. Each
metric uses rows with finite actual returns, residuals, and loading columns;
dropped asset indices refer to input row positions. Constraints are optional and
must contain one entry per time; absent constraints report zero error. Metrics are
unweighted, matching the former daily report, even when estimation used WLS.
Dates and factor names stay in the caller's preprocessing layer, so, for example,
`pd.Series(report.r2, index=dates[valid_times]).plot()` recreates the earlier
R² plot when preprocessing applied a `valid_times` mask. The date index must
match exactly the input time columns used for fitting.

This report is separate from aggregate covariance/reconstruction diagnostics and
does not change the model's current complete-panel fitting requirement. The
reconstruction diagnostics using one fixed loading matrix are not a substitute
for these historical regression metrics.
