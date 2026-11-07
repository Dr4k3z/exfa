# Model-independent preprocessing

`exfa.preprocessing` cleans NumPy panels without importing either model. The
pipeline is a list of callables satisfying `CleaningStep`: each accepts a
`PanelData` and returns a `PanelData`. There are no implicit steps, and ordinary
functions, callable objects, and `functools.partial` all work.

```python
from functools import partial
from exfa.preprocessing import (
    PanelData, PreprocessingPipeline, drop_empty_times, drop_incomplete_assets,
    fill_missing_loadings, require_finite, winsorize_loadings,
)

pipeline = PreprocessingPipeline([
    partial(fill_missing_loadings, columns=[0, 1]),
    partial(winsorize_loadings, columns=[0, 1], lower=0.01, upper=0.99),
    drop_empty_times,
    drop_incomplete_assets,
    require_finite,
])
cleaned = pipeline.run(PanelData(returns, loadings, weights))

# Apply identical selection to labels retained outside the model.
cleaned_assets = asset_names[cleaned.asset_indices]
cleaned_dates = dates[cleaned.time_indices]
```

Returns are `(assets, times)`. Loadings may be static `(assets, factors)` or
`(assets, factors, times)` and are normalized to the latter shape. Weights may
be `(assets,)` or `(assets, times)` and are normalized to the latter shape.
Loadings and weights are optional, so the same pipeline can prepare statistical
model inputs. All arrays are copied on entry. Built-in steps preserve input
order and never modify the caller's panel.

## Explicit missing-data policies

- NaN and infinity are treated as missing. Invalid weights (nonfinite or not
  positive) make their corresponding observations unusable.
- `fill_missing_loadings` imputes selected columns using the median of finite
  assets **at the same time**. It does not impute returns and cannot invent an
  entirely missing cross-section. Selected column indices are required: choose
  continuous columns, leaving intercepts and categorical indicator columns out.
- `winsorize_loadings` optionally clips selected columns using same-time finite
  cross-sectional quantiles. Missing entries remain missing. It is not enabled
  automatically and should not be applied indiscriminately to dummy columns.
- `drop_empty_times` drops a time only when no asset has usable returns,
  loadings, and weights together. This covers initial return and loading warmup
  periods. One isolated missing observation does **not** discard an entire date.
- `drop_incomplete_assets` then keeps assets usable at every retained time. This
  produces a complete rectangle without dropping otherwise usable dates, but it
  can discard many assets. This is an explicit sample-selection policy, not a
  universal solution to missing data. Empty selections raise an error.
- `require_finite` checks the result against current model requirements: at least
  one asset, two times, finite fields, and positive weights. Model-specific
  constraints, such as minimum cross-sectional observations, are still checked
  by the model.

No built-in imputation or winsorization step borrows information across time.
Complete-case asset filtering examines the whole supplied window: run a pipeline
on each intended estimation window separately, **never** on concatenated training
and test data. A model with a changing asset universe or arbitrarily incomplete
panels is not introduced by this module. There is no automatic temporal forward
fill, interpolation, zero-return substitution, or train/test fitting machinery.

`main.py` keeps the leading price history needed by rolling factors and removes
return warmup explicitly. It then applies `[drop_incomplete_assets, require_finite]`
to returns and applies the retained asset indices to market features before
calling `FundamentalFactorModel(factors).fit(cleaned.returns, market_data)`.
It preserves the contiguous time axis so rolling price windows stay aligned.
Factors perform their configured loading transformations during fitting.

The fundamental model accepts market features rather than precomputed loadings.
Loading transformations in this module are for independently prepared loading
panels; passing `cleaned.loadings` directly into the model is not supported.
The statistical model can consume `cleaned.returns` directly.

## Add a user-defined step

```python
from exfa.preprocessing import CleaningStep, PanelData, PreprocessingPipeline

def scale_returns(panel: PanelData) -> PanelData:
    """
    Convert percentage returns to decimal returns.
    """

    result = panel.copy()
    result.returns /= 100
    return result

step: CleaningStep = scale_returns
pipeline = PreprocessingPipeline([step, require_finite])
```

Use `panel.select(asset_mask, time_mask)` for filtering: it applies masks to every
field and carries original positions forward. Use `panel.copy()` before changing
values. Steps must retain the factor axis and preserve alignment; do not reorder
arrays independently or reset original index positions. The pipeline revalidates
array shapes and indices after each step and rejects reordered or newly invented
asset/time positions. It cannot detect a custom function that changes the meaning
of values while falsely preserving metadata; that remains the function author's
responsibility.
