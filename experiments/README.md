# Numerical experiments

Run the commands in the table from the repository root. From any other folder,
add `--directory` with the absolute path to your checkout before `-m`:

```sh
uv run --directory /path/to/exfa -m experiments.eigenvalue_spreading
```

You can also run any experiment by script path. Use `--directory` so uv can
find the project environment and PostgreSQL can load settings from the checkout:

```sh
uv run --directory /path/to/exfa experiments/statistical_rfm.py --help
```

| Experiment | Command | Purpose and outputs |
| --- | --- | --- |
| [Eigenvalue spreading](eigenvalue_spreading.py) | `uv run -m experiments.eigenvalue_spreading` | Uses Gaussian Monte Carlo simulations to reproduce the spectral spreading of identity and heterogeneous covariance matrices in Johnstone and Paul's Figure 2. Recreates the Marchenko–Pastur comparisons in Figure 3 and adds a quarter-circle comparison for singular values at aspect ratio one. Prints spectral summaries and writes `johnstone_paul_figure2_spreading.png` and `johnstone_paul_figure3_mp_quarter_circle.png` in this directory. |
| [PPCA eigenvalue bias](ppca_eigenvalue_bias.py) | `uv run -m experiments.ppca_eigenvalue_bias` | Compares PCA and PPCA factor variance estimates at asset-to-observation ratios of 4 and 12. Prints bias summaries and writes [estimated versus population variances](ppca_eigenvalue_comparison.png) and [signed bias with Monte Carlo standard errors](ppca_eigenvalue_bias.png). |
| [Statistical factor model](statistical_rfm.py) | `uv run -m experiments.statistical_rfm` | Fits and evaluates PPCA on daily LSEG VWAP log returns. Saves the [covariance spectrum](statistical_rfm/statistical_rfm_spectrum.png), [residual diagnostics](statistical_rfm/statistical_rfm_residuals.png), [asset reconstructions](statistical_rfm/statistical_rfm_reconstruction.png), [rank validation](statistical_rfm/statistical_rfm_rank_validation.png), [portfolio calibration](statistical_rfm/statistical_rfm_portfolio_risk.png), and [factor stability](statistical_rfm/statistical_rfm_factor_stability.png). |
| [Fundamental factor model](fundamental_rfm.py) | `uv run -m experiments.fundamental_rfm` | Fits daily cross-sectional regressions to the same PostgreSQL VWAP panel. Saves [cumulative factor returns](fundamental_rfm/fundamental_rfm_factor_returns.png), [regression diagnostics](fundamental_rfm/fundamental_rfm_regression.png), [loadings](fundamental_rfm/fundamental_rfm_loadings.png), [covariance spectrum](fundamental_rfm/fundamental_rfm_spectrum.png), [residual diagnostics](fundamental_rfm/fundamental_rfm_residuals.png), [asset reconstructions](fundamental_rfm/fundamental_rfm_reconstruction.png), [covariance validation](fundamental_rfm/fundamental_rfm_covariance_validation.png), [portfolio calibration](fundamental_rfm/fundamental_rfm_portfolio_risk.png), and [factor stability](fundamental_rfm/fundamental_rfm_factor_stability.png). |
| [Matched risk-model comparison](compare_rfm.py) | `uv run -m experiments.compare_rfm` | Compares fundamental, PPCA, diagonal-residual PPCA, and residual-factor hybrid forecasts on identical samples. Writes PNGs and CSV scores in `compare_rfm/`. See the [research note](factor_model_reconciliation.md) for the design and live results. |

The statistical factor model experiment queries all assets available in PostgreSQL
from January 3, 2022 through June 30, 2026, inclusive. It treats missing, nonfinite,
and nonpositive VWAPs as gaps and carries the
last valid VWAP forward for up to five consecutive observed sessions. Assets
with any remaining gaps, including gaps before their first valid VWAP, are
excluded. Set `--max-fill-sessions 0` for complete cases or increase it to
admit longer gaps. The experiment reports how many assets and observations
forward-filling recovered. Carried prices imply zero returns on the filled
sessions and put the price change on the next observed session.
The PostgreSQL runs for both models additionally require country and sector
coverage and share the return dates after the fundamental momentum warm-up.
The out-of-sample results are conditional on full-period price coverage. The spectrum, residuals, and asset
reconstructions use the full-period fit. The three reconstruction paths use
assets near the 10th, 50th, and 90th percentiles of explained variation.
The full-period fit also produces [explained variance and residual variance by
rank](statistical_rfm/statistical_rfm_explained_variance.png), a [signed loading heatmap](statistical_rfm/statistical_rfm_loadings.png)
with assets ordered by identifier, and [return and projected-residual correlation
heatmaps](statistical_rfm/statistical_rfm_correlations.png). The run reports factor correlations with
the equal-weight return series as an internal market proxy. The [dimensional
stability figure](statistical_rfm/statistical_rfm_dimensional_stability.png) compares adjacent
non-overlapping historical windows and Gaussian simulations from a fixed fitted
covariance. It reports median subspace similarity and interquartile ranges at each
realized asset-to-session ratio.

The rolling analysis fits on 252 sessions and evaluates the next 21, advancing
21 sessions between forecasts. The [factor-return figure](statistical_rfm/statistical_rfm_factor_returns.png)
plots the cumulative sum of daily latent scores. Each score projects that day's
centered returns onto loadings estimated strictly before its block. Consecutive loading bases
are aligned by orthogonal Procrustes so arbitrary signs and rotations do not
create discontinuities in the coordinates. Scores use latent units. The plot
starts after the initial training period and includes any final partial block.
The first half of the complete blocks compare
ranks 1 through 10 by validation likelihood. The remaining blocks compare
equal-weight portfolio volatility under PPCA and a diagonal sample covariance.
Both methods use the same training-window means to center subsequent returns.
The rank configured with `--num-factors` (default 5) is used for the test and
factor stability plots. The validation plot reports the best-scoring rank
without changing the configured rank. Use `--start-date`, `--end-date`,
`--max-factors`, `--train-sessions`, and `--test-sessions` to change the setup.

For an exploratory PPCA run on a local LSEG snapshot, provide a
`(date, asset)` index and a `vwap` column. This mode uses the snapshot's own
universe and return dates.

```sh
uv run -m experiments.statistical_rfm --parquet data/lseg/market_data.parquet
```

The fundamental experiment uses the same price query, date defaults, and
forward-fill rules as the statistical experiment. It requires a country of
headquarters and a TRBC economic sector for each asset, from the classification
table. It uses the same factor definitions as `main.py`: standardized,
winsorized and imputed 21-, 126-, and 252-session momentum, country and sector
indicators, and an intercept. Momentum uses prices through the session before
the modeled return. The 252-session warm-up therefore precedes the first
daily regression. The factor return plot shows the running sum of the daily
regression coefficients.

Rolling covariance forecasts use the preceding 252 daily factor returns and
asset residuals. Each forecast fixes the loadings available at the start of the
block and compares the next 21 sessions against a diagonal covariance benchmark.
The factor stability figure measures principal angles between consecutive
forecast loading spaces. The fitted spectrum uses the full regression history
and the latest loadings. Classifications are the available table snapshot,
so the forecast results are conditional on those labels and full-period price
coverage. Both experiments accept the same date, fill-limit, and window flags.

The PPCA experiment follows the setup of Figure 7.2 in Chapter 7 of
Giuseppe A. Paleologo's *The Elements of Quantitative Investing*. It runs 50
Gaussian trials for each of two universes (1,000 and 3,000 assets), each with
250 observations, 10 factor volatilities from 1 to 10, and unit residual variance.
PCA uses the leading sample covariance eigenvalues as factor variance estimates.
PPCA subtracts the mean of the remaining sample eigenvalues, including zeros.
[Publisher's book page](https://www.wiley.com/en-us/The%2BElements%2Bof%2BQuantitative%2BInvesting-p-9781394265466)

Figures use the shared Matplotlib theme in `plotting.py`, with Computer Modern
text and math, a white background, and a muted palette. The theme uses fonts
bundled with Matplotlib and requires no LaTeX installation.
