# Comparing and reconciling fundamental and statistical risk factors

## Research question

Can the country, sector, and momentum exposures in a fundamental risk model
account for the main sources of co-movement found by PPCA? Do the remaining
latent directions improve covariance forecasts?

The fundamental model supplies economic labels and time-varying loadings. PPCA
estimates a lower-dimensional loading space directly from asset returns. Its
individual axes can change sign, and directions associated with close
eigenvalues can rotate across samples. We therefore compare loading spaces and
test whether linear combinations of their scores recover named factor returns.
[Bai and Ng, *Evaluating Latent and Observed Factors in Macroeconomics and Finance*](https://www.columbia.edu/~jb3064/papers/2006_Evaluating_latent_and_observed_factors_in_macroeconomics_and_finance.pdf)
and [*Spanning Latent and Observable Factors*](https://www.sciencedirect.com/science/article/pii/S0304407624000897)
provide precedents for assessing proxy quality and shared subspaces.

## Matched design

- Use the intersection of assets with usable PostgreSQL VWAP history and both a
  headquarters country and TRBC sector. Use the same asset order, 2022–2026
  price panel, forward-fill rule, and dates after the 252-session momentum
  warm-up. Momentum loadings use prices through the prior session.
- Re-estimate covariance from 252 sessions and score the next 21 non-overlapping
  sessions using the training mean of each asset. The first half of the
  complete blocks selects the residual PPCA rank. The second half evaluates the
  fixed choice. Later training windows may include previously observed returns.
- Compare the fundamental covariance, rank-five PPCA, PPCA with asset-specific
  residual variances, and a diagonal sample covariance. The PPCA diagonal
  variant keeps the same common covariance as PPCA, allowing a direct
  comparison of the two residual variance treatments.

## Common metrics

| Metric | Measurement |
| --- | --- |
| Covariance forecast | Mean Gaussian negative log-likelihood per asset and session, paired by test block. Lower is better. |
| Portfolio variance | Realized mean squared centered portfolio return divided by predicted variance, plus QLIKE. Use the same equal-weight, sector, and sector-neutral 126-session momentum weights for each model. |
| Reconstruction | Regress the same held-out return vector on each model's available loadings by least squares. Compare pooled \(R^2\), RMSE, and effective rank. The projection uses the realized return, so this is contemporaneous fit. |
| Residual dependence | Compare mean absolute pairwise residual correlation, the largest residual covariance eigenvalue, and residual correlation heatmaps ordered by sector. |

For portfolio variance, QLIKE is evaluated as
\(\log(\hat v) + v/\hat v\), which preserves model comparisons when realized
variance is zero. Portfolio weights are determined at the forecast origin and
held fixed within the block. The momentum portfolio demeans the asset's
momentum loading within each sector and normalizes gross exposure to one.

## Reconciliation

At each test origin, calculate principal angles between the loading spaces.
For ranks \(r_P\) and \(r_F\), the two directional overlaps are
\(\sum_i\cos^2\theta_i/r_P\) and \(\sum_i\cos^2\theta_i/r_F\). These distinguish
whether the higher-rank fundamental model contains the PPCA space from how much
of the fundamental model a smaller PPCA representation retains. Add the
intercept, country, sector, and momentum groups in that order and calculate the
incremental overlap. These increments are order-dependent.

In each training block, regress each named factor return on all PPCA scores
with an intercept. Freeze the mapping and apply it to scores computed from
subsequent returns and the training loadings. Report held-out \(R^2\) against a
training-mean forecast. The named factor returns and PPCA scores are both
estimated from the same panel, so this measures the reproducibility of the
mapping. The experiment can apply the same procedure to independent market or
sector returns supplied by CSV to test the economic interpretation externally.
No external benchmark was used in the run below.

The hybrid adds PPCA directions fitted to fundamental training residuals.
It estimates the covariance of named and residual factor returns jointly
and adds the diagonal variance left after removing the residual factors.
Rank zero is the fundamental model. This asks whether the latent directions add
forecast value beyond the named exposures. Giglio and Xiu combine named and
latent factors for a related question about risk premia. The experiment here
evaluates covariance forecasts. [Giglio and Xiu, *Asset Pricing with Omitted Factors*](https://www.journals.uchicago.edu/doi/abs/10.1086/714090)

## PostgreSQL results

Both models used **48 assets and 895 return dates**, from 27 December 2022 to
30 June 2026 after momentum warm-up. The experiment used 15 validation blocks
and 15 test blocks.

| Model | Mean test loss per asset | Paired difference vs. fundamental | Blocks beating fundamental |
| --- | ---: | ---: | ---: |
| Diagonal baseline | −2.6235 | +0.1607 | 2 / 15 |
| Fundamental | −2.7841 | 0 | — |
| PPCA | −2.8057 | −0.0215 | 11 / 15 |
| PPCA with diagonal residuals | −2.8232 | −0.0391 | 14 / 15 |
| Validation-selected hybrid | −2.7841 | 0 | — |

Validation selected zero added residual factors among ranks zero through three.
On this sample, assigning each asset its own residual variance improved PPCA
forecast performance, and the five-factor covariance forecast scored better
than the fundamental model. The fundamental model achieved higher median
reconstruction \(R^2\) (61.0% versus 45.3%) with a larger effective loading
rank (18 versus 5).

The median loading-space overlap was **71.5% of PPCA in the fundamental space**
and **19.9% in the reverse direction**. The fundamental intercept was the most
recoverable named factor in the time-series exercise, with median held-out
\(R^2\) of **0.80**. Consumer cyclicals (0.44), technology (0.38), and
252-session momentum (0.35) showed more limited overlap in factor returns.
These results support a shared market-like direction and partial overlap in
sector and style effects. The snapshot classifications, full-period price
coverage filter, and 15 test blocks limit the scope of the empirical conclusion.

## Reproduce

```sh
uv run -m experiments.compare_rfm
```

The experiment writes paired window scores, per-portfolio calibration, rank
selection, principal angles, reconstruction diagnostics, and spanning results
to `experiments/compare_rfm/`, alongside the PNG figures. Supply
`--benchmark-returns path/to/benchmarks.csv` with a `date` column and one
decimal daily return column per benchmark to include independent targets.
