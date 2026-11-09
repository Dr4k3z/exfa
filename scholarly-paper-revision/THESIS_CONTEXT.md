# Thesis context and source map

## Working scope

Master's thesis in mathematical engineering / quantitative finance.

Working title: **Random Matrix Theory for Statistical Factor Models:
Identification, Covariance Estimation, and Portfolio Risk**.

Research question: Can Random Matrix Theory improve the identification of
latent risk factors and the out-of-sample reliability of statistical factor
risk models in high-dimensional financial markets?

The project specification and the organized paper library agree on this focused
scope. Use them as the working basis. The broader investment research plan and
the covariance-estimation notes provide background; do not let them silently
expand the thesis into an estimator competition or return-prediction project.

## Local context read during setup

Paths below are relative to this folder.

| Resource | Role and evidence status |
| --- | --- |
| `../RMT_Statistical_Factor_Risk_Engine_Agent_Spec.md` | Main research and implementation specification; proposed methods and hypotheses, not results. |
| `../papers/README.md` | Library inventory and reading order aligned with the specification. |
| `../papers/LOCAL_PAPER_NOTES.md` | Existing content-review notes; useful navigation, not a substitute for checking the original PDF when making a thesis claim. |
| `../papers/LITERATURE_MAP.md` | Screened literature leads with declared access/review limits. |
| `../../high_dimensional_statistics_investment_thesis_research_plan.docx` | Broader research plan, including optimization, robustness, and backtest inference; optional context where it exceeds the focused specification. |
| `../../latent/thesis.md` | Mathematical background on covariance estimation, likelihood, shrinkage and RMT. Treat as working notes needing claim-level review. |
| `../README.md` | Existing ExFa software context. Implemented features must be checked in code before describing them as complete. |

No university thesis class, submission rules, or final citation style has been
identified. The unrelated `astori_tex` and `metodi-edp` projects are not evidence
of this thesis's required template.

## Scientific priorities

- H1: factor selection/detectability; compare selectors holding residual
  modelling fixed.
- H2: diagonal versus structured residual covariance; hold factor selection
  fixed. Whole-covariance shrinkage is a reference, with its role explicit.
- H3: factor-subspace persistence, risk calibration, and forecast stability.
- Primary evidence: walk-forward covariance forecast quasi-likelihood and
  portfolio-risk calibration, with uncertainty and matched information sets.
- Planned initial data: French 100 size/book-to-market and 49 industry
  portfolios; 252/504-day windows. These are proposed choices until confirmed
  by actual data/configuration records, not experiments already performed.
- Simulations check algorithms and illuminate assumptions; empirical financial
  risk forecasting remains the main evaluation.

## Checks to apply while writing

Distinguish population spike thresholds from sample spectral edges, bounded
spikes from pervasive-factor asymptotics, and covariance from correlation PCA.
Do not equate volatility standardization with residual whitening. Check the
conditions for any shrinkage rule applied after estimated-factor removal.
Compare subspaces under a declared normalization; individual factor labels and
factor-wise risk allocations depend on coordinates. Track centering, n versus
n-1 normalization, return units, forecast horizons, and loss argument order.

Avoid leakage: preprocessing, tuning, portfolio construction, and parameter
selection must use information available at forecast time. Trace every reported
figure/table to its data, configuration, code version, and evaluation period.
Never infer empirical conclusions from the research plan.

## Reading and citation workflow

Start with Johnstone–Paul, then Bai–Ng and POET; use loss-specific shrinkage and
correlation-eigenstructure papers where the argument needs them. The library
identifies POET as a missing local core paper and forecast-evaluation literature
as another priority. Recheck availability when processing the relevant section.

No original research PDF has been independently reviewed in this setup session,
and no bibliography entry has been newly verified. When drafting a cited claim,
read the relevant original text, check assumptions and bibliographic metadata,
and record the exact location and verification status in `thesis/notes/editorial.md`.
