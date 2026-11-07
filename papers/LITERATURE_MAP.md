# Additional literature and resources

Research date: 2026-10-31. Scope: the statistical factor risk engine in the repository specification. This is a targeted, broad literature search, not a systematic review or an exhaustive account of the latest work.

**Evidence status:** entries below were screened using original-paper abstracts, publisher records, author-hosted material, or institutional records. They have not received the same content review as the five local PDFs, and their proofs have not been audited. Journal years are used where verified; otherwise an explicit preprint year is given. Additional PDFs have not been downloaded. Each title links to a source from which to continue reading.

Priority: **A** = supports the core thesis; **B** = strong extension candidate; **C** = background or later exploration. These are reading priorities, not a recommendation to implement every method.

## Factor selection and strength

### R01 · A · Dobriban & Owen (2019)

[Deterministic parallel analysis: An improved method for selecting factors and principal components](https://arxiv.org/abs/1711.04155). JRSS B 81(1), 163–183.

Makes parallel analysis deterministic; studies strong-factor shadowing, deflation, and modified selection thresholds. Useful alternative to a single-scale MP edge. Read the precise variance-profile and sampling assumptions before financial adaptation. [Authors' implementation](https://github.com/dobriban/DPA) is a useful reference, subject to a code/license review before reuse. Connects directly to H1 and the dominant-market-factor experiment.

### R02 · A · Fan, Guo & Zheng (2022)

[Estimating Number of Factors by Adjusted Eigenvalues Thresholding](https://www.tandfonline.com/doi/abs/10.1080/01621459.2020.1825448) · [Preprint](https://arxiv.org/abs/1909.10710). JASA 117(538), 852–861.

Adjusted correlation thresholding (ACT) addresses heterogeneous scales and eigenvalue bias. Especially close to the thesis's covariance/correlation decision. Its population factor-count characterization is conditional on model assumptions; it is not permission to count raw sample correlation eigenvalues above one. Worth reading before inventing a new noise-scale correction.

### R03 · B · Ahn & Horenstein (2013)

[Eigenvalue Ratio Test for the Number of Factors](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2942796). Econometrica 81(3), 1203–1227; DOI: 10.3982/ECTA8968.

Adjacent-eigenvalue ratios yield simple factor-number estimators in approximate-factor models. A useful sensitivity check if Bai–Ng and MP disagree. Assumes a distinction between diverging factor eigenvalues and bounded remaining eigenvalues. The publisher DOI was inaccessible in this browsing session; the authors' working-paper abstract was available. Check the zero-factor convention and search bounds before implementation.

### R04 · B · Bai & Ng (2017 preprint; later revisions)

[Principal Components and Regularized Estimation of Factor Models](https://arxiv.org/abs/1708.08137).

Connects ridge-style factor estimation with singular-value shrinkage and rank regularization. Helps bridge the supplied matrix-denoising paper and an approximate-factor application. Read for the distinction between estimating the common component and selecting dimension. Do not assume its MSE improvements imply future risk-score improvements.

### R05 · B · Choi & Yuan (2024 preprint)

[High Dimensional Factor Analysis with Weak Factors](https://arxiv.org/abs/2402.05789).

Studies loadings scaling with p^alpha for alpha between zero and one, under specified noise dependence restrictions. A useful intermediate regime between bounded spikes and pervasive factors. Use for a strength-scaling simulation and the theory chapter; check the current version's conditions rather than claiming unrestricted weak-factor consistency.

## Residual covariance and robust factor models

### R06 · A · Fan, Liao & Mincheva (2013)

[Large Covariance Estimation by Thresholding Principal Orthogonal Complements](https://arxiv.org/abs/1201.0175) · [Journal record](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12016). JRSS B 75(4), 603–680.

**Highest-priority missing local paper.** POET combines a low-rank factor component with thresholded residual covariance. It directly supports H2 and preserves the factor-risk interpretation. Requires pervasive factors and conditional sparsity under additional regularity conditions. Read the threshold, tuning, positive-definiteness treatment and dependence assumptions before coding a method called POET.

### R07 · A · Cai & Liu (2011)

[Adaptive Thresholding for Sparse Covariance Matrix Estimation](https://arxiv.org/abs/1102.2237).

Adapts thresholds to the variability of individual covariance entries. Explains why one universal residual threshold can be inappropriate. Useful implementation background for POET-style regularization. Its results for observed samples must not be transplanted to estimated factor residuals without accounting for factor estimation.

### R08 · B · Fan, Liu & Wang (2018)

[Large Covariance Estimation through Elliptical Factor Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC6133289/) · [2015 preprint](https://arxiv.org/abs/1507.08377).

Generalizes the POET framework using robust pilot estimates, including Kendall-based methods for elliptical data. Directly relevant if a few extreme days dominate factor extraction. Scatter and covariance require a scale identification; the covariance-estimation construction still has moment conditions. Robustness does not remove all tail or dependence assumptions.

### R09 · B · Zhang, Cai & Wu (2018 preprint; revised 2021)

[Heteroskedastic PCA: Algorithm, Optimality, and Applications](https://arxiv.org/abs/1810.08316).

HeteroPCA iteratively imputes the covariance diagonal to reduce bias from unequal noise variances. Promising alternative to treating total-volatility standardization as residual whitening. Primarily a subspace method: a complete positive-definite covariance and risk decomposition still need specification.

### R10 · B · Zhou & Chen (2025)

[Deflated HeteroPCA: Overcoming the curse of ill-conditioning in heteroskedastic PCA](https://arxiv.org/abs/2303.06198). Annals of Statistics 53(1), 91–116; DOI: 10.1214/24-AOS2456.

Uses spectral blocks and successive deflation to improve heterogeneous-noise subspace estimation when signal strengths are badly conditioned. Relevant to one dominant market component plus weaker components. A recent extension to R09, not a reason to add both algorithms to the initial pipeline.

## Spectral shrinkage and the factor-versus-full-spectrum question

### R11 · A · Ledoit & Wolf (2004)

[A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices](https://www.ledoit.net/ole1_abstract.htm).

Linear shrinkage provides a simple conditioning and sanity-check reference. Its quadratic-loss optimality does not identify latent factors or validate conditional risk forecasts. Keep this baseline small and clearly motivated.

### R12 · A · Ledoit & Wolf (2012)

[Nonlinear shrinkage estimation of large-dimensional covariance matrices](https://arxiv.org/abs/1207.5322).

General-spectrum nonlinear eigenvalue shrinkage provides the main external reference to a structured factor model. Distinct from fixed-rank spike shrinkage in the supplied Donoho–Gavish–Johnstone paper. Start with whole-covariance estimation; projected factor residuals do not automatically satisfy the same model.

### R13 · A · Ledoit & Wolf (2020)

[Analytical nonlinear shrinkage of large-dimensional covariance matrices](https://www.ledoit.net/Analytical_AoS_2020.pdf) · [Author publication index](https://www.econ.uzh.ch/en/people/researchers/ledoit/publications.html). Annals of Statistics 48, 3043–3065.

An analytical approach using spectral density and its Hilbert transform, attractive for a reproducible reference implementation. Verify normalization, bandwidth, dimension/sample-size regime and singular-case handling. The author PDF was search-indexed, but a subsequent direct open returned HTTP 502; bibliographic details were also checked on the university index.

### R14 · A · Bun, Bouchaud & Potters (2017)

[Cleaning large correlation matrices: tools from random matrix theory](https://arxiv.org/abs/1610.08104). Physics Reports 666, 1–109.

Broad RIE review covering resolvents, eigenvector overlaps, shrinkage and financial examples. The best additional synthesis linking RMT mathematics to covariance cleaning. Read selectively around the estimator actually used; empirical success in the review is not evidence for the thesis's future experiment.

### R15 · B · Bartz (2016 preprint)

[Cross-validation based Nonlinear Shrinkage](https://arxiv.org/abs/1611.00798).

An alternative route to estimating covariance along sample directions through cross-validation. Inspires a sample-splitting study that distinguishes directional fitting from amplitude estimation. Financial validation must respect time order; adapting the splitting scheme changes the setting and needs separate assessment.

## Dynamics and portfolio application

### R16 · B · Pafka, Potters & Kondor (2004 preprint)

[Exponential Weighting and Random-Matrix-Theory-Based Filtering of Financial Covariance Matrices for Portfolio Optimization](https://arxiv.org/abs/cond-mat/0402573).

Derives a noise spectrum for exponential weighting and combines it with filtering. Relevant to the optional EWSCM milestone. Replacing n with an effective sample size in the ordinary MP edge is not equivalent to using the appropriate weighted spectral law.

### R17 · B · Engle, Ledoit & Wolf (2019)

[Large Dynamic Covariance Matrices](https://www.ledoit.net/Engle_GARCH_2019.pdf). Journal of Business & Economic Statistics 37(2), 363–375.

Combines dynamic conditional correlation with nonlinear shrinkage. Helps distinguish stale volatility forecasts from poor cross-sectional factor estimation. Full DCC would expand the initial scope; a shared, simple causal volatility layer is an easier first mechanism experiment.

### R18 · A · De Nard, Ledoit & Wolf (2021)

[Factor Models for Portfolio Selection in Large Dimensions: The Good, the Better and the Ugly](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4368040). Journal of Financial Econometrics 19, 236–257. [University working-paper version](https://www.econstor.eu/bitstream/10419/192899/1/econwp290.pdf).

Especially close prior work: combines factor structure with dynamic residual covariance and compares portfolio consequences. Essential context before claiming novelty for factor-plus-shrinkage combinations. The thesis can instead focus on recovery, decomposition, calibration and controlled selection ablations.

## Evaluation

### R19 · A · Patton (2011)

[Volatility Forecast Comparison using Imperfect Volatility Proxies — author's research page and paper links](https://public.econ.duke.edu/~ap172/research.html). Journal of Econometrics 160(1), 246–256.

Explains forecast-loss comparisons with imperfect variance proxies. Important justification for using suitable losses with noisy squared returns. Verify the conditional-unbiasedness/information-set conditions. A calibration ratio near one alone cannot establish good forecast timing.

### R20 · A · Diebold & Mariano (1995)

[Comparing Predictive Accuracy](https://doi.org/10.1080/07350015.1995.10524599). Journal of Business & Economic Statistics 13(3), 253–263.

Foundational reference for paired forecast-loss comparisons allowing serial dependence. Use aligned loss differentials and an appropriate long-run variance estimate; rolling fitting, overlap, tuning and many comparisons need additional care. A block-bootstrap interval is also useful but requires a separately specified resampling design.

## RMT foundations and historical finance context

### R21 · A · Baik & Silverstein (2004 preprint)

[Eigenvalues of Large Sample Covariance Matrices of Spiked Population Models](https://arxiv.org/abs/math/0408165).

Almost-sure limits for sample eigenvalues in a spiked model under a general class of samples. A direct source for the sample/population spike mapping. Particularly useful alongside the original complex-Wishart transition result when making claims about real return data.

### R22 · B · Baik, Ben Arous & Péché (2005)

[Phase transition of the largest eigenvalue for non-null complex sample covariance matrices](https://arxiv.org/abs/math/0403022).

Original BBP transition reference. Its complex-data setting and critical fluctuation laws must be distinguished from later real-data results. Read for theorem provenance rather than reproducing all the machinery in a master's thesis.

### R23 · B · Wang & Fan (2017)

[Asymptotics of empirical eigenstructure for high dimensional spiked covariance](https://pmc.ncbi.nlm.nih.gov/articles/PMC5563862/). Annals of Statistics 45(3), 1342–1374.

Studies eigenstructure with diverging signals. Helps bridge fixed-strength spike theory and factor-model asymptotics without treating the regimes as interchangeable. Relevant when explaining why dominant pervasive factors may be estimated more accurately than BBP intuition alone suggests.

### R24 · C · Laloux, Cizeau, Bouchaud & Potters (1999)

[Noise Dressing of Financial Correlation Matrices](https://arxiv.org/abs/cond-mat/9810255).

Historical motivation for applying RMT to financial correlation noise. Useful in the introduction. Agreement of a bulk histogram with an RMT law does not establish the absence of useful weak structure inside the bulk.

### R25 · B · Plerou et al. (2002)

[A Random Matrix Approach to Cross-Correlations in Financial Data](https://arxiv.org/abs/cond-mat/0108023).

Studies financial spectra, sector structure and temporal stability of outlying eigenvectors. Useful precedent for persistence diagnostics and economic interpretation. Treat its empirical findings as historical evidence, not a universal property of future samples.

## Data resource

### D01 · A · Kenneth R. French Data Library

[Official library](https://mba.tuck.dartmouth.edu/pages/Faculty/ken.french/data_library.html).

Use the daily 100 size/book-to-market portfolios and 49 industries identified in the specification. Save a frozen download with source URL, retrieval date, checksum, units and selected panel. The library explicitly notes historical reconstruction/revisions and missing-value sentinels. A present-day historical file supports a reproducible retrospective walk-forward study; it does not reconstruct the exact historical data vintage available on each forecast date.

## What to read next

The shortest useful expansion is **R06 POET → R02 ACT → R01 DPA → R19 Patton → R18 factor models and dynamic residuals**. These fill methodological gaps around the supplied papers. Read R09/R10 if unequal idiosyncratic noise appears decisive; R16/R17 if stale volatility is the main failure; R05/R23 if weak-factor asymptotics become the main theoretical contribution.
