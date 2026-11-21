# Numerical experiments

Run these commands from the repository root.

You can also run the scripts directly, for example
`uv run experiments/eigenvalue_spreading.py`.

| Experiment | Command | Purpose and outputs |
| --- | --- | --- |
| [Eigenvalue spreading](eigenvalue_spreading.py) | `uv run -m experiments.eigenvalue_spreading` | Uses Gaussian Monte Carlo simulations to reproduce the spectral spreading of identity and heterogeneous covariance matrices in Johnstone and Paul's Figure 2. Recreates the Marchenko–Pastur comparisons in Figure 3 and adds a quarter-circle comparison for singular values at aspect ratio one. Prints spectral summaries and writes `johnstone_paul_figure2_spreading.png` and `johnstone_paul_figure3_mp_quarter_circle.png` in this directory. |
| [PPCA eigenvalue bias](ppca_eigenvalue_bias.py) | `uv run -m experiments.ppca_eigenvalue_bias` | Compares PCA and PPCA factor variance estimates at asset-to-observation ratios of 4 and 12. Prints bias summaries and writes [estimated versus population variances](ppca_eigenvalue_comparison.png) and [signed bias with Monte Carlo standard errors](ppca_eigenvalue_bias.png). |

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
