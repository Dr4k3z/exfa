"""
Reproduce Johnstone--Paul spectral-spreading and MP figures.

Run from the project root with uv run -m experiments.eigenvalue_spreading.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from exfa import MCEngine
from exfa.rmt import mp_density, mp_edges, quarter_circle_density
from exfa.rmt.utils import sample_eigenvalues
from exfa.simulations.distributions import Distribution, GaussianDistribution
from exfa.simulations.models import IIDModel

if TYPE_CHECKING or __package__:
    from .plotting import LATEX_STYLE, save_figure
else:
    # Direct script execution puts this directory on the import path.
    from plotting import LATEX_STYLE, save_figure

SEED = 20261120


def simulate_spectra(
    population_eigenvalues: np.ndarray,
    T: int,
    num_trials: int,
    innovations: Distribution | None = None,
) -> np.ndarray:
    """
    Return ordered sample spectra with shape (num_trials, num_assets).

    `innovations` must be standardized, with mean zero and identity covariance.
    Keeping trial axes allows later Monte Carlo bias and variance analysis.
    """

    population = np.asarray(population_eigenvalues, dtype=float)
    if (
        population.ndim != 1
        or population.size == 0
        or not np.isfinite(population).all()
        or np.any(population <= 0)
    ):
        raise ValueError("Population eigenvalues must be finite and positive")

    distribution = (
        innovations if innovations is not None else GaussianDistribution()
    ).affine_transformation(covariance=np.diag(population))

    model = IIDModel(num_assets=population.size, distribution=distribution)
    mc = MCEngine(model=model, num_trials=num_trials, seed=SEED)
    return mc.run(T, lambda R: {"eigenvalues": sample_eigenvalues(R)})["eigenvalues"]


def fit_mp_moments(eigenvalues: np.ndarray) -> tuple[float, float]:
    """
    Fit MP moments for a white population.
    """

    sigma2 = float(np.mean(eigenvalues))
    if sigma2 <= 0:
        raise ValueError("Eigenvalues must have a positive mean")
    gamma = float(np.var(eigenvalues, ddof=0) / sigma2**2)
    if gamma <= 0:
        raise ValueError("Spectrum must have positive variance")
    return gamma, sigma2


def plot_figure_2() -> None:
    with plt.rc_context(LATEX_STYLE):
        p, T, num_trials = 100, 200, 80
        gamma = p / T
        cases = (
            (r"White population ($\Sigma = I$)", np.ones(p)),
            (
                r"Heterogeneous population ($\mathrm{eig}(\Sigma) \in [5, 25]$)",
                np.linspace(5, 25, p),
            ),
        )

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7))
        for ax, (title, population) in zip(axes, cases, strict=True):
            spectra = simulate_spectra(population, T=T, num_trials=num_trials)
            eigenvalues = spectra.ravel()  # Flatten only for the pooled histogram.

            ax.hist(
                eigenvalues,
                bins=55,
                density=True,
                alpha=0.65,
                label="Sample eigenvalues",
            )
            if np.all(population == population[0]):
                ax.axvline(
                    population[0],
                    linewidth=2.3,
                    label="Population eigenvalues (all 1)",
                )
                lo, hi = mp_edges(gamma)
                x = np.linspace(lo + 1e-6, hi - 1e-6, 500)
                ax.plot(x, mp_density(x, gamma), linewidth=2, label="Marchenko-Pastur law")
            else:
                ax.hist(
                    population,
                    bins=20,
                    density=True,
                    histtype="step",
                    linewidth=2,
                    label="Population eigenvalues",
                )

            ax.set_title(title)
            ax.set_xlabel("Eigenvalue")
            ax.set_ylabel("Spectral density")
            ax.legend(fontsize=9)
            print(
                f"Fig 2 {title}: sample min={eigenvalues.min():.4f}, "
                f"max={eigenvalues.max():.4f}, mean={eigenvalues.mean():.4f}"
            )

        fig.suptitle(
            rf"Figure 2. Eigenvalue spreading ($p={p}$, $T={T}$, "
            rf"$\gamma={gamma:.2f}$, {num_trials} simulations)",
            fontsize=13,
        )
        save_figure(fig, "johnstone_paul_figure2_spreading.png")


def plot_figure_3() -> None:
    with plt.rc_context(LATEX_STYLE):
        p, num_trials = 240, 24
        gammas = (0.25, 1.0)
        spectra_by_gamma: dict[float, np.ndarray] = {}
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))

        for gamma, ax in zip(gammas, axes[:2], strict=True):
            T = round(p / gamma)
            spectra = simulate_spectra(np.ones(p), T=T, num_trials=num_trials)
            spectra_by_gamma[gamma] = spectra
            eigenvalues = spectra.ravel()

            fitted_gamma, fitted_sigma2 = fit_mp_moments(eigenvalues)
            lo, hi = mp_edges(gamma)
            lo_fit, hi_fit = mp_edges(fitted_gamma, fitted_sigma2)

            ax.hist(
                eigenvalues,
                bins=65,
                density=True,
                alpha=0.65,
                label="Simulated covariance eigenvalues",
            )
            # Avoid the integrable MP singularity at zero for gamma=1.
            x = np.linspace(max(lo + 0.005, 0.01), hi, 550)
            ax.plot(
                x,
                mp_density(x, gamma),
                linewidth=2.1,
                label=rf"MP theory ($\gamma={gamma:g}$, $\sigma^2=1$)",
            )
            x_fit = np.linspace(max(lo_fit + 0.005, 0.01), hi_fit, 550)
            ax.plot(
                x_fit,
                mp_density(x_fit, fitted_gamma, fitted_sigma2),
                linestyle="--",
                linewidth=1.6,
                label=rf"Moment fit ($\hat{{\gamma}}={fitted_gamma:.3f}$, $\hat{{\sigma}}^2={fitted_sigma2:.3f}$)",
            )
            ax.set_xlim(0, hi + 0.15)
            ax.set_title(f"MP eigenvalue law ($p={p}$, $T={T}$)")
            ax.set_xlabel(r"Covariance eigenvalue $\lambda$")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            print(
                f"Fig 3 gamma={gamma:.2f}: theoretical support=[{lo:.4f}, {hi:.4f}], "
                f"moment fit gamma={fitted_gamma:.4f}, sigma2={fitted_sigma2:.4f}"
            )

        singular_values = np.sqrt(spectra_by_gamma[1.0]).ravel()
        ax = axes[2]
        ax.hist(
            singular_values,
            bins=65,
            density=True,
            alpha=0.65,
            label="Simulated singular values",
        )
        s = np.linspace(0, 2, 500)
        ax.plot(s, quarter_circle_density(s), linewidth=2.1, label="Quarter-circle theory")
        ax.set_xlim(0, 2.15)
        ax.set_title(r"Quarter-circle law ($\gamma=1$)")
        ax.set_xlabel(r"Singular value $s = \sqrt{\lambda}$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Figure 3. Eigenvalue and singular-value laws ({num_trials} simulations each)",
            fontsize=13,
        )
        save_figure(fig, "johnstone_paul_figure3_mp_quarter_circle.png")


if __name__ == "__main__":
    plot_figure_2()
    plot_figure_3()
