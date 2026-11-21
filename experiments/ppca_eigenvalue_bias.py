"""
Compare PCA and probabilistic PCA in the high-dimensional setting of Paleologo's Figure 7.2.
"""

from functools import partial
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from exfa.rmt.utils import top_sample_eigenvalues
from exfa.simulations import FactorModel, MCEngine

if TYPE_CHECKING or __package__:
    from .plotting import LATEX_STYLE, save_figure
else:
    from plotting import LATEX_STYLE, save_figure

NUM_ASSETS = (1000, 3000)
NUM_OBSERVATIONS = 250
NUM_TRIALS = 50
FACTOR_VOLATILITIES = np.arange(10, 0, -1, dtype=float)
NOISE_VARIANCE = 1.0


def measure_eigenvalues(returns: np.ndarray, *, num_factors: int) -> dict[str, np.ndarray]:
    """
    Estimate PCA and PPCA factor variances for a known zero mean.
    """

    num_assets, num_observations = returns.shape
    sample_eigenvalues = top_sample_eigenvalues(returns, num_factors)
    covariance_trace = np.sum(returns**2) / num_observations
    # Include the zero eigenvalues when the sample covariance is rank deficient.
    noise_variance = (covariance_trace - sample_eigenvalues.sum()) / (num_assets - num_factors)

    return {
        "PCA": sample_eigenvalues,
        "PPCA": sample_eigenvalues - noise_variance,
    }


def simulate_eigenvalues(num_assets: int) -> dict[str, np.ndarray]:
    """
    Draw Gaussian factors and residuals for one asset universe.
    """

    rng = np.random.default_rng(123)
    directions, _ = np.linalg.qr(rng.standard_normal((num_assets, FACTOR_VOLATILITIES.size)))
    model = FactorModel(B=directions * FACTOR_VOLATILITIES, noise_variance=NOISE_VARIANCE)
    engine = MCEngine(model=model, num_trials=NUM_TRIALS, seed=42)
    measure = partial(measure_eigenvalues, num_factors=model.num_factors)
    return engine.run(NUM_OBSERVATIONS, measure)


def plot_eigenvalue_bias(
    estimates: dict[int, dict[str, np.ndarray]], population_variances: np.ndarray
) -> None:
    """
    Plot the mean estimates against population values and show their signed bias.
    """

    with plt.rc_context(LATEX_STYLE):
        comparison, comparison_axes = plt.subplots(
            1, len(estimates), figsize=(9, 4.5), sharex=True, sharey=True, squeeze=False
        )
        bias, bias_axes = plt.subplots(
            1, len(estimates), figsize=(9, 4.2), sharex=True, sharey=True, squeeze=False
        )
        upper = 1.05 * max(
            population_variances.max(),
            *(
                values.mean(axis=0).max()
                for methods in estimates.values()
                for values in methods.values()
            ),
        )

        for (num_assets, methods), comparison_ax, bias_ax in zip(
            estimates.items(), comparison_axes[0], bias_axes[0], strict=True
        ):
            title = f"$n={num_assets}$, $n/T={num_assets / NUM_OBSERVATIONS:g}$"
            comparison_ax.set(title=title, xlabel="Population factor variance")
            comparison_ax.plot(
                [0, upper], [0, upper], color="0.5", linestyle="--", linewidth=1
            )
            comparison_ax.set(xlim=(0, upper), ylim=(0, upper), aspect="equal")

            bias_ax.set(title=title, xlabel="Population factor variance")
            bias_ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
            for (method, values), marker in zip(methods.items(), ("o", "^"), strict=True):
                mean = values.mean(axis=0)
                signed_bias = mean - population_variances
                standard_error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
                comparison_ax.plot(
                    population_variances, mean, marker, markerfacecolor="none", label=method
                )
                bias_ax.errorbar(
                    population_variances,
                    signed_bias,
                    yerr=standard_error,
                    fmt=marker,
                    capsize=2,
                    label=method,
                )
                print(
                    f"n={num_assets}, {method}: average bias={signed_bias.mean():.3f}, "
                    f"weakest factor bias={signed_bias[-1]:.3f}"
                )

        comparison_axes[0, 0].set_ylabel("Mean estimated factor variance")
        bias_axes[0, 0].set_ylabel("Mean estimate minus population variance")
        comparison_axes[0, 0].legend()
        bias_axes[0, 0].legend()
        comparison.suptitle("PCA and PPCA factor variance estimates")
        bias.suptitle("Factor variance bias (error bars show one Monte Carlo SE)")
        save_figure(comparison, "ppca_eigenvalue_comparison.png")
        save_figure(bias, "ppca_eigenvalue_bias.png")


def run_experiment() -> None:
    """
    Simulate both asset universes and save the comparison figures.
    """

    estimates = {num_assets: simulate_eigenvalues(num_assets) for num_assets in NUM_ASSETS}
    plot_eigenvalue_bias(estimates, FACTOR_VOLATILITIES**2)


if __name__ == "__main__":
    run_experiment()
