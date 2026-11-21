"""Monte Carlo experiment: PPCA eigenvalue inflation versus spiked-RMT limits."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

num_factors = 10
num_assets = 500
num_trials = 50
T_values = (2500, 1000, 500, 250)
residual_sigma = 1.0
sigma2 = residual_sigma**2

rng = np.random.default_rng(42)

# Fixed random factor directions, but controlled factor variances 1^2, ..., 10^2.
Q, _ = np.linalg.qr(rng.standard_normal((num_assets, num_factors)))
factor_variances = np.arange(num_factors, 0, -1, dtype=float) ** 2
B = Q * np.sqrt(factor_variances)

# Exact population eigenvalues: factor variances + isotropic noise variance.
population_spikes = factor_variances + sigma2


def top_sample_eigenvalues(R: np.ndarray, m: int) -> np.ndarray:
    n, T = R.shape
    # R R^T and R^T R have the same nonzero eigenvalues.
    gram = (R @ R.T if n <= T else R.T @ R) / T
    dim = gram.shape[0]
    return eigh(
        gram,
        subset_by_index=(dim - m, dim - 1),
        eigvals_only=True,
        check_finite=False,
    )[::-1]


def bbp_limit(spikes: np.ndarray, gamma: float, noise_variance: float) -> np.ndarray:
    edge = noise_variance * (1 + np.sqrt(gamma)) ** 2
    threshold = noise_variance * (1 + np.sqrt(gamma))
    result = np.full_like(spikes, edge, dtype=float)
    mask = spikes > threshold
    result[mask] = spikes[mask] * (
        1 + gamma * noise_variance / (spikes[mask] - noise_variance)
    )
    return result


results = {}
for T in T_values:
    sample_spikes = np.empty((num_trials, num_factors))
    ppca_factors = np.empty_like(sample_spikes)
    estimated_sigma2 = np.empty(num_trials)

    for trial in range(num_trials):
        F = rng.standard_normal((num_factors, T))
        E = residual_sigma * rng.standard_normal((num_assets, T))
        R = B @ F + E
        eigs = top_sample_eigenvalues(R, num_factors)

        # PPCA ML estimate of isotropic noise variance; zeros from rank deficiency
        # are automatically included via trace, even when T < n.
        sigma2_hat = (np.sum(R * R) / T - eigs.sum()) / (num_assets - num_factors)
        sample_spikes[trial] = eigs
        estimated_sigma2[trial] = sigma2_hat
        ppca_factors[trial] = eigs - sigma2_hat

    results[T] = {
        "sample": sample_spikes.mean(axis=0),
        "sample_se": sample_spikes.std(axis=0, ddof=1) / np.sqrt(num_trials),
        "ppca": ppca_factors.mean(axis=0),
        "noise": estimated_sigma2.mean(),
    }

# Figure 1: same comparison as Paleologo's Figure 7.2, for small and large n/T.
fig, ax = plt.subplots(figsize=(9, 6))
for T in (2500, 250):
    gamma = num_assets / T
    a = results[T]
    ax.plot(factor_variances, a["sample"], "o", label=f"Sample PCA (n/T={gamma:g})")
    ax.plot(factor_variances, a["ppca"], "^", label=f"PPCA factor estimate (n/T={gamma:g})")
maximum = max(factor_variances.max(), max(v["sample"].max() for v in results.values())) * 1.05
ax.plot([0, maximum], [0, maximum], linestyle="--", label="Unbiased reference")
ax.set(
    xlabel="Population factor variance",
    ylabel="Estimated variance",
    title="PPCA factor variance: sample vs population",
    xlim=(0, maximum),
    ylim=(0, maximum),
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("/mnt/data/ppca_factor_variance_comparison.png", dpi=160)
plt.close(fig)

# Figure 2: predicted versus empirical sample eigenvalue inflation.
fig, ax = plt.subplots(figsize=(9, 6))
for T in T_values:
    gamma = num_assets / T
    mean_sample = results[T]["sample"]
    observed_inflation = mean_sample - population_spikes
    theory_inflation = bbp_limit(population_spikes, gamma, sigma2) - population_spikes
    (line,) = ax.plot(
        population_spikes, observed_inflation, "o", label=f"Monte Carlo (n/T={gamma:g})"
    )
    ax.plot(population_spikes, theory_inflation, linestyle=":", color=line.get_color())
ax.axhline(0, linestyle="--", linewidth=1)
ax.set(
    xlabel="Population covariance eigenvalue",
    ylabel="Sample eigenvalue minus population eigenvalue",
    title="Sample eigenvalue inflation: Monte Carlo (points) vs RMT (dotted)",
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig("/mnt/data/ppca_rmt_eigenvalue_inflation.png", dpi=160)
plt.close(fig)

print(f"n={num_assets}, m={num_factors}, repetitions={num_trials}, noise variance={sigma2}")
for T in T_values:
    a = results[T]
    print(
        f"T={T:4d}  n/T={num_assets / T:4.2f}  avg noise var={a['noise']:.4f}  max spike sample={a['sample'][0]:.3f} (pop={population_spikes[0]:.1f})  weakest spike sample={a['sample'][-1]:.3f} (pop={population_spikes[-1]:.1f})"
    )
