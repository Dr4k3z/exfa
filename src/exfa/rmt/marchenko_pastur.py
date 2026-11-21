import numpy as np


def mp_edges(gamma: float, sigma2: float = 1.0) -> tuple[float, float]:
    """
    Endpoints of the continuous Marchenko--Pastur support.
    """

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be finite and positive")
    return (
        sigma2 * (1 - np.sqrt(gamma)) ** 2,
        sigma2 * (1 + np.sqrt(gamma)) ** 2,
    )


def mp_density(x: np.ndarray, gamma: float, sigma2: float = 1.0) -> np.ndarray:
    """
    Continuous MP density (excluding the zero atom when gamma > 1).
    """

    x = np.asarray(x, dtype=float)
    lo, hi = mp_edges(gamma, sigma2)
    density = np.zeros_like(x)
    mask = (x > 0) & (x > lo) & (x < hi)
    density[mask] = np.sqrt((hi - x[mask]) * (x[mask] - lo)) / (
        2 * np.pi * gamma * sigma2 * x[mask]
    )
    return density


def quarter_circle_density(s: np.ndarray, sigma2: float = 1.0) -> np.ndarray:
    """
    Limiting singular-value density of X/sqrt(T) for n/T -> 1.
    """

    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be finite and positive")
    s = np.asarray(s, dtype=float)
    density = np.zeros_like(s)
    mask = (s >= 0) & (s <= 2 * np.sqrt(sigma2))
    density[mask] = np.sqrt(4 * sigma2 - s[mask] ** 2) / (np.pi * sigma2)
    return density
