from typing import Protocol

import numpy as np

from .distributions import Distribution, GaussianDistribution


class Model(Protocol):
    @property
    def num_assets(self) -> int: ...

    @property
    def mean(self) -> np.ndarray: ...

    @property
    def covariance(self) -> np.ndarray: ...

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray: ...


class IIDModel:
    def __init__(
        self,
        num_assets: int,
        distribution: Distribution,
    ) -> None:
        if num_assets < 1:
            raise ValueError("num_assets must be positive")

        self._num_assets = num_assets
        self.distribution = distribution

        self._mean, self._covariance = distribution.moments(num_assets)

    @property
    def num_assets(self) -> int:
        return self._num_assets

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        if T < 1:
            raise ValueError("T must be positive")

        return self.distribution.sample((self.num_assets, T), rng)


class FactorModel:
    def __init__(
        self,
        B: np.ndarray,
        noise_variance: float = 1.0,
        factors_distribution: Distribution | None = None,
        residuals_distribution: Distribution | None = None,
    ) -> None:
        B = np.array(B, dtype=float, copy=True)

        if B.ndim != 2 or min(B.shape) < 1 or not np.isfinite(B).all():
            raise ValueError("Invalid loading matrix")

        if not np.isfinite(noise_variance) or noise_variance < 0:
            raise ValueError("noise_variance must be nonnegative")

        self.B = B
        self.sigma2 = noise_variance

        self.factors_distribution = (
            factors_distribution
            if factors_distribution is not None
            else GaussianDistribution()
        )
        self.residuals_distribution = (
            residuals_distribution
            if residuals_distribution is not None
            else GaussianDistribution()
        )

        n, m = self.B.shape

        mu_f, Sigma_f = self.factors_distribution.moments(m)
        mu_e, Sigma_e = self.residuals_distribution.moments(n)

        sigma = np.sqrt(self.sigma2)

        self._mean = self.B @ mu_f + sigma * mu_e
        self._covariance = self.B @ Sigma_f @ self.B.T + self.sigma2 * Sigma_e

    @property
    def num_assets(self) -> int:
        return self.B.shape[0]

    @property
    def num_factors(self) -> int:
        return self.B.shape[1]

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        if T < 1:
            raise ValueError("T must be positive")

        n, m = self.B.shape

        F = self.factors_distribution.sample((m, T), rng)
        E = self.residuals_distribution.sample((n, T), rng)

        return self.B @ F + np.sqrt(self.sigma2) * E
