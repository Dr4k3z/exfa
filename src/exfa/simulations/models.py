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
        self._distribution = distribution

        self._mean, self._covariance = distribution.moments(num_assets)

    @property
    def distribution(self) -> Distribution:
        return self._distribution

    @property
    def num_assets(self) -> int:
        return self._num_assets

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

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

        B.flags.writeable = False
        self._B = B
        self._sigma2 = noise_variance

        self._factors_distribution = (
            factors_distribution
            if factors_distribution is not None
            else GaussianDistribution()
        )
        self._residuals_distribution = (
            residuals_distribution
            if residuals_distribution is not None
            else GaussianDistribution()
        )

        n, m = self._B.shape

        mu_f, Sigma_f = self.factors_distribution.moments(m)
        mu_e, Sigma_e = self.residuals_distribution.moments(n)

        sigma = np.sqrt(self.sigma2)

        self._mean = self._B @ mu_f + sigma * mu_e
        self._covariance = self._B @ Sigma_f @ self._B.T + self.sigma2 * Sigma_e

    @property
    def B(self) -> np.ndarray:
        loadings = self._B.copy()
        loadings.flags.writeable = False
        return loadings

    @property
    def sigma2(self) -> float:
        return self._sigma2

    @property
    def factors_distribution(self) -> Distribution:
        return self._factors_distribution

    @property
    def residuals_distribution(self) -> Distribution:
        return self._residuals_distribution

    @property
    def num_assets(self) -> int:
        return self._B.shape[0]

    @property
    def num_factors(self) -> int:
        return self._B.shape[1]

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        if T < 1:
            raise ValueError("T must be positive")

        n, m = self._B.shape

        F = self.factors_distribution.sample((m, T), rng)
        E = self.residuals_distribution.sample((n, T), rng)

        return self._B @ F + np.sqrt(self.sigma2) * E
