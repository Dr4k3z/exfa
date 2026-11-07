from typing import Self

import numpy as np


class RiskFactorModel:
    def __init__(self, num_factors: int) -> None:
        if num_factors < 1:
            raise ValueError("num_factors must be positive")

        self._num_factors = num_factors

        self._returns: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._loadings: np.ndarray | None = None
        self._residual_variance: float | None = None

    @property
    def num_factors(self) -> int:
        return self._num_factors

    @property
    def num_assets(self) -> int:
        return self.returns.shape[0]

    @property
    def num_times(self) -> int:
        return self.returns.shape[1]

    @property
    def returns(self) -> np.ndarray:
        if self._returns is None:
            raise RuntimeError("Model has not been fitted")
        return self._returns

    @property
    def covariance(self) -> np.ndarray:
        if self._covariance is None:
            raise RuntimeError("Model has not been fitted")
        return self._covariance

    @property
    def loadings(self) -> np.ndarray:
        if self._loadings is None:
            raise RuntimeError("Model has not been fitted")
        return self._loadings

    @property
    def residual_variance(self) -> float:
        if self._residual_variance is None:
            raise RuntimeError("Model has not been fitted")
        return self._residual_variance

    def fit(self, R: np.ndarray, center: bool = True) -> Self:
        R = np.asarray(R, dtype=float)

        if R.ndim != 2:
            raise ValueError("R must be a two-dimensional array")

        n, T = R.shape
        m = self._num_factors

        if not 0 < m < min(n, T):
            raise ValueError("Expected 0 < num_factors < min(n, T)")

        # Preprocessing
        returns = R - R.mean(axis=1, keepdims=True) if center else R.copy()

        # Singular value decomposition
        U, s, _ = np.linalg.svd(returns, full_matrices=False)
        eigenvalues = s**2 / T

        # Residual variance
        residual_variance = max(
            (np.sum(eigenvalues) - np.sum(eigenvalues[:m])) / (n - m),
            0.0,
        )

        # Factor loadings
        loadings = U[:, :m] * np.sqrt(np.maximum(eigenvalues[:m] - residual_variance, 0.0))

        # Covariance estimate
        covariance = loadings @ loadings.T + residual_variance * np.eye(n)

        # Commit fitted state
        self._returns = returns
        self._loadings = loadings
        self._residual_variance = residual_variance
        self._covariance = covariance

        return self
