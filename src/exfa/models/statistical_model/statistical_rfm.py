from typing import Self, override

import numpy as np

from exfa.models.rfm import RiskFactorModel


class StatisticalFactorModel(RiskFactorModel):
    def __init__(self, num_factors: int) -> None:
        if num_factors < 1:
            raise ValueError("num_factors must be a positive integer")

        self._num_factors = num_factors

        super().__init__()

    @override
    def fit(self, returns: np.ndarray, *, center: bool = True) -> Self:
        """
        Fit PPCA to returns.
        """

        returns = self._validate_returns(returns)
        num_assets, num_times = returns.shape
        num_factors = self._num_factors
        if num_factors >= min(num_assets, num_times):
            raise ValueError("Expected num_factors < min(num_assets, num_times)")

        # Preprocessing
        returns = returns - returns.mean(axis=1, keepdims=True) if center else returns.copy()

        # Singular value decomposition
        left_vectors, singular_values, _ = np.linalg.svd(returns, full_matrices=False)
        eigenvalues = singular_values**2 / num_times

        # Residual variance
        residual_variance = max(
            (np.sum(eigenvalues) - np.sum(eigenvalues[:num_factors])) / (num_assets - num_factors),
            0.0,
        )

        # Factor loadings
        loadings = left_vectors[:, :num_factors] * np.sqrt(
            np.maximum(eigenvalues[:num_factors] - residual_variance, 0.0)
        )

        # Covariance estimate
        covariance = loadings @ loadings.T + residual_variance * np.eye(num_assets)

        factor_returns = np.linalg.lstsq(loadings, returns, rcond=None)[0]
        residuals = returns - loadings @ factor_returns

        # Commit fitted state
        self._factor_returns = factor_returns
        self._residuals = residuals
        self._returns = returns
        self._loadings = loadings
        self._residual_variance = residual_variance
        self._covariance = covariance

        return self
