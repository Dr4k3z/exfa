from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .wls import ConstrainedWLS, LinearConstraint


class RegressionFitError(ValueError):
    """
    A cross-section has insufficient usable observations.
    """


@dataclass(frozen=True)
class CrossSectionalRegressionResult:
    factor_returns: np.ndarray
    residuals: np.ndarray
    loadings: np.ndarray
    constraint: LinearConstraint | None
    dropped_assets: tuple[int, ...]

    @property
    def fitted_returns(self) -> np.ndarray:
        fitted_returns = self.loadings @ self.factor_returns
        fitted_returns[list(self.dropped_assets)] = np.nan
        return fitted_returns

    @property
    def n_obs(self) -> int:
        return self.loadings.shape[0] - len(self.dropped_assets)


class CrossSectionalRegression:
    """
    NumPy cross-sectional WLS with optional sum-to-zero factor groups.
    """

    def __init__(self, min_obs: int = 20, ridge: float | None = None) -> None:
        if (
            isinstance(min_obs, bool)
            or not isinstance(min_obs, (int, np.integer))
            or min_obs < 1
        ):
            raise ValueError("min_obs must be a positive integer")
        if ridge is not None and (not np.isfinite(ridge) or ridge < 0):
            raise ValueError("ridge must be finite and non-negative")
        self.min_obs = min_obs
        self.ridge = ridge

    @staticmethod
    def build_linear_constraint(
        num_factors: int,
        categorical_groups: Sequence[Sequence[int]],
    ) -> LinearConstraint | None:
        rows = []
        for group in categorical_groups:
            indices = tuple(group)
            if not indices or len(set(indices)) != len(indices):
                raise ValueError("Categorical groups must contain distinct factor indices")
            if any(
                isinstance(i, bool)
                or not isinstance(i, (int, np.integer))
                or i < 0
                or i >= num_factors
                for i in indices
            ):
                raise ValueError("Categorical group index is out of bounds")
            row = np.zeros(num_factors)
            row[list(indices)] = 1.0
            rows.append(row)
        return LinearConstraint(np.stack(rows), np.zeros(len(rows))) if rows else None

    def fit(
        self,
        returns: np.ndarray,
        loadings: np.ndarray,
        categorical_groups: Sequence[Sequence[int]] = (),
        weights: np.ndarray | None = None,
    ) -> CrossSectionalRegressionResult:
        returns = np.asarray(returns, dtype=float)
        loadings = np.asarray(loadings, dtype=float)
        if (
            returns.ndim != 1
            or loadings.ndim != 2
            or loadings.shape[0] != returns.size
            or loadings.shape[1] == 0
        ):
            raise ValueError("Expected returns (assets,) and loadings (assets, factors)")
        constraint = self.build_linear_constraint(loadings.shape[1], categorical_groups)
        valid = np.isfinite(returns) & np.isfinite(loadings).all(axis=1)
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != returns.shape:
                raise ValueError("weights must have shape (assets,)")
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("weights must be finite and positive")
        if np.count_nonzero(valid) < self.min_obs:
            raise RegressionFitError("Minimum number of observations not met")
        result = ConstrainedWLS(ridge=self.ridge).fit(
            y=returns[valid],
            x=loadings[valid],
            weights=weights[valid] if weights is not None else None,
            constraint=constraint,
        )
        residuals = np.full(returns.shape, np.nan)
        residuals[valid] = result.residuals
        return CrossSectionalRegressionResult(
            result.coefficients,
            residuals,
            loadings.copy(),
            constraint,
            tuple(int(i) for i in np.flatnonzero(~valid)),
        )
