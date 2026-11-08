from collections.abc import Mapping, Sequence
from typing import Self, override

import numpy as np

from exfa.models.fundamental_model.factors import Factor
from exfa.models.fundamental_model.regression import (
    CrossSectionalRegression,
    CrossSectionalRegressionResult,
)
from exfa.models.rfm import RiskFactorModel, fitted_property


class FundamentalFactorModel(RiskFactorModel):
    """
    Compute configured Factors from market features and estimate returns using WLS.

    Array axes retain their input order; no sorting or implicit asset alignment takes place.
    """

    def __init__(self, factors: Sequence[Factor] = ()) -> None:
        self._factors = list(factors)
        self._continuous_columns: tuple[int, ...] | None = None
        self._loadings_history: np.ndarray | None = None
        self._daily_results: tuple[CrossSectionalRegressionResult, ...] | None = None

        super().__init__()

    @property
    def factors(self) -> list[Factor]:
        return self._factors

    @fitted_property
    def loadings_history(self) -> np.ndarray | None:
        """
        Loadings with shape (assets, factors, times).
        """

        return self._loadings_history

    @fitted_property
    def daily_results(self) -> tuple[CrossSectionalRegressionResult, ...] | None:
        """
        Cross-sectional results in input time order.
        """

        return self._daily_results

    @fitted_property
    def continuous_columns(self) -> tuple[int, ...] | None:
        return self._continuous_columns

    def add_factor(self, factor: Factor) -> Self:
        self._factors.append(factor)
        return self

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        if weights.shape == (self.num_assets,):
            weights = np.broadcast_to(weights[:, None], self.returns.shape)
        if (
            weights.shape != self.returns.shape
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError(
                "weights must be finite, positive, and shaped (assets,) or (assets, times)"
            )
        return weights

    @override
    def _fit(
        self,
        returns: np.ndarray,
        market_data: Mapping[str, np.ndarray],
        weights: np.ndarray | None = None,
        ridge: float | None = None,
        min_obs: int = 20,
    ) -> Self:
        """
        Validate returns and market features, compute Factors, then fit WLS.

        Returns are (assets, times). Features are (assets,) for static data or
        (assets, times), optionally with leading history for rolling factors.
        Trailing feature columns align with returns; callers own axis ordering.
        Momentum requires window leading price columns. No dates are silently
        removed. Weights are (assets,) or (assets, times), finite and positive.
        Failed validation or fitting leaves previously fitted state unchanged.
        """

        for factor in self.factors:
            factor.validate_data(market_data, returns.shape)

        blocks = []
        categorical_groups = []
        continuous_columns = []
        names = []
        offset = 0
        for factor in self.factors:
            block = factor.compute(market_data, returns.shape)

            indices = tuple(range(offset, offset + block.shape[1]))
            if factor.requires_constraint:
                categorical_groups.append(indices)
            if factor.include_in_vif:
                continuous_columns.extend(indices)
            if factor.name in names:
                raise ValueError(f"Duplicate factor name: {factor.name!r}")
            names.append(factor.name)
            offset += block.shape[1]
            blocks.append(block)

        loadings = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

        regression = CrossSectionalRegression(min_obs=min_obs, ridge=ridge)
        daily_results = tuple(
            regression.fit(
                returns=returns[:, time],
                loadings=loadings[:, :, time],
                categorical_groups=categorical_groups,
                weights=weights[:, time] if weights is not None else None,
            )
            for time in range(self.num_times)
        )
        factor_returns = np.column_stack([result.factor_returns for result in daily_results])
        residuals = np.column_stack([result.residuals for result in daily_results])
        latest_loadings = loadings[:, :, -1]
        residual_variance = np.var(residuals, axis=1, ddof=1)
        factor_covariance = np.atleast_2d(np.cov(factor_returns, ddof=1))
        covariance = latest_loadings @ factor_covariance @ latest_loadings.T + np.diag(
            residual_variance
        )

        self._loadings = latest_loadings
        self._loadings_history = loadings
        self._factor_returns = factor_returns
        self._residuals = residuals
        self._residual_variance = residual_variance
        self._covariance = covariance
        self._daily_results = daily_results
        self._continuous_columns = tuple(continuous_columns)
        return self
