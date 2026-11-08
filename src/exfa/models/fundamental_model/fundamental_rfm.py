from collections.abc import Mapping, Sequence
from typing import Self, override

import numpy as np

from exfa.models.fundamental_model.factors import Factor
from exfa.models.fundamental_model.factors.base import CategoricalFactor, ContinuousFactor
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
        self._factors = tuple(factors)
        self._daily_results: tuple[CrossSectionalRegressionResult, ...] | None = None

        super().__init__()

    @property
    def factors(self) -> tuple[Factor, ...]:
        return self._factors

    @property
    def categorical_factors(self) -> tuple[CategoricalFactor, ...]:
        return tuple(factor for factor in self.factors if isinstance(factor, CategoricalFactor))

    @property
    def continuous_factors(self) -> tuple[ContinuousFactor, ...]:
        return tuple(factor for factor in self.factors if isinstance(factor, ContinuousFactor))

    @fitted_property
    def loadings_history(self) -> np.ndarray | None:
        """
        Loadings with shape (assets, factors, times).
        """

        if self._daily_results is None:
            return None
        return np.stack([result.loadings for result in self.daily_results], axis=2)

    @fitted_property
    def daily_results(self) -> tuple[CrossSectionalRegressionResult, ...] | None:
        """
        Cross-sectional results in input time order.
        """

        return self._daily_results

    @fitted_property
    @override
    def loadings(self) -> np.ndarray:
        return self.daily_results[-1].loadings

    @fitted_property
    @override
    def factor_returns(self) -> np.ndarray:
        return np.column_stack([result.factor_returns for result in self.daily_results])

    @fitted_property
    @override
    def residuals(self) -> np.ndarray:
        return np.column_stack([result.residuals for result in self.daily_results])

    def add_factor(self, factor: Factor) -> Self:
        """
        Append a factor and invalidate all fitted results; fit again before use.

        The factors tuple is immutable, but Factor objects remain caller-owned.
        After modifying a Factor object directly, callers must fit again.
        """

        self._factors += (factor,)
        RiskFactorModel.__init__(self)
        self._daily_results = None
        return self

    @staticmethod
    def _validate_weights(weights: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        if weights.shape == (shape[0],):
            weights = np.broadcast_to(weights[:, None], shape)
        if weights.shape != shape or not np.isfinite(weights).all() or np.any(weights <= 0):
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

        if weights is not None:
            weights = self._validate_weights(weights, returns.shape)

        for factor in self.factors:
            factor.validate_data(market_data, returns.shape)

        if not self.factors:
            raise ValueError("At least one factor is required")

        blocks = []
        categorical_groups = []
        names = []
        offset = 0
        for factor in self.factors:
            block = factor.compute(market_data, returns.shape)

            indices = tuple(range(offset, offset + block.shape[1]))
            if factor.requires_constraint:
                categorical_groups.append(indices)
            if factor.name in names:
                raise ValueError(f"Duplicate factor name: {factor.name!r}")
            names.append(factor.name)
            offset += block.shape[1]
            blocks.append(block)

        loadings = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

        if not np.isfinite(loadings).all():
            raise ValueError("loadings must contain only finite values across the complete panel")

        regression = CrossSectionalRegression(min_obs=min_obs, ridge=ridge)
        daily_results = tuple(
            regression.fit(
                returns=returns[:, time],
                loadings=loadings[:, :, time],
                categorical_groups=categorical_groups,
                weights=weights[:, time] if weights is not None else None,
            )
            for time in range(returns.shape[1])
        )
        self._returns = returns
        self._daily_results = daily_results
        return self
