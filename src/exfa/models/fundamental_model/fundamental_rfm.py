from collections.abc import Sequence
from typing import Self, override

import numpy as np

from exfa.models.rfm import RiskFactorModel, fitted_property

from .regression import CrossSectionalRegression, CrossSectionalRegressionResult


class FundamentalFactorModel(RiskFactorModel):
    """
    Estimate factor returns from supplied asset loadings using NumPy WLS.

    Factor construction is a separate preprocessing step. All array axes retain
    their input order; no sorting or implicit asset alignment takes place.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loadings_history: np.ndarray | None = None
        self._daily_results: tuple[CrossSectionalRegressionResult, ...] | None = None

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

    @override
    def fit(
        self,
        returns: np.ndarray,
        loadings: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        categorical_groups: Sequence[Sequence[int]] = (),
        min_obs: int = 20,
        ridge: float | None = None,
    ) -> Self:
        """
        Fit returns (assets, times), with static or time-varying loadings.

        Loadings have shape (assets, factors) or (assets, factors, times).
        Weights have shape (assets,) or (assets, times) and must be positive.
        Categorical groups contain factor column indices constrained to sum to
        zero. Covariance uses the last time's loadings, sample factor covariance,
        and diagonal sample residual variances (ddof=1). Requires two times and
        finite complete panels. A failed fit preserves the previous fitted state.
        """

        returns = self._validate_returns(returns)
        num_assets, num_times = returns.shape
        loadings = np.asarray(loadings, dtype=float)
        if loadings.ndim == 2:
            loadings = np.repeat(loadings[:, :, None], num_times, axis=2)
        if (
            loadings.ndim != 3
            or loadings.shape[0] != num_assets
            or loadings.shape[2] != num_times
            or loadings.shape[1] < 1
        ):
            raise ValueError(
                "loadings must have shape (assets, factors) or (assets, factors, times)"
            )
        if not np.isfinite(loadings).all():
            raise ValueError("loadings must contain only finite values")
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape == (num_assets,):
                weights = np.repeat(weights[:, None], num_times, axis=1)
            if (
                weights.shape != returns.shape
                or not np.isfinite(weights).all()
                or np.any(weights <= 0)
            ):
                raise ValueError(
                    "weights must be finite, positive, and shaped (assets,) or (assets, times)"
                )
        regression = CrossSectionalRegression(min_obs=min_obs, ridge=ridge)
        daily_results = tuple(
            regression.fit(
                returns=returns[:, time],
                loadings=loadings[:, :, time],
                categorical_groups=categorical_groups,
                weights=weights[:, time] if weights is not None else None,
            )
            for time in range(num_times)
        )
        factor_returns = np.column_stack([result.factor_returns for result in daily_results])
        residuals = np.column_stack([result.residuals for result in daily_results])
        latest_loadings = loadings[:, :, -1].copy()
        residual_variance = np.var(residuals, axis=1, ddof=1)
        factor_covariance = np.atleast_2d(np.cov(factor_returns, ddof=1))
        covariance = latest_loadings @ factor_covariance @ latest_loadings.T + np.diag(
            residual_variance
        )
        self._returns = returns.copy()
        self._loadings = latest_loadings
        self._loadings_history = loadings.copy()
        self._factor_returns = factor_returns
        self._residuals = residuals
        self._residual_variance = residual_variance
        self._covariance = covariance
        self._daily_results = daily_results
        return self
