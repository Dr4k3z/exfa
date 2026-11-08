from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from exfa.models.fundamental_model.regression import CrossSectionalRegressionResult
from exfa.models.fundamental_model.regression.wls import LinearConstraint


@dataclass(frozen=True)
class DailyRegressionReport:
    """
    Per-time regression metrics in input order, with VIF shaped (columns, times).

    Undefined metrics are NaN. Dropped assets are positional input row indices.
    continuous_columns identifies the factor rows of vif; labels remain external.
    Observation counts are derived from num_assets and dropped_assets.
    """

    r2: np.ndarray
    fitted_return_corr: np.ndarray
    constraint_error: np.ndarray
    num_assets: int
    dropped_assets: tuple[tuple[int, ...], ...]
    vif: np.ndarray
    continuous_columns: tuple[int, ...]

    @property
    def n_dropped(self) -> np.ndarray:
        return np.array([len(assets) for assets in self.dropped_assets], dtype=int)

    @property
    def n_obs(self) -> np.ndarray:
        return self.num_assets - self.n_dropped


class FittedRegressionModel(Protocol):
    """
    Canonical fitted state needed by the daily diagnostic adapter.
    """

    @property
    def returns(self) -> np.ndarray: ...

    @property
    def daily_results(self) -> tuple[CrossSectionalRegressionResult, ...]: ...


def compute_vif(loadings: np.ndarray) -> np.ndarray:
    """
    Compute uncentered VIF for explicitly selected continuous loading columns.

    Match the former regression pipeline's through-origin auxiliary regressions.
    Constant columns and insufficient observations produce NaN. Collinear
    nonconstant columns produce infinity; a single varying column has VIF one.
    Do not pass intercepts or complete categorical indicator groups.
    """

    loadings = np.asarray(loadings, dtype=float)
    if loadings.ndim != 2 or not np.isfinite(loadings).all():
        raise ValueError("loadings must be a finite (assets, columns) array")
    num_assets, num_columns = loadings.shape
    vif = np.full(num_columns, np.nan)
    if num_assets <= num_columns:
        return vif
    for column in range(num_columns):
        values = loadings[:, column]
        if np.ptp(values) == 0:
            continue
        if num_columns == 1:
            vif[column] = 1.0
            continue
        other = np.delete(loadings, column, axis=1)
        coefficients = np.linalg.lstsq(other, values, rcond=None)[0]
        squared_error = float(np.sum((values - other @ coefficients) ** 2))
        total = float(values @ values)
        tolerance = np.finfo(float).eps * max(loadings.shape) * total
        vif[column] = np.inf if squared_error <= tolerance else total / squared_error
    return vif


def compute_daily_regression_report(
    returns: np.ndarray,
    residuals: np.ndarray,
    loadings_history: np.ndarray,
    factor_returns: np.ndarray,
    *,
    continuous_columns: Sequence[int] = (),
    constraints: Sequence[LinearConstraint | None] = (),
) -> DailyRegressionReport:
    """
    Evaluate stored regressions without refitting or using latest-only loadings.

    returns/residuals have shape (assets, times), loadings_history is
    (assets, factors, times), and factor_returns is (factors, times).
    Every metric uses the same per-time rows with finite returns, residuals,
    and complete loadings. R2 uses cross-sectionally centered returns and is
    NaN when their total variation is zero. Correlation is NaN for constant
    actual or fitted returns. Missing rows are counted and retained by position.

    Select continuous factor indices explicitly for historical, uncentered VIF;
    the default omits VIF rather than guessing intercept/categorical metadata.
    Supply one constraint per time to evaluate C @ factor_returns - target;
    absent constraints have error zero. The report does not change fitting or
    covariance missing-data policy. Dates and factor names remain caller-owned.
    """

    returns = np.asarray(returns, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    loadings_history = np.asarray(loadings_history, dtype=float)
    factor_returns = np.asarray(factor_returns, dtype=float)
    if returns.ndim != 2 or residuals.shape != returns.shape:
        raise ValueError("returns and residuals must have matching (assets, times) shapes")
    num_assets, num_times = returns.shape
    if (
        loadings_history.ndim != 3
        or loadings_history.shape[0] != num_assets
        or loadings_history.shape[2] != num_times
    ):
        raise ValueError("loadings_history must have shape (assets, factors, times)")
    num_factors = loadings_history.shape[1]
    if num_factors == 0 or factor_returns.shape != (num_factors, num_times):
        raise ValueError(
            "factor_returns must have shape (factors, times) with at least one factor"
        )
    columns = tuple(continuous_columns)
    if any(
        isinstance(column, bool)
        or not isinstance(column, (int, np.integer))
        or column < 0
        or column >= num_factors
        for column in columns
    ) or len(set(columns)) != len(columns):
        raise ValueError("continuous_columns must contain distinct valid factor indices")
    if constraints and len(constraints) != num_times:
        raise ValueError("constraints must be empty or contain one entry per time")
    r2 = np.full(num_times, np.nan)
    correlations = np.full(num_times, np.nan)
    constraint_errors = np.zeros(num_times)
    dropped_assets = []
    vif = np.full((len(columns), num_times), np.nan)
    for time in range(num_times):
        loadings = loadings_history[:, :, time]
        valid = (
            np.isfinite(returns[:, time])
            & np.isfinite(residuals[:, time])
            & np.isfinite(loadings).all(axis=1)
        )
        dropped_assets.append(tuple(int(asset) for asset in np.flatnonzero(~valid)))
        actual = returns[valid, time]
        errors = residuals[valid, time]
        fitted = actual - errors
        if actual.size:
            total = float(np.sum((actual - actual.mean()) ** 2))
            if total > 0:
                r2[time] = 1.0 - float(errors @ errors) / total
            if actual.size > 1 and np.std(actual) > 0 and np.std(fitted) > 0:
                correlations[time] = float(np.corrcoef(actual, fitted)[0, 1])
        vif[:, time] = compute_vif(loadings[valid][:, columns])
        constraint = constraints[time] if constraints else None
        if constraint is not None:
            if (
                constraint.matrix.shape[1] != num_factors
                or not np.isfinite(constraint.matrix).all()
                or not np.isfinite(constraint.target).all()
            ):
                raise ValueError("constraint must be finite and match the factor dimension")
            violations = constraint.matrix @ factor_returns[:, time] - constraint.target
            constraint_errors[time] = (
                float(np.max(np.abs(violations))) if violations.size else 0.0
            )
    return DailyRegressionReport(
        r2=r2,
        fitted_return_corr=correlations,
        constraint_error=constraint_errors,
        num_assets=num_assets,
        dropped_assets=tuple(dropped_assets),
        vif=vif,
        continuous_columns=columns,
    )


def compute_model_regression_report(
    model: FittedRegressionModel,
    *,
    continuous_columns: Sequence[int] = (),
) -> DailyRegressionReport:
    """
    Build a daily report from a fitted fundamental model's daily results.

    This adapter preserves fitted-property pre-fit guards and uses the saved
    constraints and historical loadings. It does not refit the model.
    """

    returns = model.returns
    daily_results = model.daily_results
    if len(daily_results) != returns.shape[1]:
        raise ValueError("daily_results must contain one result per return time")
    return compute_daily_regression_report(
        returns=returns,
        residuals=np.column_stack([result.residuals for result in daily_results]),
        loadings_history=np.stack([result.loadings for result in daily_results], axis=2),
        factor_returns=np.column_stack([result.factor_returns for result in daily_results]),
        continuous_columns=continuous_columns,
        constraints=tuple(result.constraint for result in daily_results),
    )
