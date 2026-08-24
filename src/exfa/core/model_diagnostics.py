from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from exfa.regression.cross_sectional import CrossSectionalRegressionResult


@dataclass(frozen=True)
class RegressionDiagnostics:
    r2: float
    fitted_return_corr: float
    constraint_error: float
    vif: pd.Series


def compute_regression_diagnostics(
    result: CrossSectionalRegressionResult,
    *,
    continuous_columns: tuple[str, ...],
) -> RegressionDiagnostics:
    y = result.fitted_returns + result.residuals
    residual_values = result.residuals.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    fitted_values = result.fitted_returns.to_numpy(dtype=float)

    ss_total = float(np.sum((y_values - y_values.mean()) ** 2))
    r2 = np.nan if ss_total <= 0 else 1.0 - float(np.sum(residual_values**2)) / ss_total
    fitted_return_corr = (
        np.nan
        if np.std(y_values) == 0 or np.std(fitted_values) == 0
        else float(np.corrcoef(fitted_values, y_values)[0, 1])
    )

    if result.constraint is None:
        constraint_error = 0.0
    else:
        active_beta = result.factor_returns.reindex(result.exposures.columns)
        violations = (
            result.constraint.matrix @ active_beta.to_numpy(dtype=float) - result.constraint.target
        )
        constraint_error = float(np.max(np.abs(violations)))

    return RegressionDiagnostics(
        r2=r2,
        fitted_return_corr=fitted_return_corr,
        constraint_error=constraint_error,
        vif=compute_vif(result.exposures, continuous_columns),
    )


def compute_vif(exposures: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    columns = tuple(column for column in columns if column in exposures.columns)
    if not columns:
        return pd.Series(dtype=float, name="vif")

    values = exposures.loc[:, list(columns)].dropna()
    if len(values) <= len(columns):
        return pd.Series(np.nan, index=columns, dtype=float, name="vif")
    if len(columns) == 1:
        return pd.Series(1.0, index=columns, dtype=float, name="vif")

    matrix = values.to_numpy(dtype=float)
    vif: dict[str, float] = {}
    for i, column in enumerate(columns):
        try:
            value = variance_inflation_factor(matrix, i)
        except ValueError, np.linalg.LinAlgError:
            value = np.nan
        vif[column] = float(value)

    return pd.Series(vif, dtype=float, name="vif")
