from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as Date

import numpy as np
import pandas as pd

from .wls import ConstrainedWLS, LinearConstraint


class RegressionFitError(ValueError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"CrossSectionalRegression could not be fitted: {msg}")


@dataclass(frozen=True)
class CrossSectionalRegressionResult:
    date: Date
    factor_returns: pd.Series
    fitted_returns: pd.Series
    residuals: pd.Series
    exposures: pd.DataFrame
    constraint: LinearConstraint | None
    dropped_assets: tuple[str, ...]

    @property
    def n_obs(self) -> int:
        return len(self.exposures)


class CrossSectionalRegression:
    """
    Constrained weighted regression for one factor-model cross-section.
    """

    def __init__(
        self,
        min_obs: int = 20,
        ridge: float | None = None,
    ) -> None:
        if min_obs <= 0:
            raise ValueError("`min_obs` must be positive.")

        if ridge is not None and ridge < 0:
            raise ValueError("`ridge` must be non-negative.")

        self.min_obs = min_obs
        self.ridge = ridge

    @staticmethod
    def build_linear_constraint(
        columns: pd.Index,
        categorical_groups: Sequence[tuple[str, ...]],
    ) -> LinearConstraint | None:
        positions = {column: i for i, column in enumerate(columns)}

        constraint_rows: list[np.ndarray] = []

        for group in categorical_groups:
            active = [column for column in group if column in positions]

            if not active:
                continue

            row = np.zeros(
                len(columns),
                dtype=np.float64,
            )

            for column in active:
                row[positions[column]] = 1.0

            constraint_rows.append(row)

        if not constraint_rows:
            return None

        matrix = np.vstack(constraint_rows)
        target = np.zeros(
            len(constraint_rows),
            dtype=np.float64,
        )

        return LinearConstraint(
            matrix=matrix,
            target=target,
        )

    def fit(
        self,
        date: Date,
        exposures: pd.DataFrame,
        returns: pd.Series,
        categorical_groups: Sequence[tuple[str, ...]],
        weights: pd.Series | None = None,
    ) -> CrossSectionalRegressionResult:
        if not exposures.index.equals(returns.index):
            returns = returns.reindex(exposures.index)

        if weights is not None and not exposures.index.equals(weights.index):
            weights = weights.reindex(exposures.index)

        all_columns = exposures.columns.copy()

        valid = (
            returns.notna()
            & np.isfinite(returns)
            & exposures.notna().all(axis=1)
            & np.isfinite(exposures).all(axis=1)
        )

        if weights is not None:
            valid &= weights.notna() & np.isfinite(weights) & (weights > 0)

        dropped_index = exposures.index[~valid]

        if isinstance(dropped_index, pd.MultiIndex):
            if "asset" in dropped_index.names:
                dropped_assets = tuple(dropped_index.get_level_values("asset").astype(str).tolist())
            else:
                dropped_assets = tuple(map(str, dropped_index.tolist()))
        else:
            dropped_assets = tuple(dropped_index.astype(str).tolist())

        exposures = exposures.loc[valid]
        returns = returns.loc[valid]

        if weights is not None:
            weights = weights.loc[valid]

        n_obs = len(exposures)

        if n_obs < self.min_obs:
            raise RegressionFitError(
                f"Minimum number of observations not met: minimum is {self.min_obs}, got {n_obs}."
            )

        active_columns = exposures.columns[exposures.ne(0).any(axis=0)]

        active_exposures = exposures.loc[:, active_columns]

        if active_exposures.empty:
            raise RegressionFitError("No valid active exposures.")

        constraint = self.build_linear_constraint(
            columns=active_columns, categorical_groups=categorical_groups
        )

        regression = ConstrainedWLS(
            ridge=self.ridge,
        ).fit(
            y=returns.to_numpy(dtype=np.float64),
            x=active_exposures.to_numpy(dtype=np.float64),
            weights=(weights.to_numpy(dtype=np.float64) if weights is not None else None),
            constraint=constraint,
        )

        coefficients = pd.Series(
            regression.coefficients,
            index=active_columns,
            dtype=float,
            name="factor_return",
        )

        factor_returns = coefficients.reindex(all_columns)
        factor_returns.name = date

        fitted_returns = pd.Series(
            regression.fitted_values,
            index=active_exposures.index,
            dtype=float,
            name="fitted_return",
        )

        residuals = pd.Series(
            regression.residuals,
            index=active_exposures.index,
            dtype=float,
            name="residual",
        )

        return CrossSectionalRegressionResult(
            date=date,
            factor_returns=factor_returns,
            fitted_returns=fitted_returns,
            residuals=residuals,
            exposures=active_exposures,
            constraint=constraint,
            dropped_assets=dropped_assets,
        )
