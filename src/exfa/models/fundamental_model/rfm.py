from collections.abc import Sequence
from datetime import date as Date
from typing import Self

import pandas as pd

from exfa.factors.base import ContinuousFactor, Factor
from exfa.regression import CrossSectionalRegression, RegressionFitError

from ...core.result import FactorModelResult


class RiskFactorModel:
    def __init__(self, factors: Sequence[Factor]) -> None:
        self.factors = list(factors)

    def add_factor(self, factor: Factor) -> Self:
        self.factors.append(factor)
        return self

    def fit(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        weight_col: str | None = None,
        min_obs: int = 20,
        ridge: float | None = None,
    ) -> FactorModelResult:
        self._validate_fit_inputs(data, returns, weight_col)

        exposures, categorical_groups, vif_columns = self._compute_exposures(data)

        weights = data[weight_col].reindex(exposures.index) if weight_col is not None else None

        regression = CrossSectionalRegression(
            min_obs=min_obs,
            ridge=ridge,
        )

        daily_results = {}

        for date, daily_exposures in exposures.groupby(
            level="date",
            sort=True,
        ):
            if not isinstance(date, Date):
                raise TypeError("Exposure index level 'date' must contain datetime.date values.")

            try:
                result = regression.fit(
                    date=date,
                    exposures=daily_exposures,
                    returns=returns.reindex(daily_exposures.index),
                    categorical_groups=categorical_groups,
                    weights=(
                        weights.reindex(daily_exposures.index) if weights is not None else None
                    ),
                )
            except RegressionFitError:
                continue

            daily_results[result.date] = result

        return FactorModelResult.from_daily_results(
            daily_results=daily_results,
            index=exposures.index,
            columns=exposures.columns,
            vif_columns=vif_columns,
        )

    def _compute_exposures(
        self,
        data: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        tuple[tuple[str, ...], ...],
        tuple[str, ...],
    ]:
        if not self.factors:
            raise ValueError("RiskFactorModel must contain at least one factor.")

        factor_exposures = []
        categorical_groups = []
        vif_columns = []

        for factor in self.factors:
            exposure = factor.compute(data)
            columns = tuple(exposure.columns)

            factor_exposures.append(exposure)

            if factor.requires_constraint:
                categorical_groups.append(columns)

            if isinstance(factor, ContinuousFactor):
                vif_columns.extend(columns)

        exposures = pd.concat(
            factor_exposures,
            axis=1,
        )

        duplicated = exposures.columns[exposures.columns.duplicated()].unique()

        if len(duplicated):
            raise ValueError(f"Duplicate exposure columns: {duplicated.tolist()}")

        return (
            exposures,
            tuple(categorical_groups),
            tuple(vif_columns),
        )

    @staticmethod
    def _validate_fit_inputs(
        data: pd.DataFrame,
        returns: pd.Series,
        weight_col: str | None,
    ) -> None:
        RiskFactorModel._validate_index(data, "Data")
        RiskFactorModel._validate_index(returns, "Returns")

        if weight_col is not None and weight_col not in data.columns:
            raise ValueError(f"Weight column {weight_col!r} not found in data.")

    @staticmethod
    def _validate_index(
        obj: pd.DataFrame | pd.Series,
        name: str,
    ) -> None:
        if not isinstance(obj.index, pd.MultiIndex):
            raise TypeError(f"{name} must use a MultiIndex.")

        missing_levels = {"date", "asset"} - set(obj.index.names)

        if missing_levels:
            raise ValueError(f"{name} index is missing levels: {sorted(missing_levels)}")

        if not obj.index.is_unique:
            raise ValueError(f"{name} index must be unique.")

        dates = obj.index.get_level_values("date")

        if not all(isinstance(value, Date) for value in dates):
            raise TypeError(f"{name} index level 'date' must contain datetime.date values.")
