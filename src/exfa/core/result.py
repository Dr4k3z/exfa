from dataclasses import dataclass
from datetime import date as Date
from typing import Self

import numpy as np
import pandas as pd

from exfa.regression import CrossSectionalRegressionResult

from .model_diagnostics import compute_regression_diagnostics


@dataclass
class FactorModelResult:
    factor_returns: pd.DataFrame
    fitted_returns: pd.Series
    residuals: pd.Series
    daily_results: dict[Date, CrossSectionalRegressionResult]
    vif_columns: tuple[str, ...]

    @classmethod
    def from_daily_results(
        cls,
        daily_results: dict[Date, CrossSectionalRegressionResult],
        index: pd.Index,
        columns: pd.Index,
        vif_columns: tuple[str, ...],
    ) -> Self:
        factor_returns = pd.DataFrame(
            {date: result.factor_returns for date, result in daily_results.items()}
        ).T

        factor_returns.index.name = "date"
        factor_returns = factor_returns.reindex(columns=columns).sort_index()

        fitted_returns = pd.Series(
            np.nan,
            index=index,
            dtype=float,
            name="fitted_return",
        )

        residuals = pd.Series(
            np.nan,
            index=index,
            dtype=float,
            name="residual",
        )

        for result in daily_results.values():
            fitted_returns.loc[result.fitted_returns.index] = result.fitted_returns
            residuals.loc[result.residuals.index] = result.residuals

        return cls(
            factor_returns=factor_returns,
            fitted_returns=fitted_returns,
            residuals=residuals,
            daily_results=daily_results,
            vif_columns=vif_columns,
        )

    @property
    def diagnostics(self) -> pd.DataFrame:
        records = []

        for date, result in self.daily_results.items():
            diagnostics = compute_regression_diagnostics(
                result,
                continuous_columns=self.vif_columns,
            )

            records.append(
                {
                    "date": date,
                    "n_obs": result.n_obs,
                    "r2": diagnostics.r2,
                    "fitted_return_corr": diagnostics.fitted_return_corr,
                    "constraint_error": diagnostics.constraint_error,
                    "n_dropped": len(result.dropped_assets),
                    "dropped_assets": result.dropped_assets,
                    **{f"vif_{column}": value for column, value in diagnostics.vif.items()},
                }
            )

        if not records:
            return pd.DataFrame(index=pd.Index([], dtype=object, name="date"))

        return pd.DataFrame.from_records(records).set_index("date").sort_index()

    @property
    def r2(self) -> pd.Series:
        return self.diagnostics["r2"]

    @property
    def vif(self) -> pd.DataFrame:
        columns = [column for column in self.diagnostics.columns if column.startswith("vif_")]
        return self.diagnostics[columns]
