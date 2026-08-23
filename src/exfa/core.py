from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from exfa.factors.base import CategoricalFactor, ContinuousFactor, Factor


@dataclass(frozen=True)
class FactorExposureGroup:
    factor: Factor
    columns: tuple[str, ...]


@dataclass(frozen=True)
class FactorExposureMatrix:
    exposures: pd.DataFrame
    groups: tuple[FactorExposureGroup, ...]


@dataclass(frozen=True)
class DailyRegressionResult:
    date: pd.Timestamp
    factor_returns: pd.Series
    fitted_returns: pd.Series
    residuals: pd.Series
    r2: float
    fitted_return_corr: float
    vif: pd.Series
    n_obs: int
    constraint_error: float
    dropped_assets: tuple[str, ...]


@dataclass(frozen=True)
class FactorModelResult:
    factor_returns: pd.DataFrame
    fitted_returns: pd.Series
    residuals: pd.Series
    diagnostics: pd.DataFrame


class RiskFactorModel:
    def __init__(
        self,
        factors: Sequence[Factor],
    ) -> None:
        self.factors = list(factors)

    def add_factor(self, factor: Factor) -> Self:
        self.factors.append(factor)
        return self

    def compute_exposures(
        self,
        data: pd.DataFrame,
    ) -> FactorExposureMatrix:
        exposure_frames: list[pd.DataFrame] = []
        groups: list[FactorExposureGroup] = []

        for factor in self.factors:
            exposure = factor.compute(data)

            if isinstance(exposure, pd.Series):
                frame = exposure.to_frame()
            elif isinstance(exposure, pd.DataFrame):
                frame = exposure
            else:
                raise TypeError(
                    f"{type(factor).__name__}.compute() must return a pandas Series or DataFrame."
                )

            if not frame.index.equals(data.index):
                raise ValueError(
                    f"Factor {factor.name!r} returned exposures "
                    "with an index different from the input data."
                )

            exposure_frames.append(frame)

            groups.append(
                FactorExposureGroup(
                    factor=factor,
                    columns=tuple(frame.columns),
                )
            )

        if not exposure_frames:
            raise ValueError("FactorModel must contain at least one factor.")

        exposures = pd.concat(
            exposure_frames,
            axis=1,
        )

        if exposures.columns.duplicated().any():
            duplicated = exposures.columns[exposures.columns.duplicated()].unique().tolist()

            raise ValueError(f"Duplicate exposure columns: {duplicated}")

        return FactorExposureMatrix(
            exposures=exposures,
            groups=tuple(groups),
        )

    def fit(
        self,
        data: pd.DataFrame,
        returns: pd.Series,
        *,
        weight_col: str | None = None,
        min_obs: int = 20,
        ridge: float | None = None,
    ) -> FactorModelResult:
        """
        Estimate daily cross-sectional factor returns.

        For each date, solve

            min_beta (y - X beta)' W (y - X beta)

        subject to one sum-to-zero constraint for every categorical
        factor family.

        Parameters
        ----------
        data
            Input panel indexed by ('date', 'asset').

        returns
            Realised returns indexed by ('date', 'asset').

        weight_col
            Optional column in `data` containing regression weights.

        min_obs
            Minimum number of valid assets required to run a daily
            cross-sectional regression.

        ridge
            Optional ridge penalty added to X'WX. No penalty is used
            when None.
        """
        self._validate_fit_inputs(
            data=data,
            returns=returns,
            weight_col=weight_col,
            min_obs=min_obs,
            ridge=ridge,
        )

        factor_exposures = self.compute_exposures(data)
        exposures = factor_exposures.exposures

        if weight_col is not None:
            weights = data[weight_col].reindex(exposures.index)
        else:
            weights = None

        exposure_dates = exposures.index.get_level_values("date").unique()

        return_dates = returns.index.get_level_values("date").unique()

        # Do not attempt dates for which no returns exist at all.
        dates = exposure_dates.intersection(return_dates)

        daily_results: list[DailyRegressionResult] = []

        for date in dates:
            regression_date = pd.Timestamp(date)

            if not isinstance(regression_date, pd.Timestamp):
                continue

            X = exposures.xs(
                date,
                level="date",
                drop_level=False,
            )

            # reindex rather than .loc:
            # individual securities may have missing returns.
            y = returns.reindex(X.index)

            w = weights.reindex(X.index) if weights is not None else None

            result = self._fit_cross_section(
                date=regression_date,
                X=X,
                y=y,
                groups=factor_exposures.groups,
                weights=w,
                min_obs=min_obs,
                ridge=ridge,
            )

            if result is not None:
                daily_results.append(result)

        return self._combine_results(
            results=daily_results,
            exposure_index=exposures.index,
            factor_columns=exposures.columns,
        )

    def _fit_cross_section(
        self,
        *,
        date: pd.Timestamp,
        X: pd.DataFrame,
        y: pd.Series,
        groups: tuple[FactorExposureGroup, ...],
        weights: pd.Series | None,
        min_obs: int,
        ridge: float | None,
    ) -> DailyRegressionResult | None:
        """
        Estimate one cross-sectional factor regression.
        """
        all_columns = X.columns.copy()

        valid = y.notna() & X.notna().all(axis=1)

        if weights is not None:
            valid &= weights.notna() & np.isfinite(weights) & (weights > 0)

        dropped_assets = tuple(X.index.get_level_values("asset")[~valid].astype(str).tolist())

        X = X.loc[valid]
        y = y.loc[valid]

        if weights is not None:
            weights = weights.loc[valid]

        if len(X) < min_obs:
            return None

        #
        # A categorical column may exist in the full historical panel
        # without being represented on this particular date.
        #
        active_columns = X.columns[X.ne(0).any(axis=0)]

        X_active = X.loc[:, active_columns]

        if X_active.empty:
            return None

        constraints = self._build_constraints(
            columns=X_active.columns,
            groups=groups,
        )

        beta_active = self._solve_constrained_wls(
            X=X_active,
            y=y,
            weights=weights,
            constraints=constraints,
            ridge=ridge,
        )

        #
        # Restore inactive factor columns as NaN.
        #
        factor_returns = beta_active.reindex(all_columns)
        factor_returns.name = date

        fitted = X_active @ beta_active
        fitted.name = "fitted_return"

        residuals = y - fitted
        residuals.name = "residual"

        r2 = self._compute_r2(
            y=y,
            residuals=residuals,
        )

        fitted_return_corr = self._compute_fitted_return_corr(
            y=y,
            fitted=fitted,
        )

        constraint_error = self._compute_constraint_error(
            beta=beta_active,
            constraints=constraints,
        )

        vif = self._compute_vif(
            X=X,
            groups=groups,
        )

        return DailyRegressionResult(
            date=date,
            factor_returns=factor_returns,
            fitted_returns=fitted,
            residuals=residuals,
            r2=r2,
            fitted_return_corr=fitted_return_corr,
            n_obs=len(X),
            constraint_error=constraint_error,
            dropped_assets=dropped_assets,
            vif=vif,
        )

    def _build_constraints(
        self,
        columns: pd.Index,
        groups: tuple[FactorExposureGroup, ...],
    ) -> np.ndarray | None:
        """
        Construct one sum-to-zero constraint for each categorical
        factor group.

        Example:

            country_DE + country_FR + country_IT = 0

        The rows of the returned matrix A define

            A beta = 0.
        """
        constraints: list[np.ndarray] = []

        column_positions = {column: i for i, column in enumerate(columns)}

        for group in groups:
            if not isinstance(
                group.factor,
                CategoricalFactor,
            ):
                continue

            active_group_columns = [
                column for column in group.columns if column in column_positions
            ]

            #
            # If no category from this factor is represented today,
            # there is nothing to constrain.
            #
            if not active_group_columns:
                continue

            constraint = np.zeros(
                len(columns),
                dtype=float,
            )

            for column in active_group_columns:
                constraint[column_positions[column]] = 1.0

            constraints.append(constraint)

        if not constraints:
            return None

        return np.vstack(constraints)

    @staticmethod
    def _solve_constrained_wls(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        weights: pd.Series | None,
        constraints: np.ndarray | None,
        ridge: float | None,
    ) -> pd.Series:
        """
        Solve

            min_beta
                (y - X beta)' W (y - X beta)
                + ridge * ||beta||^2

            subject to

                A beta = 0

        using the KKT system.
        """
        X_values = X.to_numpy(dtype=float)
        y_values = y.to_numpy(dtype=float)

        n_factors = X_values.shape[1]

        if weights is None:
            sqrt_w = np.ones(
                len(y_values),
                dtype=float,
            )
        else:
            sqrt_w = np.sqrt(weights.to_numpy(dtype=float))

        X_weighted = X_values * sqrt_w[:, None]

        y_weighted = y_values * sqrt_w

        xtx = X_weighted.T @ X_weighted

        xty = X_weighted.T @ y_weighted

        if ridge is not None:
            xtx = xtx + ridge * np.eye(n_factors)

        #
        # Unconstrained regression.
        #
        if constraints is None or constraints.shape[0] == 0:
            try:
                beta = np.linalg.solve(
                    xtx,
                    xty,
                )
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(
                    X_weighted,
                    y_weighted,
                    rcond=None,
                )[0]

        #
        # Constrained regression:
        #
        # [X'WX  A'] [beta]   [X'Wy]
        # [ A     0] [lambda] = [  0 ]
        #
        else:
            n_constraints = constraints.shape[0]

            kkt = np.block(
                [
                    [
                        xtx,
                        constraints.T,
                    ],
                    [
                        constraints,
                        np.zeros(
                            (
                                n_constraints,
                                n_constraints,
                            )
                        ),
                    ],
                ]
            )

            rhs = np.concatenate(
                [
                    xty,
                    np.zeros(n_constraints),
                ]
            )

            try:
                solution = np.linalg.solve(
                    kkt,
                    rhs,
                )
            except np.linalg.LinAlgError:
                solution = np.linalg.lstsq(
                    kkt,
                    rhs,
                    rcond=None,
                )[0]

            beta = solution[:n_factors]

        return pd.Series(
            beta,
            index=X.columns,
            dtype=float,
            name="factor_return",
        )

    @staticmethod
    def _compute_r2(
        y: pd.Series,
        residuals: pd.Series,
    ) -> float:
        """
        Cross-sectional in-sample R².
        """
        ss_res = float(np.sum(residuals.to_numpy() ** 2))

        y_values = y.to_numpy(dtype=float)

        ss_tot = float(np.sum((y_values - y_values.mean()) ** 2))

        if ss_tot <= 0:
            return np.nan

        return 1.0 - ss_res / ss_tot

    @staticmethod
    def _compute_fitted_return_corr(
        y: pd.Series,
        fitted: pd.Series,
    ) -> float:
        """
        Correlation between fitted and realised returns within the
        same cross-section.

        This is an in-sample diagnostic, not a predictive IC.
        """
        y_values = y.to_numpy(dtype=float)
        fitted_values = fitted.to_numpy(dtype=float)

        if np.std(y_values) == 0 or np.std(fitted_values) == 0:
            return np.nan

        return float(
            np.corrcoef(
                fitted_values,
                y_values,
            )[0, 1]
        )

    @staticmethod
    def _compute_constraint_error(
        beta: pd.Series,
        constraints: np.ndarray | None,
    ) -> float:
        """
        Maximum absolute violation of A beta = 0.
        """
        if constraints is None or constraints.shape[0] == 0:
            return 0.0

        violations = constraints @ beta.to_numpy(dtype=float)

        return float(np.max(np.abs(violations)))

    @staticmethod
    def _compute_vif(
        X: pd.DataFrame,
        groups: tuple[FactorExposureGroup, ...],
    ) -> pd.Series:
        """
        Compute variance inflation factors for continuous factor exposures.

        Categorical factor dummy columns are excluded because their
        multicollinearity is structural and handled separately through
        regression constraints.
        """
        continuous_columns = [
            column
            for group in groups
            if isinstance(group.factor, ContinuousFactor)
            for column in group.columns
            if column in X.columns
        ]

        if not continuous_columns:
            return pd.Series(
                dtype=float,
                name="vif",
            )

        X_vif = X.loc[:, continuous_columns].dropna()

        #
        # VIF requires enough observations relative to the number
        # of regressors.
        #
        if len(X_vif) <= len(continuous_columns):
            return pd.Series(
                np.nan,
                index=continuous_columns,
                dtype=float,
                name="vif",
            )

        # With no other continuous exposure to regress against, there is no
        # variance inflation. statsmodels cannot fit this zero-predictor
        # auxiliary regression, so handle the case directly.
        if len(continuous_columns) == 1:
            return pd.Series(
                1.0,
                index=continuous_columns,
                dtype=float,
                name="vif",
            )

        values = X_vif.to_numpy(dtype=float)

        vif_values: dict[str, float] = {}

        for i, column in enumerate(continuous_columns):
            try:
                vif = variance_inflation_factor(
                    values,
                    i,
                )
            except ValueError, np.linalg.LinAlgError:
                vif = np.nan

            vif_values[column] = float(vif)

        return pd.Series(
            vif_values,
            dtype=float,
            name="vif",
        )

    @staticmethod
    def _combine_results(
        *,
        results: Sequence[DailyRegressionResult],
        exposure_index: pd.Index,
        factor_columns: pd.Index,
    ) -> FactorModelResult:
        """
        Combine daily regression outputs into model-level results.
        """
        fitted_returns = pd.Series(
            np.nan,
            index=exposure_index,
            dtype=float,
            name="fitted_return",
        )

        residuals = pd.Series(
            np.nan,
            index=exposure_index,
            dtype=float,
            name="residual",
        )

        vif_columns = sorted(
            {
                column
                for result in results
                for column in result.vif.index
            }
        )

        if not results:
            factor_returns = pd.DataFrame(
                columns=factor_columns,
                dtype=float,
            )

            factor_returns.index = pd.Index(
                [],
                name="date",
            )

            diagnostics = pd.DataFrame(
                columns=[
                    "n_obs",
                    "r2",
                    "fitted_return_corr",
                    "constraint_error",
                    "n_dropped",
                    "dropped_assets",
                ]
            )

            diagnostics.index.name = "date"

            return FactorModelResult(
                factor_returns=factor_returns,
                fitted_returns=fitted_returns,
                residuals=residuals,
                diagnostics=diagnostics,
            )

        #
        # Factor returns
        #
        factor_returns = pd.DataFrame([result.factor_returns for result in results])

        factor_returns.index = pd.Index(
            [result.date for result in results],
            name="date",
        )

        factor_returns = factor_returns.reindex(columns=factor_columns).sort_index()

        #
        # Asset-level outputs
        #
        for result in results:
            fitted_returns.loc[result.fitted_returns.index] = result.fitted_returns

            residuals.loc[result.residuals.index] = result.residuals

        #
        # Diagnostics
        #
        diagnostics = pd.DataFrame(
            {
                "date": [result.date for result in results],
                "n_obs": [result.n_obs for result in results],
                "r2": [result.r2 for result in results],
                "fitted_return_corr": [result.fitted_return_corr for result in results],
                "constraint_error": [result.constraint_error for result in results],
                "n_dropped": [len(result.dropped_assets) for result in results],
                "dropped_assets": [result.dropped_assets for result in results],
                **{
                    f"vif_{column}": [result.vif.get(column, np.nan) for result in results]
                    for column in vif_columns
                },
            }
        )

        diagnostics = diagnostics.set_index("date").sort_index()

        return FactorModelResult(
            factor_returns=factor_returns,
            fitted_returns=fitted_returns,
            residuals=residuals,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_fit_inputs(
        *,
        data: pd.DataFrame,
        returns: pd.Series,
        weight_col: str | None,
        min_obs: int,
        ridge: float | None,
    ) -> None:
        if not isinstance(
            data.index,
            pd.MultiIndex,
        ):
            raise TypeError("Data must use a MultiIndex.")

        if not isinstance(
            returns.index,
            pd.MultiIndex,
        ):
            raise TypeError("Returns must use a MultiIndex.")

        required_levels = {
            "date",
            "asset",
        }

        missing_data_levels = required_levels - set(data.index.names)

        if missing_data_levels:
            raise ValueError(f"Data index is missing levels: {sorted(missing_data_levels)}")

        missing_return_levels = required_levels - set(returns.index.names)

        if missing_return_levels:
            raise ValueError(f"Returns index is missing levels: {sorted(missing_return_levels)}")

        if not returns.index.is_unique:
            raise ValueError("Returns index must be unique.")

        if not data.index.is_unique:
            raise ValueError("Data index must be unique.")

        if weight_col is not None:
            if weight_col not in data.columns:
                raise ValueError(f"Weight column {weight_col!r} not found in data.")

        if min_obs <= 0:
            raise ValueError("min_obs must be positive.")

        if ridge is not None:
            if ridge < 0:
                raise ValueError("ridge must be non-negative.")
