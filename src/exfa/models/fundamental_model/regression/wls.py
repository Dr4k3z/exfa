from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RegressionResult:
    coefficients: FloatArray
    residuals: FloatArray
    fitted_values: FloatArray
    rank: int
    residual_sum_squares: float


@dataclass
class LinearConstraint:
    """
    Linear equality constraint: C @ beta = d
    """

    matrix: FloatArray
    target: FloatArray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)

        if self.matrix.ndim != 2:
            raise ValueError("`matrix` must be two-dimensional.")

        if self.target.ndim != 1:
            raise ValueError("`target` must be one-dimensional.")

        if self.matrix.shape[0] != self.target.shape[0]:
            raise ValueError(
                "Constraint matrix and target have incompatible shapes: "
                f"{self.matrix.shape} and {self.target.shape}."
            )


class ConstrainedWLS:
    """
    Weighted least-squares estimator with optional linear equality constraints.
        min_beta (y - X beta)' W (y - X beta) + ridge * beta' beta
        s.t. C beta = d.
    """

    def __init__(
        self,
        rcond: float | None = None,
        ridge: float | None = None,
    ) -> None:
        if ridge is not None and ridge < 0:
            raise ValueError("`ridge` must be non-negative.")

        self.rcond = rcond
        self.ridge = ridge

    def fit(
        self,
        y: FloatArray,
        x: FloatArray,
        weights: FloatArray | None = None,
        constraint: LinearConstraint | None = None,
    ) -> RegressionResult:
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)

        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)

        self._validate_inputs(
            y=y,
            x=x,
            weights=weights,
            constraint=constraint,
        )

        if weights is None:
            weights_array = np.ones(y.shape[0], dtype=np.float64)
        else:
            weights_array = weights

        if constraint is None:
            coefficients = self.solve_unconstrained(
                y=y,
                x=x,
                weights=weights_array,
            )
        else:
            coefficients = self.solve_constrained(
                y=y,
                x=x,
                weights=weights_array,
                constraint=constraint,
            )

        fitted_values = x @ coefficients
        residuals = y - fitted_values

        weighted_residuals = np.sqrt(weights_array) * residuals
        residual_sum_squares = float(weighted_residuals @ weighted_residuals)

        rank = int(np.linalg.matrix_rank(x))

        return RegressionResult(
            coefficients=coefficients,
            residuals=residuals,
            fitted_values=fitted_values,
            rank=rank,
            residual_sum_squares=residual_sum_squares,
        )

    def solve_unconstrained(
        self,
        y: FloatArray,
        x: FloatArray,
        weights: FloatArray,
    ) -> FloatArray:
        sqrt_weights = np.sqrt(weights)

        x_weighted = x * sqrt_weights[:, None]
        y_weighted = y * sqrt_weights

        if self.ridge is None or self.ridge == 0:
            coefficients = np.linalg.lstsq(
                x_weighted,
                y_weighted,
                rcond=self.rcond,
            )[0]
        else:
            n_features = x.shape[1]

            x_augmented = np.vstack(
                [
                    x_weighted,
                    np.sqrt(self.ridge)
                    * np.eye(
                        n_features,
                        dtype=np.float64,
                    ),
                ]
            )
            y_augmented = np.concatenate(
                [
                    y_weighted,
                    np.zeros(n_features, dtype=np.float64),
                ]
            )

            coefficients = np.linalg.lstsq(
                x_augmented,
                y_augmented,
                rcond=self.rcond,
            )[0]

        return coefficients

    def solve_constrained(
        self,
        y: FloatArray,
        x: FloatArray,
        weights: FloatArray,
        constraint: LinearConstraint,
    ) -> FloatArray:
        c = constraint.matrix
        d = constraint.target

        weighted_x = weights[:, None] * x

        xtwx = x.T @ weighted_x
        xtwy = x.T @ (weights * y)

        if self.ridge is not None and self.ridge > 0:
            xtwx = xtwx + self.ridge * np.eye(
                x.shape[1],
                dtype=np.float64,
            )

        n_coefficients = x.shape[1]
        n_constraints = c.shape[0]

        kkt_matrix = np.block(
            [
                [
                    xtwx,
                    c.T,
                ],
                [
                    c,
                    np.zeros(
                        (n_constraints, n_constraints),
                        dtype=np.float64,
                    ),
                ],
            ]
        )

        kkt_rhs = np.concatenate(
            [
                xtwy,
                d,
            ]
        )

        solution = np.linalg.lstsq(
            kkt_matrix,
            kkt_rhs,
            rcond=self.rcond,
        )[0]

        return solution[:n_coefficients]

    @staticmethod
    def _validate_inputs(
        y: FloatArray,
        x: FloatArray,
        weights: FloatArray | None,
        constraint: LinearConstraint | None,
    ) -> None:
        if y.ndim != 1:
            raise ValueError(f"`y` must have shape (n_samples,), got {y.shape}.")

        if x.ndim != 2:
            raise ValueError(f"`x` must have shape (n_samples, n_features), got {x.shape}.")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "`x` and `y` contain different numbers of observations: "
                f"{x.shape[0]} != {y.shape[0]}."
            )

        if x.shape[1] == 0:
            raise ValueError("`x` must contain at least one explanatory variable.")

        if not np.all(np.isfinite(y)):
            raise ValueError("`y` contains NaN or infinite values.")

        if not np.all(np.isfinite(x)):
            raise ValueError("`x` contains NaN or infinite values.")

        if weights is not None:
            if weights.ndim != 1:
                raise ValueError("`weights` must be one-dimensional.")

            if weights.shape[0] != y.shape[0]:
                raise ValueError(
                    "`weights` and `y` contain different numbers of observations."
                )

            if not np.all(np.isfinite(weights)):
                raise ValueError("`weights` contains NaN or infinite values.")

            if np.any(weights < 0):
                raise ValueError("`weights` cannot contain negative values.")

            if not np.any(weights > 0):
                raise ValueError("`weights` must contain at least one positive value.")

        if constraint is not None:
            if constraint.matrix.shape[1] != x.shape[1]:
                raise ValueError(
                    "Constraint matrix has incompatible number of columns: "
                    f"expected {x.shape[1]}, "
                    f"got {constraint.matrix.shape[1]}."
                )

            if not np.all(np.isfinite(constraint.matrix)):
                raise ValueError("Constraint matrix contains NaN or infinite values.")

            if not np.all(np.isfinite(constraint.target)):
                raise ValueError("Constraint target contains NaN or infinite values.")
