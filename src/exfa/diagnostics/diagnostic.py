from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PortfolioRisk(NamedTuple):
    predicted_variance: float
    realized_variance: float

    @property
    def relative_error(self) -> float:
        return self.realized_variance / self.predicted_variance - 1.0


class FactorSubspace:
    def __init__(self, principal_angles: np.ndarray) -> None:
        self._angles = np.array(principal_angles, copy=True)

    @property
    def principal_angles(self) -> np.ndarray:
        return self._angles.copy()

    @property
    def squared_projection_distance(self) -> float:
        return float(np.sum(np.sin(self._angles) ** 2))

    @property
    def mean_squared_sine(self) -> float:
        return self.squared_projection_distance / len(self._angles)

    @property
    def mean_squared_cosine(self) -> float:
        return 1.0 - self.mean_squared_sine

    @property
    def largest_principal_angle_degrees(self) -> float:
        return float(np.degrees(np.max(self._angles)))


class RiskFactorDiagnostics:
    def __init__(self, covariance: np.ndarray, loadings: np.ndarray) -> None:
        self._estimated_covariance = np.array(covariance, copy=True)
        self._loadings = np.array(loadings, copy=True)

    @property
    def cholesky_factor(self) -> np.ndarray:
        return np.linalg.cholesky(self._estimated_covariance)

    @property
    def logdet(self) -> float:
        return float(2 * np.sum(np.log(np.diag(self.cholesky_factor))))

    @property
    def num_assets(self) -> int:
        return self._estimated_covariance.shape[0]

    @staticmethod
    def subspace_comparison(
        B1: np.ndarray,
        B2: np.ndarray,
    ) -> FactorSubspace:
        """
        Compare numerical column spaces using rank-aware orthonormal bases.

        Discard singular values at or below eps * max(shape) * largest singular
        value. Return min(rank(B1), rank(B2)) principal angles, as for full-rank
        inputs. Zero-rank spaces are rejected because angle summaries are undefined.
        Both loading matrices must be finite and have the same asset dimension.
        """

        first_loadings = np.asarray(B1, dtype=float)
        second_loadings = np.asarray(B2, dtype=float)
        if (
            first_loadings.ndim != 2
            or second_loadings.ndim != 2
            or first_loadings.shape[0] != second_loadings.shape[0]
        ):
            raise ValueError("loadings must be matrices with the same asset dimension")

        bases = []
        for loadings in (first_loadings, second_loadings):
            if not np.isfinite(loadings).all():
                raise ValueError("loadings must contain only finite values")
            left_vectors, singular_values, _ = np.linalg.svd(loadings, full_matrices=False)
            if not singular_values.size or singular_values[0] == 0:
                raise ValueError("loadings must have positive numerical rank")
            tolerance = np.finfo(float).eps * max(loadings.shape) * singular_values[0]
            bases.append(left_vectors[:, singular_values > tolerance])

        cosine = np.clip(
            np.linalg.svd(bases[0].T @ bases[1], compute_uv=False),
            0.0,
            1.0,
        )

        return FactorSubspace(np.arccos(cosine))

    def covariance_quality(self, population_covariance: np.ndarray) -> dict[str, float]:
        """
        Simulation-only evaluation requiring known population covariance.
        """

        delta = self._estimated_covariance - population_covariance
        relative_frobenius = np.linalg.norm(delta, "fro") / np.linalg.norm(
            population_covariance, "fro"
        )
        operator_error = np.linalg.norm(delta, ord=2)
        trace_term = np.trace(
            np.linalg.solve(self._estimated_covariance, population_covariance)
        )

        try:
            population_cholesky = np.linalg.cholesky(population_covariance)
        except np.linalg.LinAlgError as err:
            raise ValueError("Population covariance must be positive definite") from err

        true_logdet = 2 * np.sum(np.log(np.diag(population_cholesky)))
        stein = trace_term + self.logdet - true_logdet - self.num_assets

        return {
            "relative_frobenius": float(relative_frobenius),
            "operator_error": float(operator_error),
            "stein_loss": float(stein),
        }

    def oos_gaussian_nll(self, test_returns: np.ndarray) -> float:
        """
        Mean negative Gaussian log-likelihood, assuming zero-mean returns.
        """

        whitened = np.linalg.solve(self.cholesky_factor, test_returns)
        mahalanobis_mean = np.mean(np.sum(whitened**2, axis=0))

        return float(
            0.5 * (self.num_assets * np.log(2 * np.pi) + self.logdet + mahalanobis_mean)
        )

    def portfolio_risk(
        self,
        test_returns: np.ndarray,
        portfolios: dict[str, np.ndarray] | None = None,
    ) -> dict[str, PortfolioRisk]:
        """
        Compare ex-ante variance with subsequent mean squared portfolio returns.
        """

        if portfolios is None:
            portfolios = {
                "equal_weight": np.full(self.num_assets, 1.0 / self.num_assets),
            }

        output = {}
        for name, weights in portfolios.items():
            w = np.asarray(weights, dtype=float)

            pred = float(w @ self._estimated_covariance @ w)
            if pred <= 0:
                raise ValueError(f"Portfolio {name!r} must have positive predicted variance")

            observed = float(np.mean((w @ test_returns) ** 2))

            output[name] = PortfolioRisk(
                predicted_variance=pred,
                realized_variance=observed,
            )

        return output

    def factor_recovery(self, true_loadings: np.ndarray) -> FactorSubspace:
        """
        Simulation-only comparison of estimated and population loading spaces.
        """

        return self.subspace_comparison(self._loadings, true_loadings)

    def factor_stability(self, other_loadings: np.ndarray) -> FactorSubspace:
        """
        Compare loading spaces from different estimation windows.
        """

        return self.subspace_comparison(self._loadings, other_loadings)

    def reconstruct(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return residuals returns-BF, using least-squares factors when F is absent.
        """

        if factor_returns is None:
            factor_returns = np.linalg.lstsq(self._loadings, returns, rcond=None)[0]

        return returns - self._loadings @ factor_returns

    def explanatory_power(
        self,
        returns: np.ndarray,
        factor_returns: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Reconstruction quality for centered returns.
        """

        E = self.reconstruct(returns, factor_returns)
        total = float(np.sum(returns**2))

        if total == 0:
            raise ValueError("returns must have positive total sum of squares")

        squared_error = float(np.sum(E**2))

        return {
            "r2": 1.0 - squared_error / total,
            "mean_squared_residual": squared_error / returns.size,
        }

    def residual_structure(
        self,
        returns: np.ndarray,
        *,
        factor_returns: np.ndarray | None = None,
        max_assets: int = 200,
        seed: int = 42,
    ) -> dict[str, float]:
        """
        Assess residual correlation, variance dispersion, and serial dependence.
        """

        E = self.reconstruct(returns, factor_returns)
        n, T = E.shape

        if T < 3 or max_assets < 2:
            raise ValueError("Need T >= 3 and max_assets >= 2")

        variances = np.var(E, axis=1)
        positive = variances > 1e-20

        if not np.any(positive):
            raise ValueError("All residuals are constant")

        cv = float(np.std(variances[positive]) / np.mean(variances[positive]))

        active = np.flatnonzero(positive)
        if len(active) < 2:
            raise ValueError("At least two nonconstant residual rows are needed")

        if len(active) > max_assets:
            active = np.random.default_rng(seed).choice(active, size=max_assets, replace=False)

        sampled = E[active]
        C = np.cov(sampled, bias=True)

        sdev = np.sqrt(np.diag(C))
        corr = C / np.outer(sdev, sdev)

        offdiag = np.triu_indices(len(active), k=1)
        pairwise = np.abs(corr[offdiag])

        off_cov = C - np.diag(np.diag(C))
        off_frac = float(np.linalg.norm(off_cov, "fro") / np.linalg.norm(C, "fro"))

        lag_before = E[:, :-1]
        lag_after = E[:, 1:]

        lag_before = lag_before - lag_before.mean(axis=1, keepdims=True)
        lag_after = lag_after - lag_after.mean(axis=1, keepdims=True)

        numer = np.sum(lag_before * lag_after, axis=1)
        denom = np.sqrt(np.sum(lag_before**2, axis=1) * np.sum(lag_after**2, axis=1))
        lag_abs = np.abs(numer[denom > 1e-20] / denom[denom > 1e-20])

        return {
            "mean_absolute_pairwise_correlation": float(np.mean(pairwise)),
            "max_absolute_pairwise_correlation": float(np.max(pairwise)),
            "mean_absolute_lag1_autocorrelation": float(np.mean(lag_abs))
            if lag_abs.size
            else float("nan"),
            "offdiagonal_covariance_fraction": off_frac,
            "variance_coefficient_of_variation": cv,
            "assets_used": len(active),
        }

    def factor_vif(self, center: bool = True) -> np.ndarray:
        """
        Variance inflation factors of the loading columns.
        """

        B = self._loadings
        X = B - B.mean(axis=0, keepdims=True) if center else B.copy()

        norms = np.linalg.norm(X, axis=0)
        if np.any(norms == 0):
            raise ValueError("Factor loadings has zero variation")

        X = X / norms
        m = X.shape[1]

        if np.linalg.matrix_rank(X) < m:
            raise ValueError("Factor loadings are linearly dependent")

        return np.diag(np.linalg.solve(X.T @ X, np.eye(m)))
