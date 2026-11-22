"""
Compare covariance forecasts and factor-spanning estimates on matched samples.
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from .diagnostic import PortfolioRisk, RiskFactorDiagnostics


class PortfolioScore(NamedTuple):
    risk: PortfolioRisk

    @property
    def variance_ratio(self) -> float:
        return 1 + self.risk.relative_error

    @property
    def qlike(self) -> float:
        """
        Return Gaussian quasi-likelihood loss up to an observation-only term.

        The log-forecast form is defined even when realized variance is zero.
        """

        return float(np.log(self.risk.predicted_variance) + self.variance_ratio)


class ForecastScore(NamedTuple):
    gaussian_nll_per_asset: float
    portfolios: dict[str, PortfolioScore]


class ReconstructionScore(NamedTuple):
    r2: float
    rmse: float
    mean_absolute_residual_correlation: float
    largest_residual_eigenvalue: float
    effective_rank: int


class SubspaceOverlap(NamedTuple):
    ranks: tuple[int, int]
    principal_angles_degrees: np.ndarray
    first_explained: float
    second_explained: float


def score_covariance_forecast(
    diagnostics: RiskFactorDiagnostics,
    centered_returns: np.ndarray,
    portfolios: dict[str, np.ndarray] | None = None,
) -> ForecastScore:
    """
    Score an ex-ante covariance forecast on returns centered by the training mean.
    """

    returns = np.asarray(centered_returns, dtype=float)
    if (
        returns.ndim != 2
        or returns.shape[0] != diagnostics.num_assets
        or returns.shape[1] == 0
        or not np.isfinite(returns).all()
    ):
        raise ValueError(
            "Expected finite held-out returns shaped (forecast assets, sessions)."
        )
    if portfolios is not None and any(
        weights.shape != (diagnostics.num_assets,) or not np.isfinite(weights).all()
        for weights in portfolios.values()
    ):
        raise ValueError("Portfolio weights must be finite and match the forecast assets.")
    return ForecastScore(
        diagnostics.oos_gaussian_nll(returns) / diagnostics.num_assets,
        {
            name: PortfolioScore(risk)
            for name, risk in diagnostics.portfolio_risk(returns, portfolios).items()
        },
    )


def score_factor_reconstruction(
    returns: np.ndarray, loadings: np.ndarray
) -> tuple[ReconstructionScore, np.ndarray]:
    """
    Project centered returns onto fixed or per-session loadings by least squares.
    """

    returns = np.asarray(returns, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    if (
        returns.ndim != 2
        or returns.shape[0] < 2
        or returns.shape[1] < 2
        or loadings.ndim not in (2, 3)
        or loadings.shape[0] != returns.shape[0]
        or (loadings.ndim == 3 and loadings.shape[2] != returns.shape[1])
        or not np.isfinite(returns).all()
        or not np.isfinite(loadings).all()
    ):
        raise ValueError("Expected aligned, finite returns and fixed or per-session loadings.")
    if loadings.ndim == 2:
        coefficients, _, rank, _ = np.linalg.lstsq(loadings, returns, rcond=None)
        residuals = returns - loadings @ coefficients
        ranks = [rank]
    else:
        residuals = np.empty_like(returns)
        ranks = []
        for time, design in enumerate(loadings.transpose(2, 0, 1)):
            coefficients, _, rank, _ = np.linalg.lstsq(design, returns[:, time], rcond=None)
            residuals[:, time] = returns[:, time] - design @ coefficients
            ranks.append(rank)

    total = float(np.sum(returns**2))
    if total == 0:
        raise ValueError("Returns must have positive total variation.")
    covariance = np.cov(residuals, bias=True)
    active = np.flatnonzero(np.diag(covariance) > 1e-20)
    aggregate = float("nan")
    if len(active) > 1:
        correlation = np.corrcoef(residuals[active])
        aggregate = float(np.mean(np.abs(correlation[np.triu_indices(len(active), 1)])))
    score = ReconstructionScore(
        1 - float(np.sum(residuals**2)) / total,
        float(np.sqrt(np.mean(residuals**2))),
        aggregate,
        float(np.linalg.eigvalsh(covariance)[-1]),
        int(min(ranks)),
    )
    return score, residuals


def compare_factor_subspaces(first: np.ndarray, second: np.ndarray) -> SubspaceOverlap:
    """
    Measure the fraction of each loading space contained in the other.
    """

    comparison = RiskFactorDiagnostics.subspace_comparison(first, second)
    ranks = (int(np.linalg.matrix_rank(first)), int(np.linalg.matrix_rank(second)))
    shared = float(np.sum(np.cos(comparison.principal_angles) ** 2))
    return SubspaceOverlap(
        ranks,
        np.degrees(comparison.principal_angles),
        min(1.0, shared / ranks[0]),
        min(1.0, shared / ranks[1]),
    )


def score_factor_spanning(
    training_scores: np.ndarray,
    test_scores: np.ndarray,
    training_targets: np.ndarray,
    test_targets: np.ndarray,
) -> np.ndarray:
    """
    Freeze a linear mapping of latent scores to named factors and score test R².

    The baseline is the training mean of each target. Constant targets with
    zero test variation around that baseline have undefined scores.
    """

    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (training_scores, test_scores, training_targets, test_targets)
    )
    training_scores, test_scores, training_targets, test_targets = arrays
    if (
        any(
            values.ndim != 2 or values.size == 0 or not np.isfinite(values).all()
            for values in arrays
        )
        or training_scores.shape[0] != test_scores.shape[0]
        or training_targets.shape[0] != test_targets.shape[0]
        or training_scores.shape[1] != training_targets.shape[1]
        or test_scores.shape[1] != test_targets.shape[1]
    ):
        raise ValueError(
            "Expected aligned, finite score and target matrices (factors, sessions)."
        )
    score_mean = training_scores.mean(axis=1, keepdims=True)
    target_mean = training_targets.mean(axis=1, keepdims=True)
    coefficients = np.linalg.lstsq(
        (training_scores - score_mean).T, (training_targets - target_mean).T, rcond=None
    )[0]
    centered_test = test_targets - target_mean
    errors = centered_test - coefficients.T @ (test_scores - score_mean)
    total = np.sum(centered_test**2, axis=1)
    result = np.full(total.shape, np.nan)
    np.divide(np.sum(errors**2, axis=1), total, out=result, where=total > 0)
    return 1 - result


def compute_group_overlap(
    statistical_loadings: np.ndarray, groups: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """
    Attribute statistical loading-space overlap to ordered fundamental groups.
    """

    increments = {}
    blocks = []
    previous = 0.0
    for name, loadings in groups.items():
        blocks.append(loadings)
        overlap = compare_factor_subspaces(statistical_loadings, np.column_stack(blocks))
        increments[name] = max(0.0, overlap.first_explained - previous)
        previous = overlap.first_explained
    return increments
