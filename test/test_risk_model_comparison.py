import numpy as np
import pytest

from exfa.diagnostics import (
    RiskFactorDiagnostics,
    compare_factor_subspaces,
    compute_group_overlap,
    score_covariance_forecast,
    score_factor_reconstruction,
    score_factor_spanning,
)


def test_covariance_and_portfolio_scores_share_training_centered_returns():
    loadings = np.eye(2)
    diagnostics = RiskFactorDiagnostics(4 * np.eye(2), loadings)
    realized = np.array([[2.0, -2.0], [0.0, 0.0]])
    score = score_covariance_forecast(diagnostics, realized, {"first": np.array([1.0, 0.0])})
    assert score.gaussian_nll_per_asset == pytest.approx(0.5 * (np.log(8 * np.pi) + 0.5))
    portfolio = score.portfolios["first"]
    assert portfolio.variance_ratio == pytest.approx(1.0)
    assert portfolio.qlike == pytest.approx(np.log(4) + 1)
    zero_score = score_covariance_forecast(diagnostics, np.zeros_like(realized))
    assert np.isfinite(zero_score.portfolios["equal_weight"].qlike)


def test_covariance_score_validates_asset_alignment():
    diagnostics = RiskFactorDiagnostics(np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="forecast assets"):
        score_covariance_forecast(diagnostics, np.ones((3, 2)))
    with pytest.raises(ValueError, match="Portfolio weights"):
        score_covariance_forecast(diagnostics, np.ones((2, 2)), {"wrong": np.ones(3)})


def test_directional_overlap_uses_numerical_rank_and_ignores_basis_rotation():
    market, sector, style = np.eye(3)
    statistical = np.column_stack((market + sector, market - sector))
    fundamental = np.column_stack((market, sector, style, market))
    score = compare_factor_subspaces(statistical, fundamental)
    assert score.ranks == (2, 3)
    assert score.principal_angles_degrees == pytest.approx([0, 0], abs=2e-6)
    assert score.first_explained == pytest.approx(1)
    assert score.second_explained == pytest.approx(2 / 3)
    increments = compute_group_overlap(
        statistical, {"market": market[:, None], "rest": fundamental[:, 1:]}
    )
    assert increments == pytest.approx({"market": 0.5, "rest": 0.5})


def test_reconstruction_uses_the_loadings_for_each_session():
    returns = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 1.0]])
    fixed = np.array([[1.0], [0.0]])
    history = np.repeat(fixed[:, :, None], 3, axis=2)
    history[:, 0, 1] = [0, 1]
    score, residuals = score_factor_reconstruction(returns, history)
    assert residuals == pytest.approx(np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0]]))
    assert score.effective_rank == 1
    assert score.r2 == pytest.approx(1 - 9 / 20)
    assert score.rmse == pytest.approx(np.sqrt(9 / 6))
    assert score.largest_residual_eigenvalue > 0
    fixed_score, _ = score_factor_reconstruction(returns, fixed)
    assert fixed_score.r2 == pytest.approx(1 - 6 / 20)


def test_factor_spanning_freezes_the_training_mapping_and_intercept():
    train = np.array([[-2.0, -1.0, 1.0, 3.0]])
    test = np.array([[4.0, 5.0]])
    known_train = np.vstack((3 * train + 2, np.ones_like(train)))
    known_test = np.vstack((3 * test + 2, np.ones_like(test)))
    scores = score_factor_spanning(train, test, known_train, known_test)
    assert scores[0] == pytest.approx(1)
    assert np.isnan(scores[1])
    reversed_test = known_test.copy()
    reversed_test[0] *= -1
    scores = score_factor_spanning(train, test, known_train, reversed_test)
    assert scores[0] < 0
