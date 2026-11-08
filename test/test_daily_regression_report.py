"""
Test historical regression metrics independently of covariance diagnostics.
"""

from collections.abc import Sequence
from typing import NamedTuple, override

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from exfa import FundamentalFactorModel
from exfa.diagnostics import compute_daily_regression_report, compute_model_regression_report
from exfa.diagnostics.regression import compute_vif
from exfa.models.fundamental_model.factors import CountryFactor, InterceptFactor
from exfa.models.fundamental_model.factors.base import (
    ContinuousFactor,
    MarketData,
    get_feature,
)
from exfa.models.fundamental_model.regression import CrossSectionalRegressionResult
from exfa.models.fundamental_model.regression.wls import LinearConstraint


class FeatureFactor(ContinuousFactor):
    def __init__(self, name: str) -> None:
        super().__init__(name, False, False, False)

    @property
    @override
    def required_features(self) -> Sequence[str]:
        return (self.name,)

    @override
    def _compute(self, data: MarketData, shape: tuple[int, int]) -> np.ndarray:
        return get_feature(data, self.name, shape)


def test_daily_centered_r2_and_correlation():
    returns = np.array([[11.0, 3.0], [12.0, 1.0], [13.0, 2.0]])
    errors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    report = compute_daily_regression_report(
        returns, errors, np.ones((3, 1, 2)), np.ones((1, 2))
    )
    assert_allclose(report.r2, [0.0, 0.5])
    expected = [
        np.corrcoef(returns[:, time], returns[:, time] - errors[:, time])[0, 1]
        for time in range(2)
    ]
    assert_allclose(report.fitted_return_corr, expected)
    assert_array_equal(report.n_obs, [3, 3])
    assert_array_equal(report.n_dropped, [0, 0])
    assert_array_equal(report.constraint_error, [0, 0])
    assert report.vif.shape == (0, 2)
    assert report.r2[0] != pytest.approx(
        1 - np.sum(errors[:, 0] ** 2) / np.sum(returns[:, 0] ** 2)
    )


def test_model_adapter_uses_stored_residuals_and_historical_loadings():
    rng = np.random.default_rng(12)
    history = rng.normal(size=(10, 2, 4))
    scores = rng.normal(size=(2, 4))
    returns = np.einsum("akt,kt->at", history, scores) + rng.normal(scale=0.1, size=(10, 4))
    model = FundamentalFactorModel([FeatureFactor("first"), FeatureFactor("second")]).fit(
        returns, {"first": history[:, 0, :], "second": history[:, 1, :]}, min_obs=3
    )
    report = compute_model_regression_report(model, continuous_columns=[0, 1])
    expected = 1 - np.sum(model.residuals**2, axis=0) / np.sum(
        (returns - returns.mean(axis=0)) ** 2, axis=0
    )
    assert_allclose(report.r2, expected)
    for time in range(4):
        assert_allclose(report.vif[:, time], compute_vif(history[:, :, time]))
    wrong_residuals = returns - model.loadings @ model.factor_returns
    wrong_r2 = 1 - np.sum(wrong_residuals**2, axis=0) / np.sum(
        (returns - returns.mean(axis=0)) ** 2, axis=0
    )
    assert not np.allclose(report.r2[:-1], wrong_r2[:-1])
    assert report.continuous_columns == (0, 1)


def test_constraints_and_explicit_continuous_selection():
    history = np.repeat(
        np.column_stack(
            [np.ones(6), [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], np.linspace(-1, 1, 6)]
        )[:, :, None],
        3,
        axis=2,
    )
    scores = np.array([[1, 2, 3], [0.3, 0.1, 0.2], [-0.3, -0.1, -0.2], [0.2, 0.4, -0.1]])
    returns = np.einsum("akt,kt->at", history, scores)
    model = FundamentalFactorModel(
        [
            InterceptFactor(),
            CountryFactor(),
            FeatureFactor("signal"),
        ]
    ).fit(
        returns,
        {"country": np.array(["DE"] * 3 + ["US"] * 3), "signal": history[:, 3, :]},
        min_obs=3,
    )
    report = compute_model_regression_report(model, continuous_columns=[3])
    assert_allclose(report.r2, 1)
    assert_allclose(report.constraint_error, 0, atol=1e-12)
    assert_allclose(report.vif, 1)
    violation = compute_daily_regression_report(
        returns,
        model.residuals,
        history,
        scores,
        constraints=[LinearConstraint(np.array([[0.0, 1, 1, 0]]), np.array([1.0]))] * 3,
    )
    assert_allclose(violation.constraint_error, 1)


def test_missing_rows_share_metric_masks():
    returns = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    residuals = np.zeros_like(returns)
    history = np.ones((4, 1, 2))
    returns[0, 0] = np.nan
    residuals[1, 1] = np.inf
    history[2, 0, 1] = np.nan
    report = compute_daily_regression_report(returns, residuals, history, np.ones((1, 2)))
    assert_array_equal(report.n_obs, [3, 2])
    assert_array_equal(report.n_dropped, [1, 2])
    assert report.dropped_assets == ((0,), (1, 2))
    assert_allclose(report.r2, 1)
    assert_allclose(report.fitted_return_corr, 1)


def test_degenerate_metrics_do_not_abort_report():
    returns = np.ones((3, 2))
    report = compute_daily_regression_report(
        returns,
        np.zeros_like(returns),
        np.ones((3, 1, 2)),
        np.ones((1, 2)),
        continuous_columns=[0],
    )
    assert np.isnan(report.r2).all()
    assert np.isnan(report.fitted_return_corr).all()
    assert np.isnan(report.vif).all()
    returns[:] = np.nan
    empty = compute_daily_regression_report(
        returns, returns, np.ones((3, 1, 2)), np.ones((1, 2))
    )
    assert_array_equal(empty.n_obs, [0, 0])
    assert np.isnan(empty.r2).all()


def test_vif_known_values_and_collinearity():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert_allclose(compute_vif(values[:, None]), [1])
    assert np.isinf(compute_vif(np.column_stack([values, values * 2]))).all()
    assert np.isnan(compute_vif(np.ones((4, 1)))).all()
    assert np.isnan(compute_vif(np.ones((2, 2)))).all()
    matrix = np.column_stack([values, [2, 0, 1, 0, 3]])
    cosine_squared = (matrix[:, 0] @ matrix[:, 1]) ** 2 / (
        (matrix[:, 0] @ matrix[:, 0]) * (matrix[:, 1] @ matrix[:, 1])
    )
    assert_allclose(compute_vif(matrix), 1 / (1 - cosine_squared))


@pytest.mark.parametrize("columns", [[0, 0], [-1], [2], [True], [0.5]])
def test_invalid_continuous_columns(columns):
    with pytest.raises(ValueError, match="continuous_columns"):
        compute_daily_regression_report(
            np.ones((3, 2)),
            np.zeros((3, 2)),
            np.ones((3, 2, 2)),
            np.ones((2, 2)),
            continuous_columns=columns,
        )


@pytest.mark.parametrize(
    "change", ["residuals", "loadings_history", "factor_returns", "constraints"]
)
def test_invalid_shapes(change):
    returns = np.ones((3, 2))
    residuals = np.zeros((3, 2)) if change != "residuals" else np.ones(1)
    history = np.ones((3, 1, 2)) if change != "loadings_history" else np.ones(1)
    scores = np.ones((1, 2)) if change != "factor_returns" else np.ones(1)
    with pytest.raises(ValueError):
        compute_daily_regression_report(
            returns,
            residuals,
            history,
            scores,
            constraints=[None] if change == "constraints" else (),
        )


def test_prefit_guard_is_preserved():
    with pytest.raises(RuntimeError):
        compute_model_regression_report(FundamentalFactorModel())


def test_negative_r2_and_undefined_constraint_are_not_hidden():
    report = compute_daily_regression_report(
        returns=np.array([[1.0], [2.0], [3.0]]),
        residuals=np.full((3, 1), 10.0),
        loadings_history=np.ones((3, 1, 1)),
        factor_returns=np.full((1, 1), np.nan),
        constraints=[LinearConstraint(np.ones((1, 1)), np.zeros(1))],
    )
    assert report.r2[0] == pytest.approx(-149.0)
    assert np.isnan(report.constraint_error[0])


class RegressionSnapshot(NamedTuple):
    returns: np.ndarray
    daily_results: tuple[CrossSectionalRegressionResult, ...]


def test_adapter_accepts_only_canonical_fitted_state() -> None:
    model = FundamentalFactorModel([InterceptFactor()]).fit(
        np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), {}, min_obs=2
    )
    snapshot = RegressionSnapshot(model.returns, model.daily_results)
    expected = compute_model_regression_report(model)
    actual = compute_model_regression_report(snapshot)
    assert_allclose(actual.r2, expected.r2)
    assert_array_equal(actual.n_obs, expected.n_obs)
    with pytest.raises(ValueError, match="one result per return time"):
        compute_model_regression_report(
            RegressionSnapshot(snapshot.returns, snapshot.daily_results[:1])
        )


def test_report_counts_are_derived_from_dropped_rows() -> None:
    returns = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, np.nan]])
    report = compute_daily_regression_report(
        returns, np.zeros((3, 2)), np.ones((3, 1, 2)), np.ones((1, 2))
    )
    assert report.num_assets == 3
    assert report.dropped_assets == ((0,), (1, 2))
    assert_array_equal(report.n_dropped, [1, 2])
    assert_array_equal(report.n_obs, [2, 1])
    report.n_obs[:] = 999
    report.n_dropped[:] = 999
    assert_array_equal(report.n_obs + report.n_dropped, [3, 3])
