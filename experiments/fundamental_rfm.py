from argparse import ArgumentParser, Namespace
from datetime import date as Date
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exfa import FundamentalFactorModel
from exfa.diagnostics import RiskFactorDiagnostics, compute_model_regression_report
from exfa.models.fundamental_model.factors import (
    CountryFactor,
    InterceptFactor,
    MomentumFactor,
    SectorFactor,
)

if TYPE_CHECKING or __package__:
    from .market_panel import prepare_classified_panel, read_classified_prices
    from .plotting import LATEX_STYLE, plot_asset_reconstructions, save_figure
else:
    from market_panel import prepare_classified_panel, read_classified_prices
    from plotting import LATEX_STYLE, plot_asset_reconstructions, save_figure


def plot_factor_returns(
    dates: pd.DatetimeIndex, returns: np.ndarray, groups: dict[str, list[str]]
) -> None:
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, 9), sharex=True)
    offset = 0
    for ax, (group, labels) in zip(axes, groups.items(), strict=True):
        for label, series in zip(labels, returns[offset : offset + len(labels)], strict=True):
            ax.plot(dates, 100 * np.cumsum(series), label=label, linewidth=0.65)
        offset += len(labels)
        ax.set_title(group, loc="left")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7)
    save_figure(fig, "fundamental_rfm/fundamental_rfm_factor_returns.png")


def plot_fitted_diagnostics(
    model: FundamentalFactorModel, prices: pd.DataFrame, labels: list[str]
) -> None:
    dates = pd.DatetimeIndex(prices.columns[1:])
    report = compute_model_regression_report(model, continuous_columns=(0, 1, 2))
    fig, (fit, vif) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    fit.plot(dates, 100 * report.r2, alpha=0.35, linewidth=0.7, label="Daily")
    fit.plot(dates, 100 * pd.Series(report.r2).rolling(21).mean(), label="21-session mean")
    fit.set_ylabel("Cross-sectional R² (%)")
    fit.legend()
    for name, values in zip(labels[:3], report.vif, strict=True):
        vif.plot(dates, values, label=name, linewidth=0.75)
    vif.set_ylabel("Momentum VIF")
    vif.legend()
    save_figure(fig, "fundamental_rfm/fundamental_rfm_regression.png")

    fig, ax = plt.subplots(figsize=(9, max(4, len(prices) * 0.18)))
    limit = float(np.abs(model.loadings).max())
    heatmap = ax.imshow(model.loadings, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(prices)), prices.index.astype(str), fontsize=7)
    ax.set_title("Loadings for the last fitted session")
    fig.colorbar(heatmap, ax=ax, label="Loading")
    save_figure(fig, "fundamental_rfm/fundamental_rfm_loadings.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for covariance, label in (
        (np.cov(model.returns), "Sample covariance"),
        (model.covariance, "Fundamental covariance"),
    ):
        ax.plot(
            np.arange(1, model.num_assets + 1),
            np.linalg.eigvalsh(covariance)[::-1],
            label=label,
        )
    ax.set(xlabel="Eigenvalue rank", ylabel="Daily return variance")
    ax.legend()
    save_figure(fig, "fundamental_rfm/fundamental_rfm_spectrum.png")

    residuals = model.residuals
    active = (model.returns.std(axis=1) > 0) & (residuals.std(axis=1) > 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    axes[0].hist(100 * residuals.std(axis=1), bins=20)
    axes[0].set(xlabel="Daily residual volatility (%)", ylabel="Assets")
    for ax, values, title in (
        (axes[1], model.returns, "Daily returns"),
        (axes[2], residuals, "Regression residuals"),
    ):
        heatmap = ax.imshow(np.corrcoef(values[active]), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=title, xlabel="Asset", ylabel="Asset")
        fig.colorbar(heatmap, ax=ax, shrink=0.7, label="Correlation")
    save_figure(fig, "fundamental_rfm/fundamental_rfm_residuals.png")

    plot_asset_reconstructions(
        prices, model.returns, residuals, "fundamental_rfm/fundamental_rfm_reconstruction.png"
    )
    print(
        f"Median daily cross-sectional R²={np.nanmedian(report.r2):.1%}; "
        f"median asset residual volatility={np.median(residuals.std(axis=1)):.2%}."
    )


def evaluate_rolling_risk(
    model: FundamentalFactorModel,
    dates: pd.DatetimeIndex,
    train_sessions: int,
    test_sessions: int,
) -> None:
    factor_returns, residuals = model.factor_returns, model.residuals
    scores: dict[str, list[float]] = {"Fundamental": [], "Diagonal": []}
    risks: dict[str, list[tuple[float, float]]] = {"Fundamental": [], "Diagonal": []}
    forecast_dates = []
    stability = []
    previous_loadings = None

    for start in range(train_sessions, model.num_times - test_sessions + 1, test_sessions):
        training = slice(start - train_sessions, start)
        training_returns = model.returns[:, training]
        held_out = model.returns[:, start : start + test_sessions] - training_returns.mean(
            axis=1, keepdims=True
        )
        # First-day loadings use prices through the preceding close and stay fixed for the block.
        loadings = model.daily_results[start].loadings
        factor_covariance = np.cov(factor_returns[:, training])
        covariance = loadings @ factor_covariance @ loadings.T + np.diag(
            residuals[:, training].var(axis=1, ddof=1)
        )
        forecast_dates.append(dates[start])
        for name, estimate in (
            ("Fundamental", covariance),
            ("Diagonal", np.diag(training_returns.var(axis=1, ddof=1))),
        ):
            diagnostics = RiskFactorDiagnostics(estimate, loadings)
            scores[name].append(diagnostics.oos_gaussian_nll(held_out) / model.num_assets)
            risks[name].append(diagnostics.portfolio_risk(held_out)["equal_weight"])
        if previous_loadings is not None:
            comparison = RiskFactorDiagnostics.subspace_comparison(previous_loadings, loadings)
            stability.append(comparison.largest_principal_angle_degrees)
        previous_loadings = loadings

    fig, ax = plt.subplots(figsize=(9, 3.7))
    for name, values in scores.items():
        ax.plot(forecast_dates, values, marker=".", label=name)
        print(
            f"{name}: mean held-out Gaussian negative log-likelihood per asset={np.mean(values):.3f}."
        )
    ax.set(xlabel="Forecast block start", ylabel="Gaussian negative log-likelihood per asset")
    ax.legend()
    save_figure(fig, "fundamental_rfm/fundamental_rfm_covariance_validation.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    limit = 0.0
    for name, values in risks.items():
        predicted, realized = 100 * np.sqrt(np.asarray(values).T)
        ax.scatter(predicted, realized, label=name, alpha=0.75)
        limit = max(limit, float(predicted.max()), float(realized.max()))
    ax.plot([0, limit], [0, limit], "--", color="0.35", label="Perfect calibration")
    ax.set(
        xlabel="Predicted daily volatility (%)",
        ylabel="Realized daily volatility (%)",
        xlim=(0, limit * 1.05),
        ylim=(0, limit * 1.05),
        aspect="equal",
    )
    ax.legend()
    save_figure(fig, "fundamental_rfm/fundamental_rfm_portfolio_risk.png")

    fig, ax = plt.subplots(figsize=(9, 3.7))
    ax.plot(forecast_dates[1:], stability, marker=".", linewidth=0.8)
    ax.set(xlabel="Forecast block start", ylabel="Largest principal angle (degrees)")
    save_figure(fig, "fundamental_rfm/fundamental_rfm_factor_stability.png")
    print(
        f"Rolling covariance evaluation used {len(forecast_dates)} complete forecast blocks."
    )


def build_factors() -> tuple[
    list[MomentumFactor], CountryFactor, SectorFactor, InterceptFactor
]:
    # Match main.py's factors. Lag the momentum inputs by a session for forecasting.
    momentum = [
        MomentumFactor(
            name=f"mom_{window}d", window=window, impute=True, winsorize=True, standardize=True
        )
        for window in (21, 126, 252)
    ]
    country = CountryFactor(column="country")
    sector = SectorFactor(sector_col="industry")
    intercept = InterceptFactor()
    return momentum, country, sector, intercept


def fit_fundamental_model(
    prices: pd.DataFrame, classifications: pd.DataFrame
) -> tuple[FundamentalFactorModel, dict[str, list[str]], pd.DataFrame]:
    momentum, country, sector, intercept = build_factors()
    history = max(factor.window for factor in momentum)
    if prices.shape[1] < history + 3:
        raise ValueError("Need momentum warm-up and at least two fitted return sessions.")
    values = prices.to_numpy()
    market_data = {
        "price": values[:, :-1],
        "country": classifications["country"].to_numpy(),
        "industry": classifications["industry"].to_numpy(),
    }
    returns = np.diff(np.log(values), axis=1)[:, history:]
    model = FundamentalFactorModel([*momentum, country, sector, intercept]).fit(
        returns, market_data
    )
    groups = {
        "Momentum": [factor.name for factor in momentum],
        "Country": list(country.compute_categories(market_data, returns.shape)),
        "Sector": list(sector.compute_categories(market_data, returns.shape)),
        "Intercept": [intercept.name],
    }
    return model, groups, prices.iloc[:, history:]


def run_experiment(
    prices: pd.DataFrame,
    classifications: pd.DataFrame,
    train_sessions: int,
    test_sessions: int,
    max_fill_sessions: int,
) -> None:
    fitted_prices, classifications = prepare_classified_panel(
        prices, classifications, max_fill_sessions
    )
    model, groups, aligned_prices = fit_fundamental_model(fitted_prices, classifications)
    if (
        train_sessions < 2
        or test_sessions < 2
        or model.num_times < train_sessions + test_sessions
    ):
        raise ValueError("Need a covariance training window and a complete forecast block.")
    dates = pd.DatetimeIndex(aligned_prices.columns[1:])
    print(
        f"Fitted {model.num_factors} factor columns to {model.num_assets} assets "
        f"and {model.num_times} returns per asset ({dates[0]:%Y-%m-%d} through {dates[-1]:%Y-%m-%d})."
    )
    with plt.rc_context(LATEX_STYLE):
        plot_factor_returns(dates, model.factor_returns, groups)
        plot_fitted_diagnostics(model, aligned_prices, sum(groups.values(), []))
        evaluate_rolling_risk(model, dates, train_sessions, test_sessions)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Evaluate the fundamental factor model on PostgreSQL VWAPs."
    )
    parser.add_argument("--start-date", type=Date.fromisoformat, default=Date(2022, 1, 3))
    parser.add_argument("--end-date", type=Date.fromisoformat, default=Date(2026, 6, 30))
    parser.add_argument("--train-sessions", type=int, default=252)
    parser.add_argument("--test-sessions", type=int, default=21)
    parser.add_argument("--max-fill-sessions", type=int, default=5)
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    if args.max_fill_sessions < 0:
        parser.error("--max-fill-sessions must be nonnegative")
    return args


if __name__ == "__main__":
    args = parse_args()
    prices, classifications = read_classified_prices(args.start_date, args.end_date)
    run_experiment(
        prices,
        classifications,
        args.train_sessions,
        args.test_sessions,
        args.max_fill_sessions,
    )
