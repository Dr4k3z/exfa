from argparse import ArgumentParser, Namespace
from datetime import date as Date
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exfa.diagnostics import RiskFactorDiagnostics
from exfa.models import StatisticalFactorModel

if TYPE_CHECKING or __package__:
    from .fundamental_rfm import build_factors
    from .market_panel import (
        prepare_classified_panel,
        prepare_price_panel,
        read_classified_prices,
    )
    from .plotting import LATEX_STYLE, plot_asset_reconstructions, save_figure
else:
    from fundamental_rfm import build_factors
    from market_panel import (
        prepare_classified_panel,
        prepare_price_panel,
        read_classified_prices,
    )
    from plotting import LATEX_STYLE, plot_asset_reconstructions, save_figure


def plot_spectrum(model: StatisticalFactorModel) -> None:
    sample_covariance = model.returns @ model.returns.T / model.num_times
    sample_eigenvalues = np.linalg.eigvalsh(sample_covariance)[::-1]
    model_eigenvalues = np.linalg.eigvalsh(model.covariance)[::-1]
    ranks = np.arange(1, model.num_assets + 1)

    fig, (head, full) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, count in (
        (head, min(model.num_factors + 5, model.num_assets)),
        (full, model.num_assets),
    ):
        ax.plot(ranks[:count], sample_eigenvalues[:count], label="Sample covariance")
        ax.plot(ranks[:count], model_eigenvalues[:count], "--", label="PPCA covariance")
        ax.axvline(model.num_factors + 0.5, color="0.5", linestyle=":", label="Rank cutoff")
        ax.axhline(model.residual_variance, color="0.25", linestyle="-.", label="Noise floor")
        ax.set(xlabel="Eigenvalue rank", ylabel="Daily return variance")
    head.set_title("Leading eigenvalues")
    full.set_title("Full spectrum")
    full.legend(fontsize=8)
    save_figure(fig, "statistical_rfm/statistical_rfm_spectrum.png")

    retained = ranks[:-1]
    eigenvalues = np.maximum(sample_eigenvalues, 0)
    cumulative = np.cumsum(eigenvalues)
    fig, (variance, noise) = plt.subplots(1, 2, figsize=(9, 3.8))
    variance.plot(ranks, 100 * cumulative / cumulative[-1])
    for threshold in (50, 70, 90):
        variance.axhline(threshold, color="0.65", linestyle=":", linewidth=0.8)
    noise.plot(retained, np.cumsum(eigenvalues[:0:-1])[::-1] / (model.num_assets - retained))
    for ax in (variance, noise):
        ax.axvline(model.num_factors, color="0.35", linestyle="--")
        ax.set_xlabel("Number of factors")
    variance.set_ylabel("Cumulative explained variance (%)")
    noise.set_ylabel("Estimated daily residual variance")
    save_figure(fig, "statistical_rfm/statistical_rfm_explained_variance.png")


def plot_loadings(model: StatisticalFactorModel, assets: pd.Index) -> None:
    loadings = model.loadings[:, :5]
    signs = np.sign(
        loadings[np.argmax(np.abs(loadings), axis=0), np.arange(loadings.shape[1])]
    )
    # Use an equal-weight return proxy available on the same raw-return panel.
    market = model.returns.mean(axis=0)
    correlations = np.corrcoef(np.vstack((market, model.factor_returns[:5] * signs[:, None])))
    print("Factor correlations with equal-weight returns:", np.round(correlations[0, 1:], 3))
    fig, ax = plt.subplots(figsize=(5.5, max(4, 0.19 * len(assets))))
    limit = float(np.abs(loadings).max())
    image = ax.imshow(loadings * signs, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(loadings.shape[1]), np.arange(1, loadings.shape[1] + 1))
    ax.set_yticks(np.arange(len(assets)), assets.astype(str), fontsize=7)
    ax.set_xlabel("Factor")
    fig.colorbar(image, ax=ax, label="Loading")
    save_figure(fig, "statistical_rfm/statistical_rfm_loadings.png")


def plot_residual_structure(
    model: StatisticalFactorModel, diagnostics: RiskFactorDiagnostics, residuals: np.ndarray
) -> None:
    summary = diagnostics.residual_structure(
        model.returns, factor_returns=model.factor_returns
    )
    print(
        "Residual diagnostics: "
        f"mean absolute pairwise correlation={summary['mean_absolute_pairwise_correlation']:.3f}, "
        f"residual variance CV={summary['variance_coefficient_of_variation']:.3f}, "
        f"mean absolute lag-one autocorrelation={summary['mean_absolute_lag1_autocorrelation']:.3f}."
    )

    active = np.flatnonzero(
        (model.returns.var(axis=1) > 1e-20) & (residuals.var(axis=1) > 1e-20)
    )
    if len(active) > 200:
        active = np.random.default_rng(42).choice(active, size=200, replace=False)
    pairs = np.triu_indices(len(active), k=1)

    fig, (volatility, correlation) = plt.subplots(1, 2, figsize=(9, 3.8))
    volatility.hist(100 * residuals.std(axis=1), bins=25, alpha=0.8)
    volatility.axvline(
        100 * np.sqrt(model.residual_variance),
        color="0.25",
        linestyle="--",
        label=r"PPCA noise scale $\hat{\sigma}$",
    )
    volatility.set(xlabel="Daily residual volatility (%)", ylabel="Assets")
    volatility.legend(fontsize=8)
    bins = np.linspace(0, 1, 26)
    for values, label in ((model.returns, "Returns"), (residuals, "Residuals")):
        absolute_correlation = np.abs(np.corrcoef(values[active])[pairs])
        correlation.hist(
            absolute_correlation,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=label,
        )
    correlation.set(xlabel="Absolute pairwise correlation", ylabel="Density")
    correlation.legend(fontsize=8)
    save_figure(fig, "statistical_rfm/statistical_rfm_residuals.png")

    fig, (original, projected) = plt.subplots(1, 2, figsize=(8, 4))
    for ax, values, title in (
        (original, model.returns, "Daily returns"),
        (projected, residuals, "Projected residuals"),
    ):
        image = ax.imshow(np.corrcoef(values[active]), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=title, xlabel="Asset", ylabel="Asset")
        fig.colorbar(image, ax=ax, shrink=0.75, label="Correlation")
    save_figure(fig, "statistical_rfm/statistical_rfm_correlations.png")


def plot_dimensional_stability(model: StatisticalFactorModel) -> None:
    max_length = model.num_times // 2
    lengths = sorted(
        {
            min(max_length, max(model.num_factors + 2, round(model.num_assets / ratio)))
            for ratio in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
        }
    )
    rng = np.random.default_rng(42)
    simulation = np.linalg.cholesky(model.covariance)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    real_scores = []
    simulated_scores = []
    for length in lengths:
        real_scores.append([])
        simulated_scores.append([])
        for stop in range(2 * length, model.num_times + 1, 2 * length):
            first = StatisticalFactorModel(model.num_factors).fit(
                model.returns[:, stop - 2 * length : stop - length]
            )
            second = StatisticalFactorModel(model.num_factors).fit(
                model.returns[:, stop - length : stop]
            )
            real_scores[-1].append(
                RiskFactorDiagnostics.subspace_comparison(
                    first.loadings, second.loadings
                ).mean_squared_cosine
            )
        for _ in range(20):
            sample = simulation @ rng.standard_normal((model.num_assets, length))
            estimate = StatisticalFactorModel(model.num_factors).fit(sample)
            diagnostics = RiskFactorDiagnostics(estimate.covariance, estimate.loadings)
            simulated_scores[-1].append(
                diagnostics.factor_recovery(model.loadings).mean_squared_cosine
            )

    ratios = model.num_assets / np.array(lengths)
    for values, label in (
        (real_scores, "Adjacent historical windows"),
        (simulated_scores, "Simulation vs. population"),
    ):
        bounds = np.array([np.percentile(scores, (25, 75)) for scores in values])
        (line,) = ax.plot(
            ratios, [np.median(scores) for scores in values], marker="o", label=label
        )
        ax.fill_between(ratios, bounds[:, 0], bounds[:, 1], color=line.get_color(), alpha=0.15)
    ax.set(xlabel=r"Dimensional ratio $N/T$", ylabel="Mean squared cosine", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    save_figure(fig, "statistical_rfm/statistical_rfm_dimensional_stability.png")


def plot_factor_returns(dates: pd.DatetimeIndex, scores: np.ndarray) -> None:
    fig, axes = plt.subplots(
        len(scores), 1, figsize=(9, 1.65 * len(scores) + 1), sharex=True, squeeze=False
    )
    for index, (ax, series) in enumerate(zip(axes[:, 0], scores, strict=True), start=1):
        ax.plot(dates, np.cumsum(series), linewidth=0.65)
        ax.axhline(0, color="0.45", linewidth=0.6)
        ax.set_ylabel(f"Factor {index}")
    fig.suptitle("Cumulative factor scores from rolling PPCA")
    fig.supylabel("Cumulative latent score")
    save_figure(fig, "statistical_rfm/statistical_rfm_factor_returns.png")


def evaluate_rolling_windows(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    num_factors: int,
    max_factors: int,
    train_sessions: int,
    test_sessions: int,
) -> None:
    starts = list(range(train_sessions, returns.shape[1] - test_sessions + 1, test_sessions))
    num_validation = len(starts) // 2
    ranks = np.arange(1, max_factors + 1)
    scores = np.empty((num_validation, max_factors))
    risk = {"PPCA": [], "Diagonal": []}
    stability_dates = []
    angles = []
    previous_loadings = None
    previous_basis = None
    factor_scores = []

    # Keep the last partial block for daily factor scores.
    for window, start in enumerate(range(train_sessions, returns.shape[1], test_sessions)):
        training = returns[:, start - train_sessions : start]
        training_mean = training.mean(axis=1, keepdims=True)
        held_out = returns[:, start : start + test_sessions] - training_mean
        model = StatisticalFactorModel(num_factors).fit(training)
        diagnostics = RiskFactorDiagnostics(model.covariance, model.loadings)
        norms = np.linalg.norm(model.loadings, axis=0)
        if np.any(norms == 0):
            raise ValueError(
                f"PPCA loadings lose rank in the window ending {dates[start - 1]:%Y-%m-%d}."
            )
        basis = model.loadings / norms
        if previous_basis is None:
            anchors = basis[np.argmax(np.abs(basis), axis=0), np.arange(num_factors)]
            rotation = np.diag(np.sign(anchors))
        else:
            left, _, right = np.linalg.svd(basis.T @ previous_basis)
            rotation = left @ right
        previous_basis = basis @ rotation
        raw_scores = np.linalg.lstsq(model.loadings, held_out, rcond=None)[0]
        factor_scores.append(rotation.T @ raw_scores)

        if previous_loadings is not None:
            comparison = diagnostics.factor_stability(previous_loadings)
            stability_dates.append(dates[start - 1])
            angles.append(comparison.largest_principal_angle_degrees)
        previous_loadings = model.loadings

        if window < num_validation:
            for rank in ranks:
                rank_diagnostics = diagnostics
                if rank != num_factors:
                    candidate = StatisticalFactorModel(int(rank)).fit(training)
                    rank_diagnostics = RiskFactorDiagnostics(
                        candidate.covariance, candidate.loadings
                    )
                scores[window, rank - 1] = (
                    rank_diagnostics.oos_gaussian_nll(held_out) / model.num_assets
                )
        elif window < len(starts):
            diagonal = RiskFactorDiagnostics(np.diag(training.var(axis=1)), model.loadings)
            for label, estimator in (("PPCA", diagnostics), ("Diagonal", diagonal)):
                risk[label].append(estimator.portfolio_risk(held_out)["equal_weight"])

    plot_factor_returns(dates[train_sessions:], np.concatenate(factor_scores, axis=1))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for score in scores:
        ax.plot(ranks, score, color="0.65", linewidth=0.8, alpha=0.5)
    mean_score = scores.mean(axis=0)
    ax.plot(ranks, mean_score, marker="o", label="Validation mean")
    ax.axvline(num_factors, color="0.25", linestyle="--", label="Configured rank")
    ax.set(xlabel="Number of factors", ylabel="Gaussian negative log-likelihood per asset")
    ax.legend(fontsize=8)
    save_figure(fig, "statistical_rfm/statistical_rfm_rank_validation.png")

    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    limit = 0.0
    for label, estimates in risk.items():
        predicted = 100 * np.sqrt([estimate.predicted_variance for estimate in estimates])
        realized = 100 * np.sqrt([estimate.realized_variance for estimate in estimates])
        limit = max(limit, float(predicted.max()), float(realized.max()))
        ax.scatter(predicted, realized, label=label, alpha=0.75)
        ratios = [1 + estimate.relative_error for estimate in estimates]
        print(
            f"{label} test portfolio: median realized-to-predicted variance ratio={np.median(ratios):.2f}."
        )
    ax.plot([0, limit], [0, limit], color="0.35", linestyle="--", label="Perfect calibration")
    ax.set(
        xlabel="Predicted daily volatility (%)",
        ylabel="Realized daily volatility (%)",
        xlim=(0, limit * 1.05),
        ylim=(0, limit * 1.05),
        aspect="equal",
    )
    ax.legend(fontsize=8)
    save_figure(fig, "statistical_rfm/statistical_rfm_portfolio_risk.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(stability_dates, angles, marker="o", markersize=3)
    ax.axvline(
        dates[starts[num_validation] - 1],
        color="0.35",
        linestyle="--",
        label="First test forecast",
    )
    ax.set(xlabel="Training window end", ylabel="Largest principal angle (degrees)")
    ax.legend(fontsize=8)
    save_figure(fig, "statistical_rfm/statistical_rfm_factor_stability.png")

    print(
        f"Rank validation: {num_validation} windows, lowest mean score at rank "
        f"{ranks[np.argmin(mean_score)]}. Portfolio test: {len(starts) - num_validation} windows."
    )


def run_experiment(
    prices: pd.DataFrame,
    num_factors: int,
    max_factors: int,
    train_sessions: int,
    test_sessions: int,
    max_fill_sessions: int = 5,
) -> None:
    fitted_prices, recovered_assets, filled_observations = prepare_price_panel(
        prices, max_fill_sessions
    )
    returns = np.diff(np.log(fitted_prices.to_numpy()), axis=1)
    dates = pd.DatetimeIndex(prices.columns[1:])
    max_factors = min(max_factors, returns.shape[0] - 1, train_sessions - 2)
    if not 1 <= num_factors <= max_factors:
        raise ValueError(
            "The configured rank must fit within the rank grid, asset count, and training window."
        )
    if test_sessions < 2 or returns.shape[1] < train_sessions + 4 * test_sessions:
        raise ValueError(
            "Need at least two validation and two test blocks after the initial training window."
        )

    model = StatisticalFactorModel(num_factors).fit(returns)
    diagnostics = RiskFactorDiagnostics(model.covariance, model.loadings)
    residuals = diagnostics.reconstruct(model.returns, model.factor_returns)
    explained = diagnostics.explanatory_power(model.returns, model.factor_returns)["r2"]
    print(
        f"Fitted {model.num_factors} factors to {model.num_assets}/{len(prices)} assets "
        f"and {model.num_times} daily log returns per asset "
        f"({dates[0]:%Y-%m-%d} through {dates[-1]:%Y-%m-%d})."
    )
    print(f"In-sample return variation explained: {explained:.1%}")
    print(
        f"Forward-filling retained {recovered_assets} additional assets "
        f"and filled {filled_observations} VWAP observations in the fitted panel "
        f"(limit: {max_fill_sessions} sessions)."
    )
    print(f"Isotropic residual volatility per session: {np.sqrt(model.residual_variance):.2%}")

    with plt.rc_context(LATEX_STYLE):
        plot_spectrum(model)
        plot_loadings(model, fitted_prices.index)
        plot_residual_structure(model, diagnostics, residuals)
        plot_asset_reconstructions(
            fitted_prices,
            returns,
            residuals,
            "statistical_rfm/statistical_rfm_reconstruction.png",
        )
        plot_dimensional_stability(model)
        evaluate_rolling_windows(
            returns, dates, num_factors, max_factors, train_sessions, test_sessions
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Fit and evaluate a PPCA risk model on LSEG VWAPs.")
    parser.add_argument("--start-date", type=Date.fromisoformat, default=Date(2022, 1, 3))
    parser.add_argument("--end-date", type=Date.fromisoformat, default=Date(2026, 6, 30))
    parser.add_argument("--num-factors", type=int, default=5)
    parser.add_argument(
        "--max-factors", type=int, default=10, help="Highest rank to validate."
    )
    parser.add_argument("--train-sessions", type=int, default=252)
    parser.add_argument("--test-sessions", type=int, default=21)
    parser.add_argument(
        "--max-fill-sessions",
        type=int,
        default=5,
        help="Maximum consecutive missing VWAPs to forward-fill. Use 0 for complete cases.",
    )
    parser.add_argument(
        "--parquet", type=Path, help="Local LSEG snapshot with a date/asset index."
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date must be on or before --end-date")
    if args.max_fill_sessions < 0:
        parser.error("--max-fill-sessions must be nonnegative")
    return args


if __name__ == "__main__":
    args = parse_args()
    fill_limit = args.max_fill_sessions
    if args.parquet is None:
        raw_prices, classifications = read_classified_prices(args.start_date, args.end_date)
        prices, _ = prepare_classified_panel(raw_prices, classifications, fill_limit)
        momentum, *_ = build_factors()
        prices = prices.iloc[:, max(factor.window for factor in momentum) :]
        fill_limit = 0
    else:
        ref_prices = pd.read_parquet(args.parquet)
        snapshot_dates = pd.DatetimeIndex(ref_prices.index.get_level_values("date")).date
        ref_prices = ref_prices.loc[
            (snapshot_dates >= args.start_date) & (snapshot_dates <= args.end_date)
        ]
        prices = ref_prices["vwap"].unstack("date").sort_index(axis=1)

    run_experiment(
        prices,
        args.num_factors,
        args.max_factors,
        args.train_sessions,
        args.test_sessions,
        fill_limit,
    )
