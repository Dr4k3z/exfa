from argparse import ArgumentParser, Namespace
from datetime import date as Date
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exfa.diagnostics import (
    RiskFactorDiagnostics,
    compare_factor_subspaces,
    compute_group_overlap,
    score_covariance_forecast,
    score_factor_reconstruction,
    score_factor_spanning,
)
from exfa.models import StatisticalFactorModel

if TYPE_CHECKING or __package__:
    from .fundamental_rfm import fit_fundamental_model
    from .market_panel import prepare_classified_panel, read_classified_prices
    from .plotting import LATEX_STYLE, OUTPUT_DIR, save_figure
else:
    from fundamental_rfm import fit_fundamental_model
    from market_panel import prepare_classified_panel, read_classified_prices
    from plotting import LATEX_STYLE, OUTPUT_DIR, save_figure

OUTPUT = "compare_rfm"
FUNDAMENTAL = "Fundamental"
PPCA = "PPCA"
SHARED_DIAGONAL = "PPCA + diagonal residuals"
BASELINE = "Diagonal"
HYBRID = "Hybrid"


def build_portfolios(sectors: np.ndarray, momentum: np.ndarray) -> dict[str, np.ndarray]:
    portfolios = {"Equal weight": np.full(len(sectors), 1 / len(sectors))}
    neutral = momentum.copy()
    for sector in np.unique(sectors):
        members = sectors == sector
        portfolios[f"Sector: {sector}"] = members / members.sum()
        neutral[members] -= neutral[members].mean()
    gross = np.abs(neutral).sum()
    if gross > 0:
        portfolios["Sector-neutral 126-day momentum"] = neutral / gross
    return portfolios


def compute_hybrid_covariance(
    loadings: np.ndarray,
    factor_returns: np.ndarray,
    residuals: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add PPCA residual factors and estimate the joint factor covariance.
    """

    latent = StatisticalFactorModel(rank).fit(residuals)
    combined_loadings = np.column_stack((loadings, latent.loadings))
    combined_returns = np.vstack((factor_returns, latent.factor_returns))
    covariance = combined_loadings @ np.cov(combined_returns) @ combined_loadings.T
    covariance += np.diag(latent.residuals.var(axis=1, ddof=1))
    return covariance, combined_loadings


def plot_results(
    forecasts: pd.DataFrame,
    portfolio: pd.DataFrame,
    overlap: pd.DataFrame,
    spanning: pd.DataFrame,
    reconstruction: pd.DataFrame,
    residuals: dict[str, np.ndarray],
    classifications: pd.DataFrame,
) -> None:
    test = forecasts.loc[forecasts["Split"].eq("test")]
    scores = test.pivot(index="Date", columns="Model", values="Gaussian NLL per asset")
    fig, ax = plt.subplots(figsize=(9, 4))
    (scores.drop(columns=BASELINE).sub(scores[BASELINE], axis=0)).plot(ax=ax, linewidth=1)
    ax.axhline(0, color="0.35", linestyle=":")
    ax.set_ylabel("Gaussian NLL difference vs. diagonal")
    save_figure(fig, f"{OUTPUT}/covariance_scores.png")

    equal_weight = portfolio.loc[portfolio["Portfolio"].eq("Equal weight")]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    limit = 0.0
    for name, frame in equal_weight.groupby("Model"):
        predicted = 100 * np.sqrt(frame["Predicted variance"])
        realized = 100 * np.sqrt(frame["Realized variance"])
        limit = max(limit, float(predicted.max()), float(realized.max()))
        ax.scatter(predicted, realized, label=name, alpha=0.65, s=15)
    ax.plot([0, limit], [0, limit], ":", color="0.35")
    ax.set(
        xlabel="Predicted daily volatility (%)",
        ylabel="Realized daily volatility (%)",
        aspect="equal",
    )
    ax.legend(fontsize=7)
    save_figure(fig, f"{OUTPUT}/portfolio_calibration.png")

    fig, (ax, groups) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    overlap.set_index("Date")[["PPCA in fundamental", "Fundamental in PPCA"]].plot(ax=ax)
    ax.set_ylabel("Fraction of loading space")
    names = [name for name in overlap if name.startswith("Increment: ")]
    increments = overlap.set_index("Date")[names].rename(
        columns=lambda name: name.removeprefix("Increment: ")
    )
    increments.plot.area(ax=groups)
    groups.set_ylabel("Increment in PPCA overlap")
    save_figure(fig, f"{OUTPUT}/loading_overlap.png")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * spanning["Factor"].nunique())))
    medians = spanning.groupby("Factor").agg({"Held-out R²": "median"}).sort_values("Held-out R²")
    ax.barh(medians.index, medians["Held-out R²"], height=0.75)
    ax.axvline(0, color="0.35", linewidth=0.7)
    ax.set_xlabel("Median held-out R²")
    save_figure(fig, f"{OUTPUT}/factor_spanning.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, metric in zip(
        axes.flat,
        ("R²", "RMSE", "Absolute residual correlation", "Largest residual eigenvalue"),
        strict=True,
    ):
        reconstruction.pivot(index="Date", columns="Model", values=metric).plot(
            ax=ax, legend=False
        )
        ax.set_ylabel(metric)
    axes[0, 0].legend(fontsize=7)
    save_figure(fig, f"{OUTPUT}/reconstruction.png")

    fig, axes = plt.subplots(1, len(residuals), figsize=(10, 3.6))
    order = np.argsort(classifications["industry"].to_numpy(), kind="stable")
    for ax, (name, values) in zip(axes, residuals.items(), strict=True):
        image = ax.imshow(np.corrcoef(values[order]), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(title=name, xlabel="Assets ordered by sector")
        fig.colorbar(image, ax=ax, shrink=0.75)
    save_figure(fig, f"{OUTPUT}/residual_correlations.png")


def run_comparison(
    prices: pd.DataFrame,
    classifications: pd.DataFrame,
    train_sessions: int,
    test_sessions: int,
    ppca_rank: int,
    max_hybrid_rank: int,
    benchmark_returns: pd.DataFrame | None,
) -> None:
    fundamental, groups, aligned = fit_fundamental_model(prices, classifications)
    returns = fundamental.returns
    dates = pd.DatetimeIndex(aligned.columns[1:])
    if (
        min(train_sessions, test_sessions) < 2
        or returns.shape[1] < train_sessions + 4 * test_sessions
    ):
        raise ValueError("Need a training window, two validation blocks, and two test blocks.")
    rank_limit = min(returns.shape[0], train_sessions - 1)
    if not 1 <= ppca_rank < rank_limit or not 1 <= max_hybrid_rank < rank_limit:
        raise ValueError(
            "PPCA and hybrid ranks must be positive and below the training dimensions."
        )

    names = [f"{group}: {name}" for group, labels in groups.items() for name in labels]
    factor_returns = fundamental.factor_returns
    residuals = fundamental.residuals
    loading_history = fundamental.loadings_history
    targets = factor_returns
    if benchmark_returns is not None:
        external = benchmark_returns.reindex(dates).to_numpy(dtype=float)
        if not np.isfinite(external).all():
            raise ValueError("External benchmark returns must cover all aligned return dates.")
        names.extend(f"Benchmark: {name}" for name in benchmark_returns.columns)
        targets = np.vstack((targets, external.T))
    group_indices = {}
    offset = 0
    for group, group_labels in groups.items():
        group_indices[group] = slice(offset, offset + len(group_labels))
        offset += len(group_labels)

    forecast_rows, portfolio_rows, overlap_rows, spanning_rows, reconstruction_rows = (
        [],
        [],
        [],
        [],
        [],
    )
    all_residuals: dict[str, list[np.ndarray]] = {
        name: [] for name in (FUNDAMENTAL, PPCA, HYBRID)
    }
    starts = list(range(train_sessions, returns.shape[1] - test_sessions + 1, test_sessions))
    validation_count = len(starts) // 2
    candidate_scores = {rank: [] for rank in range(max_hybrid_rank + 1)}
    selected_rank = 0
    sectors = classifications["industry"].to_numpy()
    for index, start in enumerate(starts):
        training = slice(start - train_sessions, start)
        test = slice(start, start + test_sessions)
        is_validation = index < validation_count
        if index == validation_count:
            selected_rank = min(
                candidate_scores, key=lambda rank: np.mean(candidate_scores[rank])
            )
        block = {"Date": dates[start], "Split": "validation" if is_validation else "test"}
        raw_training = returns[:, training]
        held_out = returns[:, test] - raw_training.mean(axis=1, keepdims=True)
        ppca = StatisticalFactorModel(ppca_rank).fit(raw_training)
        fund_loadings = loading_history[:, :, start]
        train_factors = factor_returns[:, training]
        train_residuals = residuals[:, training]
        fund_covariance = fund_loadings @ np.cov(train_factors) @ fund_loadings.T + np.diag(
            train_residuals.var(axis=1, ddof=1)
        )
        ppca_diagonal = ppca.loadings @ ppca.loadings.T + np.diag(
            ppca.residuals.var(axis=1, ddof=1)
        )
        forecasts = {
            FUNDAMENTAL: (fund_covariance, fund_loadings),
            PPCA: (ppca.covariance, ppca.loadings),
            SHARED_DIAGONAL: (ppca_diagonal, ppca.loadings),
            BASELINE: (np.diag(raw_training.var(axis=1, ddof=1)), fund_loadings),
        }
        # Freeze forecast-origin portfolios using only known classifications and loadings.
        weights = build_portfolios(sectors, fund_loadings[:, 1])
        if is_validation:
            forecasts.update(
                {
                    f"Hybrid rank {rank}": compute_hybrid_covariance(
                        fund_loadings, train_factors, train_residuals, rank
                    )
                    for rank in range(1, max_hybrid_rank + 1)
                }
            )
        else:
            forecasts[HYBRID] = (
                compute_hybrid_covariance(
                    fund_loadings, train_factors, train_residuals, selected_rank
                )
                if selected_rank
                else forecasts[FUNDAMENTAL]
            )

        for name, (covariance, loadings) in forecasts.items():
            score = score_covariance_forecast(
                RiskFactorDiagnostics(covariance, loadings), held_out, weights
            )
            forecast_rows.append(
                {
                    **block,
                    "Model": name,
                    "Gaussian NLL per asset": score.gaussian_nll_per_asset,
                }
            )
            if is_validation:
                if name == FUNDAMENTAL:
                    candidate_scores[0].append(score.gaussian_nll_per_asset)
                elif name.startswith("Hybrid rank "):
                    candidate_scores[int(name.rsplit(" ", 1)[1])].append(
                        score.gaussian_nll_per_asset
                    )
            else:
                for portfolio, metric in score.portfolios.items():
                    portfolio_rows.append(
                        {
                            **block,
                            "Model": name,
                            "Portfolio": portfolio,
                            "Predicted variance": metric.risk.predicted_variance,
                            "Realized variance": metric.risk.realized_variance,
                            "Variance ratio": metric.variance_ratio,
                            "QLIKE": metric.qlike,
                        }
                    )
        if is_validation:
            continue

        overlap = compare_factor_subspaces(ppca.loadings, fund_loadings)
        increment = compute_group_overlap(
            ppca.loadings,
            {
                group: fund_loadings[:, group_indices[group]]
                for group in ("Intercept", "Country", "Sector", "Momentum")
            },
        )
        overlap_rows.append(
            {
                **block,
                "PPCA in fundamental": overlap.first_explained,
                "Fundamental in PPCA": overlap.second_explained,
                "PPCA rank": overlap.ranks[0],
                "Fundamental rank": overlap.ranks[1],
                **{
                    f"Angle {i}": angle
                    for i, angle in enumerate(overlap.principal_angles_degrees, start=1)
                },
                **{f"Increment: {group}": value for group, value in increment.items()},
            }
        )
        test_scores = np.linalg.lstsq(ppca.loadings, held_out, rcond=None)[0]
        spanning = score_factor_spanning(
            ppca.factor_returns, test_scores, targets[:, training], targets[:, test]
        )
        spanning_rows.extend(
            {**block, "Factor": name, "Held-out R²": score}
            for name, score in zip(names, spanning, strict=True)
        )
        for name in all_residuals:
            forecast_loadings = forecasts[name][1]
            if name != PPCA:
                daily_loadings = loading_history[:, :, test]
                if name == HYBRID and selected_rank:
                    latent = forecast_loadings[:, -selected_rank:]
                    fixed = np.broadcast_to(
                        latent[:, :, None], (len(latent), selected_rank, test_sessions)
                    )
                    daily_loadings = np.concatenate((daily_loadings, fixed), axis=1)
                forecast_loadings = daily_loadings
            reconstruction, residual = score_factor_reconstruction(held_out, forecast_loadings)
            all_residuals[name].append(residual)
            reconstruction_rows.append(
                {
                    **block,
                    "Model": name,
                    "R²": reconstruction.r2,
                    "RMSE": reconstruction.rmse,
                    "Effective rank": reconstruction.effective_rank,
                    "Absolute residual correlation": reconstruction.mean_absolute_residual_correlation,
                    "Largest residual eigenvalue": reconstruction.largest_residual_eigenvalue,
                }
            )

    output = OUTPUT_DIR / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "forecasts": pd.DataFrame(forecast_rows),
        "portfolio_scores": pd.DataFrame(portfolio_rows),
        "loading_overlap": pd.DataFrame(overlap_rows),
        "factor_spanning": pd.DataFrame(spanning_rows),
        "reconstruction": pd.DataFrame(reconstruction_rows),
    }
    test = tables["forecasts"].loc[lambda frame: frame["Split"].eq("test")]
    paired = test.pivot(index="Date", columns="Model", values="Gaussian NLL per asset")
    summary = paired.mean().rename("Mean NLL").to_frame()
    summary["Mean paired difference vs. fundamental"] = paired.sub(
        paired[FUNDAMENTAL], axis=0
    ).mean()
    summary["Win fraction vs. fundamental"] = paired.lt(paired[FUNDAMENTAL], axis=0).mean()
    tables["test_score_summary"] = summary.reset_index()
    validation = {rank: float(np.mean(scores)) for rank, scores in candidate_scores.items()}
    tables["hybrid_selection"] = pd.DataFrame(
        {
            "Additional factors": list(validation),
            "Mean validation NLL": list(validation.values()),
            "Selected": [rank == selected_rank for rank in validation],
        }
    )
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    print(
        f"Matched {len(prices)} assets and {len(dates)} returns "
        f"({dates[0]:%Y-%m-%d} through {dates[-1]:%Y-%m-%d}). "
        f"Hybrid rank {selected_rank} selected on {validation_count} validation blocks; "
        f"{len(starts) - validation_count} test blocks."
    )
    print(summary.to_string())
    with plt.rc_context(LATEX_STYLE):
        plot_results(
            tables["forecasts"],
            tables["portfolio_scores"],
            tables["loading_overlap"],
            tables["factor_spanning"],
            tables["reconstruction"],
            {name: np.concatenate(values, axis=1) for name, values in all_residuals.items()},
            classifications,
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Compare fundamental, statistical, and hybrid risk models."
    )
    parser.add_argument("--start-date", type=Date.fromisoformat, default=Date(2022, 1, 3))
    parser.add_argument("--end-date", type=Date.fromisoformat, default=Date(2026, 6, 30))
    parser.add_argument("--train-sessions", type=int, default=252)
    parser.add_argument("--test-sessions", type=int, default=21)
    parser.add_argument("--max-fill-sessions", type=int, default=5)
    parser.add_argument("--ppca-rank", type=int, default=5)
    parser.add_argument("--max-hybrid-rank", type=int, default=3)
    parser.add_argument(
        "--benchmark-returns",
        type=Path,
        help="Optional CSV of daily external returns with a date column.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prices, classifications = read_classified_prices(args.start_date, args.end_date)
    prices, classifications = prepare_classified_panel(
        prices, classifications, args.max_fill_sessions
    )
    benchmarks = (
        pd.read_csv(args.benchmark_returns, index_col="date", parse_dates=["date"])
        if args.benchmark_returns
        else None
    )
    run_comparison(
        prices,
        classifications,
        args.train_sessions,
        args.test_sessions,
        args.ppca_rank,
        args.max_hybrid_rank,
        benchmarks,
    )
