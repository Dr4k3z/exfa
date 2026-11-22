"""
Provide a shared LaTeX-style theme and PNG export for experiment plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.figure import Figure

LATEX_STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.prop_cycle": cycler(color=["#3B6380", "#A24B32", "#38684D", "#735C82"]),
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.55,
    "legend.frameon": False,
}

OUTPUT_DIR = Path(__file__).resolve().parent


def save_figure(fig: Figure, filename: str) -> None:
    """
    Save a PNG in the experiment directory and release the figure.
    """

    try:
        fig.tight_layout()
        destination = OUTPUT_DIR / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, format="png", dpi=200)
    finally:
        plt.close(fig)


def plot_asset_reconstructions(
    prices: pd.DataFrame,
    returns: np.ndarray,
    residuals: np.ndarray,
    filename: str,
) -> None:
    centered = returns - returns.mean(axis=1, keepdims=True)
    variation = np.sum(centered**2, axis=1)
    active = np.flatnonzero(variation > 0)
    explained = 1 - np.sum(residuals[active] ** 2, axis=1) / variation[active]
    ordered = np.argsort(explained, kind="stable")
    dates = pd.DatetimeIndex(prices.columns)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, quantile in zip(axes, (0.1, 0.5, 0.9), strict=True):
        rank = ordered[round(quantile * (len(ordered) - 1))]
        asset = active[rank]
        observed = prices.iloc[asset].to_numpy(dtype=float)
        fitted = returns[asset] - residuals[asset]
        ax.plot(dates, 100 * observed / observed[0], label="Observed VWAP")
        ax.plot(
            dates,
            100 * np.exp(np.r_[0.0, np.cumsum(fitted)]),
            "--",
            label="Factor-implied path",
        )
        ax.set_ylabel("Rebased VWAP")
        ax.set_title(
            f"{prices.index[asset]} ({quantile:.0%} quantile), {explained[rank]:.0%} explained"
        )
    axes[0].legend(fontsize=8)
    fig.suptitle("In-sample asset reconstructions")
    save_figure(fig, filename)
