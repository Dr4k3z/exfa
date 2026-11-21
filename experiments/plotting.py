"""
Provide a shared LaTeX-style theme and PNG export for experiment plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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
        fig.savefig(OUTPUT_DIR / filename, format="png", dpi=200)
    finally:
        plt.close(fig)
