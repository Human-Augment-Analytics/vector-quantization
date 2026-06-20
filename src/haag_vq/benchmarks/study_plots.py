# src/haag_vq/benchmarks/study_plots.py
"""Pareto plots: recall@k vs compression, and MSE vs compression."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def pareto_curves(df: pd.DataFrame, save_path: str | Path, ks: Tuple[int, ...] = (1, 10, 100)) -> None:
    import matplotlib.pyplot as plt

    n_panels = len(ks) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    for ax, k in zip(axes[: len(ks)], ks):
        col = f"recall_at_{k}"
        for method, g in df.groupby("method"):
            g = g.sort_values("compression_factor")
            ax.plot(g["compression_factor"], g[col], marker="o", label=str(method))
        ax.set_xlabel("compression factor")
        ax.set_ylabel(col)
        ax.set_title(f"Recall@{k} vs compression")
        ax.legend(fontsize=7)

    ax = axes[len(ks)]
    for method, g in df.groupby("method"):
        g = g.sort_values("compression_factor")
        ax.plot(g["compression_factor"], g["mse"], marker="s", label=str(method))
    ax.set_xlabel("compression factor")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("MSE vs compression")
    ax.legend(fontsize=7)

    fig.tight_layout()
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
