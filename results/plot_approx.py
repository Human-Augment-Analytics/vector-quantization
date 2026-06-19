"""Charts for the 3 approximation experiments (professor's heuristic-vs-optimal study).

Reads the CSVs written by the approx experiments and renders, for a dataset:
  exp1_codebook.png   Lloyd/DP codebook ratio vs bits (+ per-dim spread)
  exp2_alloc.png      greedy & Bennett vs optimal allocation ratio + distortion
  exp3_packing.png    FFD vs optimal vs LB byte packing (ratio + bytes)
  approx_combined.png HEADLINE: each stage's heuristic-vs-optimal ratio vs bpd

Headless (Agg). Path-driven so it runs in the sbatch job AND on pulled results.

Usage:
    python plot_approx.py --base results --out results/_figs --dataset dbpedia
    # --base must contain approx_codebook/, approx_alloc/, approx_packing/ subdirs
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPTIMAL_C = "#444444"


def _read(base: Path, name: str) -> pd.DataFrame | None:
    p = base / name / f"{name}.csv"
    if not p.exists():
        print(f"  (skip) {p} not found")
        return None
    return pd.read_csv(p)


def plot_exp1(base: Path, out: Path, ds: str):
    df = _read(base, "approx_codebook")
    if df is None:
        return
    summ = pd.read_csv(base / "approx_codebook" / "approx_codebook_summary.csv") \
        if (base / "approx_codebook" / "approx_codebook_summary.csv").exists() else None
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) summary lines vs bits
    if summ is not None:
        for col, lab in [("ratio_mean", "mean"), ("ratio_median", "median"),
                         ("ratio_p95", "p95"), ("ratio_varweighted", "var-weighted")]:
            ax[0].plot(summ["bits"], summ[col], marker="o", label=lab)
    ax[0].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1, label="optimal (DP)")
    ax[0].set_xlabel("bits / dim"); ax[0].set_ylabel("Lloyd MSE / DP-optimal MSE")
    ax[0].set_title(f"Exp 1 — codebook: cumsum-kmeans vs DP-optimal\n({ds})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # (b) per-dim spread: ratio distribution per bit (box)
    bits = sorted(df["bits"].unique())
    data = [df.loc[df.bits == b, "ratio"].to_numpy() for b in bits]
    ax[1].boxplot(data, positions=bits, widths=0.6, showfliers=False)
    ax[1].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1)
    ax[1].set_xlabel("bits / dim"); ax[1].set_ylabel("ratio (per-dim)")
    ax[1].set_title("Per-dim spread of the codebook ratio")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "exp1_codebook.png", dpi=130); plt.close(fig)
    print("  wrote exp1_codebook.png")


def plot_exp2(base: Path, out: Path, ds: str):
    df = _read(base, "approx_alloc")
    if df is None:
        return
    df = df.sort_values("avg_bits")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(df["avg_bits"], df["ratio_greedy_opt"], marker="o", label="greedy")
    ax[0].plot(df["avg_bits"], df["ratio_bennett_opt"], marker="s", label="analytic Bennett")
    ax[0].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1, label="optimal (DP)")
    ax[0].set_xlabel("avg bits / dim"); ax[0].set_ylabel("distortion / optimal")
    ax[0].set_title(f"Exp 2 — allocation: heuristic vs optimal\n({ds})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    for col, lab, mk in [("dist_optimal", "optimal", "o"), ("dist_greedy", "greedy", "x"),
                         ("dist_bennett", "Bennett", "s")]:
        ax[1].plot(df["avg_bits"], df[col], marker=mk, label=lab)
    ax[1].set_yscale("log"); ax[1].set_xlabel("avg bits / dim")
    ax[1].set_ylabel("total distortion (empirical MSE sum)")
    ax[1].set_title("Allocation distortion vs bit budget")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "exp2_alloc.png", dpi=130); plt.close(fig)
    print("  wrote exp2_alloc.png")


def plot_exp3(base: Path, out: Path, ds: str):
    df = _read(base, "approx_packing")
    if df is None:
        return
    df = df.sort_values("avg_bits")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(df["avg_bits"], df["ffd_over_opt"], marker="o", label="FFD / optimal")
    ax[0].plot(df["avg_bits"], df["ffd_bytes"] / df["lb_bytes"], marker="s",
               label="FFD / perfect-packing LB")
    ax[0].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1, label="optimal")
    ax[0].set_xlabel("avg bits / dim"); ax[0].set_ylabel("bytes ratio")
    ax[0].set_title(f"Exp 3 — packing: FFD vs optimal vs LB\n({ds})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].annotate("LB is unreachable when codes are large\n(FFD == OPT throughout)",
                   xy=(0.5, 0.92), xycoords="axes fraction", fontsize=7, ha="center",
                   va="top", color="#666")

    for col, lab, mk in [("opt_bytes", "optimal", "o"), ("ffd_bytes", "FFD", "x"),
                         ("lb_bytes", "perfect-pack LB", "s")]:
        ax[1].plot(df["avg_bits"], df[col], marker=mk, label=lab)
    ax[1].set_xlabel("avg bits / dim"); ax[1].set_ylabel("bytes")
    ax[1].set_title("Packed size vs bit budget")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "exp3_packing.png", dpi=130); plt.close(fig)
    print("  wrote exp3_packing.png")


def plot_combined(base: Path, out: Path, ds: str):
    """Headline: each stage's heuristic-vs-optimal ratio vs bit-rate."""
    e1 = base / "approx_codebook" / "approx_codebook_summary.csv"
    e2 = _read(base, "approx_alloc")
    e3 = _read(base, "approx_packing")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)

    if e1.exists():
        s = pd.read_csv(e1)
        ax[0].plot(s["bits"], s["ratio_mean"], marker="o", color="#1f77b4")
        ax[0].fill_between(s["bits"], s["ratio_mean"], s["ratio_p95"], alpha=0.15, color="#1f77b4")
    ax[0].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1)
    ax[0].set_title("Codebook\ncumsum-kmeans / DP-optimal"); ax[0].set_xlabel("bits/dim")
    ax[0].set_ylabel("heuristic / optimal (MSE)"); ax[0].grid(alpha=0.3)

    if e2 is not None:
        e2 = e2.sort_values("avg_bits")
        ax[1].plot(e2["avg_bits"], e2["ratio_greedy_opt"], marker="o", label="greedy", color="#2ca02c")
        ax[1].plot(e2["avg_bits"], e2["ratio_bennett_opt"], marker="s", label="Bennett", color="#d62728")
        ax[1].legend(fontsize=8)
    ax[1].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1)
    ax[1].set_title("Allocation\ngreedy / optimal"); ax[1].set_xlabel("avg bits/dim"); ax[1].grid(alpha=0.3)

    if e3 is not None:
        e3 = e3.sort_values("avg_bits")
        ax[2].plot(e3["avg_bits"], e3["ffd_over_opt"], marker="o", color="#9467bd")
    ax[2].axhline(1.0, color=OPTIMAL_C, ls="--", lw=1)
    ax[2].set_title("Packing\nFFD / optimal"); ax[2].set_xlabel("avg bits/dim"); ax[2].grid(alpha=0.3)

    fig.suptitle(f"Method-stage approximation quality vs optimal — {ds}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "approx_combined.png", dpi=130); plt.close(fig)
    print("  wrote approx_combined.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("VQ_RESULTS_DIR", "results"),
                    help="dir containing approx_codebook/ approx_alloc/ approx_packing/")
    ap.add_argument("--out", default=None, help="output dir for PNGs (default <base>/_figs)")
    ap.add_argument("--dataset", default=os.environ.get("VQ_DATASET", "dbpedia"))
    args = ap.parse_args()
    base = Path(args.base)
    out = Path(args.out) if args.out else base / "_figs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== plotting approx suite from {base} -> {out} ===")
    plot_exp1(base, out, args.dataset)
    plot_exp2(base, out, args.dataset)
    plot_exp3(base, out, args.dataset)
    plot_combined(base, out, args.dataset)
    print("=== done ===")


if __name__ == "__main__":
    main()
