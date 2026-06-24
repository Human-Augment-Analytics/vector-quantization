"""Optimality counterexamples (research-group notes, 2026-06-24): concrete proofs
that the heuristics are not optimal in the worst case, plus a test of whether the
proposed FFD fix 'suffices' for size-8 bins.

  C1. 1-D k-means is NOT optimal: {-0.99,0,0.99,2}, k=2 has two stable Lloyd
      fixed points (distortion 1.96 vs 1.00); Lloyd can land on the worse one.
  C2. FFD is NOT optimal: {4,3,3,2,2,2}, cap 8 -> FFD=3 but OPT=2.
  C3. Does the proposed fix ('odd #4s & >=2 twos -> pack 4+2+2') suffice? Sweep:
      plain FFD vs naive-fix vs min(FFD,fix) vs exact ILP, % optimal.

Writes results/_figs/theory_counterexamples.png.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

OUT = Path(os.environ.get("VQ_FIG_DIR", "results/_figs")); OUT.mkdir(parents=True, exist_ok=True)
CAP = 8
_CFG = []
def _gen(st, rem, cur):
    if cur: _CFG.append(tuple(cur))
    for s in range(st, rem + 1): cur.append(s); _gen(s, rem - s, cur); cur.pop()
_gen(1, CAP, [])


def ffd(w, cap=CAP):
    bins = []
    for x in sorted(w, reverse=True):
        for i in range(len(bins)):
            if bins[i] >= x: bins[i] -= x; break
        else: bins.append(cap - x)
    return len(bins)

def ffd_layout(w, cap=CAP):
    bins = []
    for x in sorted(w, reverse=True):
        for b in bins:
            if cap - sum(b) >= x: b.append(x); break
        else: bins.append([x])
    return bins

def opt(w, cap=CAP):
    c = Counter(w); sizes = [s for s in range(1, cap + 1) if c[s]]
    if not sizes: return 0
    A = np.array([[cfg.count(s) for cfg in _CFG] for s in sizes])
    r = milp(c=np.ones(len(_CFG)), constraints=LinearConstraint(A, lb=[c[s] for s in sizes], ub=np.inf),
             integrality=np.ones(len(_CFG)), bounds=Bounds(0, np.inf))
    return int(round(r.fun))

def mod_ffd(w, cap=CAP):
    c = Counter(w); reserved = 0
    if c[4] % 2 == 1 and c[2] >= 2: c[4] -= 1; c[2] -= 2; reserved += 1
    return reserved + ffd([s for s, n in c.items() for _ in range(n)], cap)

def best_ffd(w):
    return min(ffd(w), mod_ffd(w))


def _draw_bins(ax, layout, x0, title, color):
    for j, b in enumerate(layout):
        y = 0
        for item in b:
            ax.bar(x0 + j, item, bottom=y, width=0.8, color=color, edgecolor="white")
            ax.text(x0 + j, y + item / 2, str(item), ha="center", va="center", fontsize=8, color="white")
            y += item
    ax.axhline(CAP, color="#888", ls="--", lw=0.8)
    ax.text(x0 + (len(layout) - 1) / 2, CAP + 0.4, title, ha="center", fontsize=9)


def main():
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(16, 4.8))

    # ---- C1: 1-D k-means trap ----
    ax = fig.add_subplot(1, 3, 1)
    pts = [-0.99, 0.0, 0.99, 2.0]
    sols = [(("trap: centroids {0, 2}", 1.96), [[-0.99, 0, 0.99], [2.0]], [0.0, 2.0], "#d62728"),
            (("optimal: centroids {-0.5, 1.5}", 1.00), [[-0.99, 0], [0.99, 2.0]], [-0.5, 1.5], "#2ca02c")]
    for row, ((lab, dist), clusters, cents, col) in enumerate(sols):
        y = 1.0 - row
        cols = ["#1f77b4", "#ff7f0e"]
        for ci, cl in enumerate(clusters):
            ax.scatter(cl, [y] * len(cl), s=70, color=cols[ci], zorder=3)
        ax.scatter(cents, [y] * len(cents), marker="x", s=120, color=col, lw=2.5, zorder=4)
        ax.text(2.35, y, f"D={dist:.2f}", va="center", fontsize=9, color=col)
        ax.text(-1.25, y, lab, va="center", ha="right", fontsize=7.5)
    ax.set_xlim(-2.6, 3.0); ax.set_ylim(-0.6, 1.6); ax.set_yticks([])
    ax.set_xlabel("value"); ax.set_title("C1 — 1-D k-means is not optimal\n{-0.99, 0, 0.99, 2}: two stable Lloyd fixed points")
    ax.grid(alpha=0.2, axis="x")

    # ---- C2: FFD counterexample ----
    ax = fig.add_subplot(1, 3, 2)
    ce = [4, 3, 3, 2, 2, 2]
    _draw_bins(ax, ffd_layout(ce), 0, f"FFD = {ffd(ce)} bytes", "#d62728")
    _draw_bins(ax, [[4, 2, 2], [3, 3, 2]], 4, f"optimal = {opt(ce)} bytes", "#2ca02c")
    ax.set_xticks([]); ax.set_ylabel("bits in byte (cap 8)")
    ax.set_ylim(0, CAP + 1.5)
    ax.set_title("C2 — FFD is not optimal\nitems {4,3,3,2,2,2}, 8-bit bytes")

    # ---- C3: does the fix suffice? ----
    ax = fig.add_subplot(1, 3, 3)
    N = 20000
    rates = {"plain FFD": 0, "naive 4+2+2 fix": 0, "min(FFD, fix)": 0, "exact ILP": N}
    for _ in range(N):
        w = rng.integers(1, 9, int(rng.integers(2, 30))).tolist()
        o = opt(w)
        rates["plain FFD"] += ffd(w) == o
        rates["naive 4+2+2 fix"] += mod_ffd(w) == o
        rates["min(FFD, fix)"] += best_ffd(w) == o
    labels = list(rates); vals = [100 * rates[k] / N for k in labels]
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v - 1.2, f"{v:.2f}%", ha="center", va="top", color="white", fontsize=8)
    ax.set_ylim(95, 100.3); ax.set_ylabel("% of instances optimal")
    ax.set_title(f"C3 — does the FFD fix suffice?\n{N} random cap-8 instances")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Optimality counterexamples — heuristics are not worst-case optimal", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "theory_counterexamples.png", dpi=130); plt.close(fig)
    print(f"  rates: " + "  ".join(f"{k}={100*rates[k]/N:.2f}%" for k in rates))
    print("=== wrote theory_counterexamples.png ===")


if __name__ == "__main__":
    main()
