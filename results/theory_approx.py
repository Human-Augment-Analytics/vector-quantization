"""THEORY-side approximation experiments (professor's list), independent of any
real dataset:

  T1. How close is k-means (Lloyd) to optimal for 1-D?
      -> on canonical distributions (Gaussian/Laplace/Uniform/Student-t), Lloyd vs
         the GLOBAL 1-D optimum (DP), MSE ratio vs bits. + a restarts ablation
         (does the high-bit gap close with more Lloyd restarts -> local optima?).

  T2. How close is greedy bit allocation to FULL optimization?
      -> greedy vs exact min-plus DP over the per-dim cost table on synthetic
         variance spectra (ratio ~ 1 because rate-distortion curves are convex),
         + the convexity mechanism and a constructed NON-convex counterexample
         where greedy is provably suboptimal (characterizes the boundary).

  T3. How close is FFD to optimal byte packing?
      -> FFD vs exact optimal (config ILP) over many RANDOM bit-width instances,
         vs the classical worst-case bound FFD <= 11/9 OPT + 6/9.

Outputs (results/_figs/): theory1_codebook_1d.png, theory2_alloc.png,
theory3_packing.png, theory_combined.png. Self-contained (synthetic inputs).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import saq

OUT = Path(os.environ.get("VQ_FIG_DIR", "results/_figs"))
RNG = np.random.default_rng(0)
OPT_C = "#444444"
MAXB = 8
NUM_BINS = 80 * (1 << MAXB)   # converged DP reference (see Exp 1 calibration)


# =============================================================== T1: 1-D codebook
def _sample(dist, n=200_000):
    if dist == "Gaussian":
        return RNG.standard_normal(n)
    if dist == "Laplace":
        return RNG.laplace(0.0, 1.0, n)
    if dist == "Uniform":
        return RNG.uniform(-1.0, 1.0, n)
    if dist == "Student-t(3)":
        return RNG.standard_t(3, n)
    raise ValueError(dist)


def theory1():
    dists = ["Gaussian", "Laplace", "Uniform", "Student-t(3)"]
    bits = list(range(1, MAXB + 1))
    res = {}          # dist -> (restarts1, restarts10) ratio arrays
    for dist in dists:
        x = _sample(dist).astype(np.float32)
        dp = saq.build_codebook_dp(x, max_bits=MAXB, num_bins=NUM_BINS)
        dp_mse = np.array([saq.codebook_mse(x, dp.codebooks[b]) for b in bits])
        rr = {}
        for restarts in (1, 10):
            o = saq.LloydOpts(); o.max_bits = MAXB; o.restarts = restarts; o.seed = 0
            ll = saq.build_codebook_lloyd(x, o)
            ll_mse = np.array([saq.codebook_mse(x, ll.codebooks[b]) for b in bits])
            rr[restarts] = ll_mse / dp_mse
        res[dist] = rr
        print(f"  T1 {dist}: r1@8={rr[1][-1]:.4f} r10@8={rr[10][-1]:.4f}", flush=True)
    return bits, res


def plot_t1(ax, bits, res):
    for dist, rr in res.items():
        line, = ax.plot(bits, rr[1], marker="o", label=f"{dist} (restarts=1)")
        ax.plot(bits, rr[10], marker="^", ls="--", color=line.get_color(),
                alpha=0.6, label=f"{dist} (restarts=10)")
    ax.axhline(1.0, color=OPT_C, ls=":", lw=1)
    ax.set_xlabel("bits / level"); ax.set_ylabel("k-means MSE / 1-D optimal (DP)")
    ax.set_title("T1 — k-means vs optimal for 1-D\n(canonical distributions)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)


# =============================================================== T2: allocation
def numpy_greedy(C, budget, max_bits):
    D = C.shape[0]; bits = np.zeros(D, dtype=np.int64)
    gain = C[:, 0] - C[:, 1]
    for _ in range(int(budget)):
        d = int(np.argmax(gain))
        if not np.isfinite(gain[d]) or gain[d] <= 0:
            break
        bits[d] += 1
        gain[d] = -np.inf if bits[d] >= max_bits else C[d, bits[d]] - C[d, bits[d] + 1]
    return bits


def optimal_alloc_cost(C, budget, max_bits):
    dp = np.full(budget + 1, np.inf); dp[0] = 0.0
    for d in range(C.shape[0]):
        nd = np.full(budget + 1, np.inf)
        for b in range(0, max_bits + 1):
            if b > budget:
                break
            nd[b:] = np.minimum(nd[b:], dp[:budget + 1 - b] + C[d, b])
        dp = nd
    return float(np.min(dp))


def rescore(C, bits):
    return float(C[np.arange(C.shape[0]), np.clip(bits, 0, C.shape[1] - 1)].sum())


def _gaussian_Dg(max_bits):
    """Normalized per-bit MSE Dg[b] of the optimal b-bit quantizer for N(0,1)."""
    x = RNG.standard_normal(200_000).astype(np.float32)
    dp = saq.build_codebook_dp(x, max_bits=max_bits, num_bins=NUM_BINS)
    v = float(x.var())
    return np.array([saq.codebook_mse(x, dp.codebooks[b]) for b in range(max_bits + 1)]) / v


def theory2():
    Dg = _gaussian_Dg(MAXB)                          # convex, decreasing
    D = 256
    spectra = {
        "geometric (0.95^i)": 0.95 ** np.arange(D),
        "power-law (i^-1)": 1.0 / (1.0 + np.arange(D)),
        "power-law (i^-2)": 1.0 / (1.0 + np.arange(D)) ** 2,
    }
    abpd = [1, 2, 3, 4, 5, 6]
    ratios = {}
    for name, var in spectra.items():
        C = var[:, None] * Dg[None, :]               # C[d][b] = var_d * Dg[b]
        rr = []
        for ab in abpd:
            budget = int(round(D * ab))
            g = rescore(C, numpy_greedy(C, budget, MAXB))
            o = optimal_alloc_cost(C, budget, MAXB)
            rr.append(g / o)
        ratios[name] = rr
        print(f"  T2 {name}: greedy/opt = {np.round(rr,5)}", flush=True)
    # Non-convex counterexample: a cost curve with an increasing marginal gain.
    nonconvex = np.array([1.0, 0.95, 0.90, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])  # big drop at b=3
    return Dg, abpd, ratios, nonconvex


def plot_t2(axA, axB, Dg, abpd, ratios, nonconvex):
    for name, rr in ratios.items():
        axA.plot(abpd, rr, marker="o", label=name)
    axA.axhline(1.0, color=OPT_C, ls=":", lw=1)
    axA.set_ylim(0.99, max(1.02, max(max(r) for r in ratios.values()) * 1.01))
    axA.set_xlabel("avg bits / dim"); axA.set_ylabel("greedy / full-optimization (DP)")
    axA.set_title("T2 — greedy vs full optimization\n(synthetic variance spectra)")
    axA.legend(fontsize=7); axA.grid(alpha=0.3)

    # Mechanism: marginal gain per bit (convex Gaussian vs non-convex counter).
    b = np.arange(1, MAXB + 1)
    axB.plot(b, -np.diff(Dg), marker="o", label="Gaussian RD (convex)")
    axB.plot(np.arange(1, len(nonconvex)), -np.diff(nonconvex), marker="s",
             label="constructed (non-convex)")
    axB.set_xlabel("bit  (b -> b+1)"); axB.set_ylabel("marginal MSE reduction")
    axB.set_title("Why greedy = optimal: convex RD curves\n(greedy optimal iff marginal gain decreasing)")
    axB.legend(fontsize=7); axB.grid(alpha=0.3)


# =============================================================== T3: packing
_CAP = 8
_CONFIGS = []
def _gen(start, rem, cur):
    if cur:
        _CONFIGS.append(tuple(cur))
    for s in range(start, rem + 1):
        cur.append(s); _gen(s, rem - s, cur); cur.pop()
_gen(1, _CAP, [])


def ffd_bytes(widths, cap=_CAP):
    bins = []
    for w in sorted(widths, reverse=True):
        for i in range(len(bins)):
            if bins[i] >= w:
                bins[i] -= w; break
        else:
            bins.append(cap - w)
    return len(bins)


def opt_bytes(widths, cap=_CAP):
    counts = np.zeros(cap + 1, dtype=int)
    for w in widths:
        counts[w] += 1
    sizes = [s for s in range(1, cap + 1) if counts[s] > 0]
    if not sizes:
        return 0
    A = np.zeros((len(sizes), len(_CONFIGS)))
    for c, cfg in enumerate(_CONFIGS):
        for j, s in enumerate(sizes):
            A[j, c] = cfg.count(s)
    res = milp(c=np.ones(len(_CONFIGS)),
               constraints=LinearConstraint(A, lb=np.array([counts[s] for s in sizes]), ub=np.inf),
               integrality=np.ones(len(_CONFIGS)), bounds=Bounds(lb=0, ub=np.inf))
    return int(round(res.fun))


def theory3(n_instances=600):
    rows = []   # (mean_size, ffd/opt)
    for _ in range(n_instances):
        N = int(RNG.integers(50, 1500))
        # mix of size distributions to stress the packer
        mode = RNG.integers(0, 3)
        if mode == 0:
            w = RNG.integers(1, 9, N)                       # uniform 1..8
        elif mode == 1:
            w = np.clip(np.round(RNG.normal(5, 2, N)), 1, 8).astype(int)   # mid-heavy
        else:
            w = RNG.choice([1, 2, 3, 7, 8], size=N, p=[.2, .2, .2, .2, .2])  # bimodal
        fb = ffd_bytes(w.tolist()); ob = opt_bytes(w.tolist())
        rows.append((float(w.mean()), fb / ob))
    rows = np.array(rows)
    print(f"  T3: FFD==OPT on {np.mean(rows[:,1]==1.0)*100:.1f}% of {n_instances} instances; "
          f"max ratio {rows[:,1].max():.4f}", flush=True)
    return rows


def plot_t3(axA, axB, rows):
    axA.hist(rows[:, 1], bins=40, color="#9467bd", edgecolor="white")
    axA.axvline(1.0, color=OPT_C, ls=":", lw=1)
    axA.set_xlabel("FFD bytes / optimal bytes"); axA.set_ylabel("# random instances")
    axA.set_title("T3 — FFD vs optimal byte packing\n(distribution over random instances)")
    axA.grid(alpha=0.3)

    axB.scatter(rows[:, 0], rows[:, 1], s=8, alpha=0.4, color="#9467bd", label="random instance")
    axB.axhline(11 / 9, color="#d62728", ls="--", lw=1, label="worst-case bound 11/9")
    axB.axhline(1.0, color=OPT_C, ls=":", lw=1, label="optimal")
    axB.set_xlabel("mean code width (bits)"); axB.set_ylabel("FFD / optimal")
    axB.set_title("Observed FFD ratio vs theoretical worst case")
    axB.legend(fontsize=7); axB.grid(alpha=0.3)


# =============================================================== main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== T1: k-means vs 1-D optimal ===", flush=True)
    bits, t1 = theory1()
    print("=== T2: greedy vs full optimization ===", flush=True)
    Dg, abpd, t2r, nonconvex = theory2()
    print("=== T3: FFD vs optimal packing ===", flush=True)
    t3 = theory3()

    # individual figures
    f, a = plt.subplots(figsize=(7, 5)); plot_t1(a, bits, t1)
    f.tight_layout(); f.savefig(OUT / "theory1_codebook_1d.png", dpi=130); plt.close(f)
    f, ax = plt.subplots(1, 2, figsize=(13, 4.6)); plot_t2(ax[0], ax[1], Dg, abpd, t2r, nonconvex)
    f.tight_layout(); f.savefig(OUT / "theory2_alloc.png", dpi=130); plt.close(f)
    f, ax = plt.subplots(1, 2, figsize=(13, 4.6)); plot_t3(ax[0], ax[1], t3)
    f.tight_layout(); f.savefig(OUT / "theory3_packing.png", dpi=130); plt.close(f)

    # combined headline (one panel per theory question)
    f, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    plot_t1(ax[0], bits, t1)
    for name, rr in t2r.items():
        ax[1].plot(abpd, rr, marker="o", label=name)
    ax[1].axhline(1.0, color=OPT_C, ls=":", lw=1); ax[1].set_ylim(0.99, 1.02)
    ax[1].set_xlabel("avg bits / dim"); ax[1].set_ylabel("greedy / full optimization")
    ax[1].set_title("T2 — greedy vs full optimization"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    ax[2].scatter(t3[:, 0], t3[:, 1], s=8, alpha=0.4, color="#9467bd")
    ax[2].axhline(11 / 9, color="#d62728", ls="--", lw=1, label="worst-case 11/9")
    ax[2].axhline(1.0, color=OPT_C, ls=":", lw=1)
    ax[2].set_xlabel("mean code width (bits)"); ax[2].set_ylabel("FFD / optimal")
    ax[2].set_title("T3 — FFD vs optimal packing"); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    f.suptitle("Theory: algorithmic approximation quality (dataset-independent)", fontsize=13)
    f.tight_layout(rect=[0, 0, 1, 0.95]); f.savefig(OUT / "theory_combined.png", dpi=130); plt.close(f)
    print("=== done -> theory{1,2,3}_*.png + theory_combined.png ===", flush=True)


if __name__ == "__main__":
    main()
