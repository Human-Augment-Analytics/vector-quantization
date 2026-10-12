"""Regenerate figures/centroid_convergence.png (paper fig:centroid-conv).

1-D Lloyd codebook centroids converge to their infinite-data fixed points at
~1/sqrt(N). Setup: codebooks at b in {2,3,4} bits trained on N i.i.d. standard
normal samples (WLOG: 1-D Lloyd on a Gaussian is scale-equivariant, so unit
variance loses nothing); error(N) = max_k |c_k(N) - c_k*| against the ANALYTIC
Lloyd-Max fixed point c* of N(0,1) (deterministic iteration with closed-form
Gaussian conditional means — a true infinite-data reference, no sampling
error), averaged over SEEDS trials per N.

Replaces the uncommitted June scratch that produced the original PNG (which
used a finite 20M-sample reference, single trial).
Run:  python results/centroid_convergence.py   (writes results/_figs/ + prints
      the caption numbers)
"""
import sys
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

BITS = (2, 3, 4)
NS = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
SEEDS = 10
OUT = HERE / "_figs"


def _phi(x):   # standard normal pdf
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _Phi(x):   # standard normal cdf
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def lloyd_max_analytic(k: int, tol: float = 1e-13, max_iter: int = 200_000):
    """Exact Lloyd-Max fixed point for N(0,1): c_i = E[X | cell_i] in closed
    form via phi/Phi; deterministic, no samples."""
    from scipy.stats import norm
    c = norm.ppf((np.arange(k) + 0.5) / k)   # quantile init
    for _ in range(max_iter):
        bnd = 0.5 * (c[:-1] + c[1:])
        lo = np.concatenate(([-np.inf], bnd))
        hi = np.concatenate((bnd, [np.inf]))
        new = np.empty_like(c)
        for i in range(k):
            pl = _phi(lo[i]) if np.isfinite(lo[i]) else 0.0
            ph = _phi(hi[i]) if np.isfinite(hi[i]) else 0.0
            mass = _Phi(hi[i]) - _Phi(lo[i]) if np.isfinite(hi[i]) else 1.0 - _Phi(lo[i])
            if not np.isfinite(lo[i]):
                mass = _Phi(hi[i])
            new[i] = (pl - ph) / mass
        shift = float(np.max(np.abs(new - c)))
        c = new
        if shift < tol:
            break
    return c


def lloyd_sample_converged(sorted_x: np.ndarray, csum: np.ndarray, k: int,
                           tol: float = 1e-12, max_iter: int = 200_000):
    """Lloyd on a sample, run to convergence. Sort + cumulative sums make each
    iteration O(k log n) (the paper's cumsum speedup), so full convergence is
    cheap — the figure then measures pure ESTIMATION error, not the solver
    truncation of a fixed iteration cap."""
    n = len(sorted_x)
    q = (np.arange(k) + 0.5) / k
    c = sorted_x[np.minimum((q * n).astype(int), n - 1)].astype(np.float64)
    for _ in range(max_iter):
        bnd = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(sorted_x, bnd)
        lo = np.concatenate(([0], idx))
        hi = np.concatenate((idx, [n]))
        cnt = (hi - lo)
        sums = csum[hi] - csum[lo]
        new = np.where(cnt > 0, sums / np.maximum(cnt, 1), c)
        new = np.sort(new)
        shift = float(np.max(np.abs(new - c)))
        c = new
        if shift < tol:
            break
    return c


def pop_distortion(c: np.ndarray) -> float:
    """Population MSE of the quantizer with centroids c (midpoint boundaries)
    under N(0,1), in closed form via phi/Phi. Exact; no samples."""
    c = np.sort(np.asarray(c, dtype=np.float64))
    bnd = 0.5 * (c[:-1] + c[1:])
    lo = np.concatenate(([-np.inf], bnd))
    hi = np.concatenate((bnd, [np.inf]))
    total = 0.0
    for a, bb, ck in zip(lo, hi, c):
        pa = _phi(a) if np.isfinite(a) else 0.0
        pb = _phi(bb) if np.isfinite(bb) else 0.0
        Fa = _Phi(a) if np.isfinite(a) else 0.0
        Fb = _Phi(bb) if np.isfinite(bb) else 1.0
        xa = a * pa if np.isfinite(a) else 0.0
        xb = bb * pb if np.isfinite(bb) else 0.0
        ex2 = (Fb - Fa) - (xb - xa)      # int x^2 phi
        ex1 = pa - pb                    # int x phi
        total += ex2 - 2.0 * ck * ex1 + ck * ck * (Fb - Fa)
    return total


def main():
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    refs = {b: lloyd_max_analytic(2 ** b) for b in BITS}
    opt_d = {b: pop_distortion(refs[b]) for b in BITS}
    for b in BITS:
        print(f"b={b} analytic fixed point: {np.round(refs[b], 6)}  "
              f"optimal distortion {opt_d[b]:.6f}")
    # draw/sort each (seed, N) sample once, fit all bit-widths on it
    err = {b: {n: [] for n in NS} for b in BITS}
    xmse = {b: {n: [] for n in NS} for b in BITS}   # excess population MSE
    for s in range(SEEDS):
        for n in NS:
            x = np.sort(np.random.default_rng(1000 + s).standard_normal(n))
            csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
            for b in BITS:
                c = lloyd_sample_converged(x, csum, 2 ** b)
                err[b][n].append(float(np.max(np.abs(c - refs[b]))))
                xmse[b][n].append(pop_distortion(c) - opt_d[b])
    for b, color in zip(BITS, ("C0", "C1", "C2")):
        errs = []
        for n in NS:
            d = err[b][n]
            errs.append(np.mean(d))
            print(f"b={b} N={n:>10,}  mean max centroid error {np.mean(d):.5f}  "
                  f"(range {min(d):.5f}-{max(d):.5f})")
        ax.loglog(NS, errs, "o-", color=color, label=f"$b={b}$ ($2^{b}$ levels)")
    # 1/sqrt(N) guide through the b=4 starting point
    guide = errs[0] * np.sqrt(NS[0] / np.asarray(NS, float))
    ax.loglog(NS, guide, "k--", lw=1, alpha=0.6, label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("training samples $N$")
    ax.set_ylabel(r"max centroid error  $\max_k |\hat{c}_k - c_k^{\,*}|$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "centroid_convergence.png", dpi=200)
    print(f"wrote {OUT/'centroid_convergence.png'}")

    # excess population MSE of the trained codebook vs the Lloyd-Max optimum:
    # distortion is locally quadratic at the fixed point, so the ~1/sqrt(N)
    # centroid error should appear here as a ~1/N excess-MSE decay
    fig2, ax2 = plt.subplots(figsize=(5.2, 3.6))
    for b, color in zip(BITS, ("C0", "C1", "C2")):
        means = []
        for n in NS:
            d = xmse[b][n]
            means.append(np.mean(d))
            print(f"b={b} N={n:>10,}  mean excess MSE {np.mean(d):.3e}  "
                  f"(rel {np.mean(d)/opt_d[b]:.2e};"
                  f" range {min(d):.2e}-{max(d):.2e})")
        ax2.loglog(NS, means, "o-", color=color, label=f"$b={b}$ ($2^{b}$ levels)")
    guide = means[0] * (NS[0] / np.asarray(NS, float))
    ax2.loglog(NS, guide, "k--", lw=1, alpha=0.6, label=r"$\propto 1/N$")
    ax2.set_xlabel("training samples $N$")
    ax2.set_ylabel(r"excess MSE  $D(\hat{c}) - D(c^{*})$")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, which="both", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(OUT / "codebook_excess_mse.png", dpi=200)
    print(f"wrote {OUT/'codebook_excess_mse.png'}")


if __name__ == "__main__":
    main()
