"""PCA stability on MSMARCO moments (paper fig:pca-overshoot).

Ground truth: mean mu and covariance Sigma computed from the MSMARCO 500K
corpus; true PCA basis U, variances lambda = eigh(Sigma). For each n: draw n
samples from N(mu, Sigma), estimate (mu_hat, U_hat) by sample PCA, and
measure, per dimension, the variance the TRUE distribution has in the
ESTIMATED basis, var_j = U_hat_j^T Sigma U_hat_j (this is the exact
M -> infinity limit of projecting a large test sample). Reported metric:

    max_j (var_j - lambda_j) / lambda_j

the worst relative variance overshoot of the approximate PCA, which is what
bit allocation actually consumes. Expected decay ~ 1/sqrt(n).

Run: python results/pca_variance_overshoot.py [--base <fvecs>] [--seeds 3]
Writes results/_figs/pca_variance_overshoot.png and prints the values.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "_figs"
DEFAULT_BASE = ("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/"
                "msmarco_500k/base.fvecs")


def read_fvecs_chunked_moments(path, chunk=50_000):
    """Stream fvecs; return (n, mu, Sigma) in float64 without loading all rows."""
    raw = np.memmap(path, dtype=np.float32, mode="r")
    d = int(raw[:1].view(np.int32)[0])
    rows = raw.reshape(-1, d + 1)
    n = rows.shape[0]
    s = np.zeros(d)
    acc = np.zeros((d, d))
    for i in range(0, n, chunk):
        X = np.asarray(rows[i:i + chunk, 1:], dtype=np.float64)
        s += X.sum(0)
        acc += X.T @ X
    mu = s / n
    Sigma = acc / n - np.outer(mu, mu)
    return n, mu, Sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384, 32768, 65536, 131072,
                             262144, 524288, 1048576, 2097152])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    n0, mu, Sigma = read_fvecs_chunked_moments(args.base)
    d = len(mu)
    lam, U = np.linalg.eigh(Sigma)
    lam, U = lam[::-1].copy(), U[:, ::-1].copy()
    lam = np.maximum(lam, 1e-18)
    print(f"MSMARCO moments from {n0} rows, d={d}; lambda_1={lam[0]:.3e} "
          f"lambda_d={lam[-1]:.3e}", flush=True)
    sd = np.sqrt(lam)

    results = {n: [] for n in args.ns}
    for seed in range(args.seeds):
        for n in args.ns:
            rng = np.random.default_rng(100 * seed + 7)
            # draw n ~ N(mu, Sigma), chunked
            ssum = np.zeros(d)
            acc = np.zeros((d, d))
            left = n
            while left > 0:
                m = min(100_000, left)
                X = mu + (rng.standard_normal((m, d)) * sd) @ U.T
                ssum += X.sum(0)
                acc += X.T @ X
                left -= m
            mu_hat = ssum / n
            Shat = acc / n - np.outer(mu_hat, mu_hat)
            _, V = np.linalg.eigh(Shat)
            Uhat = V[:, ::-1]
            # exact variance of the true distribution in the estimated basis
            var_hat = np.einsum("ij,jk,ki->i", Uhat.T, Sigma, Uhat)
            overshoot = float(np.max((var_hat - lam) / lam))
            results[n].append(overshoot)
            print(f"seed {seed} n={n:>7,}  max rel overshoot {overshoot:.4f}",
                  flush=True)

    means = [np.mean(results[n]) for n in args.ns]
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.loglog(args.ns, means, "o-", color="C0",
              label="max relative variance overshoot")
    nsf = np.asarray(args.ns, float)
    g_half = means[-1] * np.sqrt(args.ns[-1] / nsf)
    ax.loglog(args.ns, g_half, "k--", lw=1, alpha=0.6, label=r"$\propto 1/\sqrt{n}$")
    g_one = means[1] * (args.ns[1] / nsf)
    ax.loglog(args.ns, g_one, "k:", lw=1, alpha=0.6, label=r"$\propto 1/n$")
    ax.set_xlabel("PCA fitting samples $n$")
    ax.set_ylabel(r"$\max_j (\hat{v}_j - \lambda_j)/\lambda_j$")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "pca_variance_overshoot.png", dpi=200)
    print(f"wrote {OUT/'pca_variance_overshoot.png'}")
    for n, m in zip(args.ns, means):
        print(f"n={n:>7,}  mean max overshoot {m:.4f}")


if __name__ == "__main__":
    main()
