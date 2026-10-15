"""Regenerate figures/covariance_convergence.png (paper tab:cov-stability).

Synthetic Gaussian with a KNOWN covariance (the only setting where "true
Sigma" exists): Sigma = U diag(lambda) U^T with random orthogonal U and a
power-law spectrum normalized to unit trace (mirroring unit-norm embeddings).

(a) Sample covariance / PCA convergence: ||Sigma_hat - Sigma||_F / ||Sigma||_F
    and top-r eigenvector subspace error vs N, with a 1/sqrt(N) guide.
(b) Quantizer stability: build the pipeline twice -- true-Sigma basis with
    analytic Lloyd-Max codebooks (scaled by sqrt(lambda_j)) vs an n_fit-sample
    basis with per-dim k-means codebooks trained on that same sample (the real
    pipeline) -- and compare held-out test distortion. Also reports the mean
    per-dimension 1-D earth-mover distance between the two pipelines'
    reconstructions (sorted-sample mean |diff|, original coordinates).

Replaces the uncommitted June scratch behind the original PNG. Prints the
tab:cov-stability numbers. Run: python results/covariance_stability.py
"""
import argparse
import sys
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "_figs"


def _phi(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def lloyd_max_analytic(k, tol=1e-13, max_iter=200_000):
    from scipy.stats import norm
    c = norm.ppf((np.arange(k) + 0.5) / k)
    for _ in range(max_iter):
        bnd = 0.5 * (c[:-1] + c[1:])
        lo = np.concatenate(([-np.inf], bnd))
        hi = np.concatenate((bnd, [np.inf]))
        new = np.empty_like(c)
        for i in range(k):
            pl = _phi(lo[i]) if np.isfinite(lo[i]) else 0.0
            ph = _phi(hi[i]) if np.isfinite(hi[i]) else 0.0
            mass = (_Phi(hi[i]) if np.isfinite(hi[i]) else 1.0) - (_Phi(lo[i]) if np.isfinite(lo[i]) else 0.0)
            new[i] = (pl - ph) / mass
        if float(np.max(np.abs(new - c))) < tol:
            c = new
            break
        c = new
    return c


def kmeans_1d(sorted_x, k, tol=1e-10, max_iter=10_000):
    csum = np.concatenate(([0.0], np.cumsum(sorted_x, dtype=np.float64)))
    n = len(sorted_x)
    q = (np.arange(k) + 0.5) / k
    c = sorted_x[np.minimum((q * n).astype(int), n - 1)].astype(np.float64)
    for _ in range(max_iter):
        bnd = 0.5 * (c[:-1] + c[1:])
        idx = np.searchsorted(sorted_x, bnd)
        lo = np.concatenate(([0], idx))
        hi = np.concatenate((idx, [n]))
        cnt = hi - lo
        new = np.where(cnt > 0, (csum[hi] - csum[lo]) / np.maximum(cnt, 1), c)
        new = np.sort(new)
        if float(np.max(np.abs(new - c))) < tol:
            return new
        c = new
    return c


def quantize_cols(Z, codebooks):
    """Nearest-codeword quantization per column; codebooks sorted."""
    Zq = np.empty_like(Z)
    for j, cb in enumerate(codebooks):
        bnd = 0.5 * (cb[:-1] + cb[1:])
        Zq[:, j] = cb[np.searchsorted(bnd, Z[:, j])]
    return Zq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--bpd", type=int, default=2, help="uniform bits per dim")
    ap.add_argument("--alpha", type=float, default=1.0, help="spectrum lambda_j ~ j^-alpha")
    ap.add_argument("--n-fit", type=int, default=10_000)
    ap.add_argument("--n-test", type=int, default=200_000)
    ap.add_argument("--subspace-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d, b = args.d, args.bpd
    rng = np.random.default_rng(args.seed)

    lam = np.arange(1, d + 1, dtype=np.float64) ** (-args.alpha)
    lam /= lam.sum()  # unit trace, like unit-norm embeddings
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    sd = np.sqrt(lam)

    def draw(n, rng_):
        return (rng_.standard_normal((n, d)).astype(np.float32) * sd.astype(np.float32)) @ U.T.astype(np.float32)

    # ---- (a) covariance / subspace convergence ----
    Sig = (U * lam) @ U.T
    sig_norm = np.linalg.norm(Sig)
    Ns = [100, 1_000, 10_000, 100_000, 1_000_000]
    cov_err, sub_err = [], []
    r = args.subspace_r
    for n in Ns:
        acc = np.zeros((d, d), dtype=np.float64)
        mean = np.zeros(d, dtype=np.float64)
        rng_n = np.random.default_rng(1000 + n)
        done = 0
        chunks = []
        while done < n:
            m = min(100_000, n - done)
            X = draw(m, rng_n)
            chunks.append(X)
            mean += X.sum(0, dtype=np.float64)
            acc += X.T.astype(np.float64) @ X.astype(np.float64)
            done += m
        mean /= n
        Shat = acc / n - np.outer(mean, mean)
        cov_err.append(np.linalg.norm(Shat - Sig) / sig_norm)
        _, V = np.linalg.eigh(Shat)
        Vr = V[:, ::-1][:, :r]
        # principal-angle subspace error vs the true top-r eigenvectors
        s = np.linalg.svd(U[:, :r].T @ Vr, compute_uv=False)
        sub_err.append(float(np.sqrt(1.0 - np.min(s) ** 2)))
        print(f"N={n:>9,}  cov rel err {cov_err[-1]:.4f}  top-{r} subspace err {sub_err[-1]:.4f}", flush=True)

    # ---- (b) quantizer stability, true vs n_fit-sample pipeline ----
    lm = lloyd_max_analytic(2 ** b)
    cbs_true = [sd[j] * lm for j in range(d)]

    fit = draw(args.n_fit, np.random.default_rng(7))
    mu_hat = fit.mean(0)
    Shat = np.cov((fit - mu_hat).astype(np.float64), rowvar=False)
    w, V = np.linalg.eigh(Shat)
    Uhat = V[:, ::-1]
    Zfit = (fit - mu_hat) @ Uhat.astype(np.float32)
    cbs_hat = [kmeans_1d(np.sort(Zfit[:, j].astype(np.float64)), 2 ** b) for j in range(d)]

    test = draw(args.n_test, np.random.default_rng(11))
    Zt_true = test @ U.astype(np.float32)
    Rec_true = quantize_cols(Zt_true.astype(np.float64), cbs_true) @ U.T
    Zt_hat = (test - mu_hat) @ Uhat.astype(np.float32)
    Rec_hat = quantize_cols(Zt_hat.astype(np.float64), cbs_hat) @ Uhat.T + mu_hat

    mse_true = float(np.mean((test - Rec_true) ** 2))
    mse_hat = float(np.mean((test - Rec_hat) ** 2))
    emd = float(np.mean([np.mean(np.abs(np.sort(Rec_true[:, j]) - np.sort(Rec_hat[:, j])))
                         for j in range(d)]))
    dmse = 100.0 * (mse_hat - mse_true) / mse_true
    print(f"\nMSE(true Sigma) {mse_true:.3e}  MSE(Sigma_hat, n={args.n_fit}) {mse_hat:.3e}  "
          f"dMSE {dmse:+.1f}%  mean 1-D EMD {emd:.2e}")
    print(f"(d={d}, {b} bpd uniform, spectrum alpha={args.alpha}, "
          f"n_test={args.n_test})")

    # ---- figure ----
    OUT.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax1.loglog(Ns, cov_err, "o-", label=r"$\|\hat\Sigma-\Sigma\|_F/\|\Sigma\|_F$")
    ax1.loglog(Ns, sub_err, "s-", label=f"top-{r} subspace err")
    guide = cov_err[0] * np.sqrt(Ns[0] / np.asarray(Ns, float))
    ax1.loglog(Ns, guide, "k--", lw=1, alpha=0.6, label=r"$\propto 1/\sqrt{N}$")
    ax1.set_xlabel("samples $N$"); ax1.set_ylabel("error")
    ax1.set_title("Sample covariance / PCA $\\to$ truth")
    ax1.legend(frameon=False, fontsize=8); ax1.grid(True, which="both", alpha=0.25)
    ax2.bar([0, 1], [mse_true, mse_hat], width=0.6, color=["black", "#1f77b4"])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["true $\\Sigma$", f"$\\hat\\Sigma$ ($n$={args.n_fit:,})"])
    ax2.set_ylabel("test quantizer MSE")
    ax2.set_title(f"Quantizer stability: $\\Delta$MSE={dmse:+.1f}%, EMD={emd:.2e}")
    fig.tight_layout()
    fig.savefig(OUT / "covariance_convergence.png", dpi=200)
    print(f"wrote {OUT/'covariance_convergence.png'}")


if __name__ == "__main__":
    main()
