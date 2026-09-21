"""Measure the TRUE rank-aware weights: E[q_d^2] per PCA dim, vs the var_d^(1+alpha)
model. The score-noise derivation says the recall-aligned allocation cost is
E[q_d^2] * mse_d; alpha models E[q_d^2] ~ var_d^alpha. This script measures the
actual query energy spectrum and fits the best alpha per dataset.

Note q is NOT centered in the score q.x_hat, so E[q_d^2] = var-like term + (mean
projection)^2 — the mean term puts a floor under tail weights, which is a candidate
explanation for why alpha < 1 wins empirically. Runs on Windows python.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
DATA = REPO.parent / "SAQ" / "data" / "datasets"


def load_fvecs(path):
    data = np.fromfile(path, dtype=np.float32)
    d = int(data[0].view(np.int32))
    return data.reshape(-1, d + 1)[:, 1:].copy()


def pca_basis(X):
    N, D = X.shape
    CHUNK = 500_000
    S = np.zeros(D); G = np.zeros((D, D))
    for s in range(0, N, CHUNK):
        blk = np.asarray(X[s:s + CHUNK], dtype=np.float64)
        S += blk.sum(axis=0); G += blk.T @ blk
    mu = S / N
    C = G / N - np.outer(mu, mu)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return mu, V[:, order], np.clip(w[order], 1e-12, None)


def analyze(name, X, Q):
    mu, V, var = pca_basis(X)
    qc = np.asarray(Q, dtype=np.float64) @ V          # query coords, UNcentered
    w = (qc ** 2).mean(axis=0)                        # E[q_d^2] per dim
    mean_proj_sq = (mu @ V) ** 2                      # floor from the data mean

    # fit E[q_d^2] ~ c * var^beta on dims carrying real variance
    mask = var > var.max() * 1e-8
    lv, lw = np.log(var[mask]), np.log(w[mask])
    beta, logc = np.polyfit(lv, lw, 1)
    resid = lw - (beta * lv + logc)
    r2 = 1 - resid.var() / lw.var()

    # head-vs-tail energy concentration
    D = var.size
    head = D // 16
    conc_var = var[:head].sum() / var.sum()
    conc_q = w[:head].sum() / w.sum()
    print(f"{name}: D={D}  E[q^2] ~ var^beta fit: beta={beta:.3f} (R2={r2:.3f})  "
          f"-> implied alpha={beta:+.3f} in var^(1+alpha)  [cost = E[q^2]*var*Dg = var^(1+beta)*Dg]")
    print(f"    top-{head} dims hold {100*conc_var:.1f}% of data var, "
          f"{100*conc_q:.1f}% of query energy; "
          f"mean-projection floor: max mu_d^2/w_d = {np.max(mean_proj_sq / w):.2f}")


for name, xf, qf in [
    ("dbpedia-100K ", DATA / "dbpedia_100k" / "vectors.fvecs", DATA / "dbpedia_100k" / "queries.fvecs"),
    ("msmarco-500K ", DATA / "msmarco_500k" / "base.fvecs", DATA / "msmarco_500k" / "query.fvecs"),
]:
    X = load_fvecs(str(xf)); Q = load_fvecs(str(qf))[:1000]
    analyze(name, X, Q)
    del X, Q

import h5py
with h5py.File(str(DATA / "sift10m" / "SIFT10M" / "SIFT10Mfeatures.mat"), "r") as f:
    raw = f["fea"][:2_000_000]                        # 2M rows suffice for the spectrum
rng = np.random.default_rng(0)
q_ids = rng.choice(raw.shape[0], 1000, replace=False)
mask = np.zeros(raw.shape[0], dtype=bool); mask[q_ids] = True
analyze("sift10m (2M) ", raw[~mask].astype(np.float32), raw[q_ids].astype(np.float32))
