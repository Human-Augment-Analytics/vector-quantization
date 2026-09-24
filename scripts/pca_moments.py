#!/usr/bin/env python3
"""Once-per-dataset EXACT PCA for the recall benchmark (feeds VQ_PCA_CACHE).

Computes the same (mu, V, var) RankAwareQuantizer.fit derives — full-corpus
float64 mean + covariance + eigh — but once, instead of per cell (~1.1e17 FLOPs
at 53M). Three modes:

  partial : moments over a row range -> npz(S, G, count). Map step for an
            sbatch array; each slice is ~one IO pass over its rows.
      python pca_moments.py partial --data vectors.fvecs --rows 0 2000000 --out part_00.npz
  merge   : sum partials (filename-sorted) -> eigh -> npz(mu, V, var).
      python pca_moments.py merge --parts 'part_*.npz' --out pca_cache.npz
  single  : whole corpus sequentially with the SAME chunking as fit() —
            bit-identical to what a cell would compute (use for the 2M gate).
      python pca_moments.py single --data vectors.fvecs --out pca_cache.npz

Note: merge-of-partials changes float summation association vs the sequential
fit, so merged caches are exact-method but not bit-identical to in-fit PCA
(last-ulp eigh differences). The 53M run uses merge; gates use single.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np

CHUNK = 1_000_000   # must match RankAwareQuantizer.fit for bit-parity in 'single'


def open_fvecs(path):
    mm = np.memmap(path, dtype=np.float32, mode="r")
    d = int(mm[:1].view(np.int32)[0])
    return mm.reshape(-1, d + 1)[:, 1:], d


def moments(X, lo, hi):
    d = X.shape[1]
    S = np.zeros(d, dtype=np.float64)
    G = np.zeros((d, d), dtype=np.float64)
    for s in range(lo, hi, CHUNK):
        blk = np.asarray(X[s:min(s + CHUNK, hi)], dtype=np.float64)
        S += blk.sum(axis=0)
        G += blk.T @ blk
        print(f"  rows {s}..{s + blk.shape[0]} done", flush=True)
    return S, G, hi - lo


def finalize(S, G, n, out):
    mu = S / n
    C = G / n - np.outer(mu, mu)
    w, Vt = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    var = np.clip(w[order], 1e-12, None)
    V = Vt[:, order]
    np.savez(out, mu=mu, V=V, var=var, n=n)
    print(f"wrote {out}  (n={n}, D={len(mu)}, top var={var[0]:.4e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["partial", "merge", "single"])
    ap.add_argument("--data")
    ap.add_argument("--rows", nargs=2, type=int)
    ap.add_argument("--parts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    if a.mode == "partial":
        X, _ = open_fvecs(a.data)
        lo, hi = a.rows
        hi = min(hi, X.shape[0])
        S, G, n = moments(X, lo, hi)
        np.savez(a.out, S=S, G=G, count=n)
        print(f"wrote {a.out} (rows {lo}..{hi}, {time.time()-t0:.0f}s)")
    elif a.mode == "merge":
        files = sorted(glob.glob(a.parts))
        assert files, f"no partials match {a.parts}"
        S = G = None
        n = 0
        for f in files:
            z = np.load(f)
            S = z["S"] if S is None else S + z["S"]
            G = z["G"] if G is None else G + z["G"]
            n += int(z["count"])
            print(f"  merged {f}")
        finalize(S, G, n, a.out)
    else:  # single — bit-identical to in-fit PCA
        X, _ = open_fvecs(a.data)
        S, G, n = moments(X, 0, X.shape[0])
        finalize(S, G, n, a.out)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
