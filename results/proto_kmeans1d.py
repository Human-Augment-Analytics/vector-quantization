"""Prototype: exact 1-D k-means (SMAWK/kmeans1d-style) vs the engine's histogram-DP
and Lloyd. Validates correctness of a portable exact DP and benchmarks the real
kmeans1d (SMAWK) against the engine builders on a real column.

Part A: my divide-and-conquer exact DP (the algorithm to port) == brute force (small)
        == kmeans1d (the oracle). One DP pass yields all bit-levels 1..maxbits.
Part B: kmeans1d (C++ SMAWK) vs engine build_codebook_dp (histogram) vs
        build_codebook_lloyd, on a dbpedia PCA column: MSE (re-scored on raw) + time.
"""
from __future__ import annotations
import sys, time
from itertools import product
import numpy as np
import kmeans1d
import saq

# ---------------------------------------------------------------- my exact DP
def exact_1d_kmeans_all_bits(x: np.ndarray, max_bits: int):
    """Divide-and-conquer DP optimization. Returns dict bit -> (centroids, sse).
    Exact global optimum (contiguous = optimal in 1-D). One pass to K=2^max_bits
    fills D[k'] for every k', so all bit-levels come free.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.size
    ps = np.concatenate([[0.0], np.cumsum(xs)])
    psq = np.concatenate([[0.0], np.cumsum(xs * xs)])

    def cost(i, j):                      # SSE of points [i, j)  (half-open)
        if j <= i: return 0.0
        c = j - i; s = ps[j] - ps[i]
        return (psq[j] - psq[i]) - s * s / c

    K = 1 << max_bits
    INF = float("inf")
    D = np.full((K + 1, n + 1), INF)
    A = np.zeros((K + 1, n + 1), dtype=np.int64)     # argmin split
    D[0, 0] = 0.0
    for m in range(1, n + 1):
        D[1, m] = cost(0, m); A[1, m] = 0
    want = {1 << b for b in range(0, max_bits + 1)}
    out = {}

    def dc(kp, lo, hi, optlo, opthi):
        if lo > hi: return
        mid = (lo + hi) // 2
        best, arg = INF, optlo
        jhi = min(mid - 1, opthi)
        for j in range(max(kp - 1, optlo), jhi + 1):
            v = D[kp - 1, j] + cost(j, mid)
            if v < best: best, arg = v, j
        D[kp, mid] = best; A[kp, mid] = arg
        dc(kp, lo, mid - 1, optlo, arg)
        dc(kp, mid + 1, hi, arg, opthi)

    for kp in range(2, K + 1):
        sys.setrecursionlimit(1 << 20)
        dc(kp, kp, n, kp - 1, n - 1)

    def backtrack(k):
        bnds = [n]; m = n
        for kk in range(k, 0, -1):
            j = A[kk, m]; bnds.append(j); m = j
        bnds = bnds[::-1]
        cen = [(ps[bnds[i + 1]] - ps[bnds[i]]) / (bnds[i + 1] - bnds[i]) for i in range(k)]
        return np.array(cen)

    for b in range(0, max_bits + 1):
        k = 1 << b
        cen = backtrack(k) if k <= n else np.unique(xs)
        out[b] = (cen, float(D[k, n]) if k <= n else 0.0)
    return out


def brute_1d(x, k):                      # exact via all contiguous splits (small n)
    xs = np.sort(np.asarray(x, float)); n = xs.size
    ps = np.concatenate([[0.0], np.cumsum(xs)]); psq = np.concatenate([[0.0], np.cumsum(xs*xs)])
    def cost(i, j):
        if j <= i: return 0.0
        c=j-i; s=ps[j]-ps[i]; return (psq[j]-psq[i]) - s*s/c
    best = float("inf")
    for cuts in product(range(1, n), repeat=k - 1):
        b = [0] + sorted(cuts) + [n]
        if any(b[i+1] <= b[i] for i in range(k)): continue
        best = min(best, sum(cost(b[i], b[i+1]) for i in range(k)))
    return best


def mse_to_centroids(x, cen):
    x = np.asarray(x, float); cen = np.sort(np.asarray(cen, float))
    idx = np.clip(np.searchsorted(0.5 * (cen[:-1] + cen[1:]), x), 0, len(cen) - 1)
    return float(((x - cen[idx]) ** 2).mean())


def main():
    rng = np.random.default_rng(0)
    print("=== Part A: correctness (my D&C DP vs brute force vs kmeans1d) ===")
    for trial in range(6):
        x = rng.standard_normal(rng.integers(8, 13))
        out = exact_1d_kmeans_all_bits(x, 2)             # k up to 4
        for k, b in [(2, 1), (4, 2)]:
            if k > x.size: continue
            mine = out[b][1]
            bf = brute_1d(x, k)
            _, cen = kmeans1d.cluster(x.tolist(), k)
            km = sum((x[np.argmin(np.abs(x[:, None] - np.array(cen)[None, :]), 1)] - x) ** 2 * 0)  # noop
            km_sse = mse_to_centroids(x, cen) * x.size
            ok = abs(mine - bf) < 1e-9 and abs(mine - km_sse) < 1e-6
            print(f"  n={x.size} k={k}: mine={mine:.6f} brute={bf:.6f} kmeans1d={km_sse:.6f}  {'OK' if ok else 'MISMATCH'}")

    print("\n=== Part B: kmeans1d (SMAWK) vs engine DP/Lloyd on a real column ===")
    PCA = "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k/vectors_pca.fvecs"
    X = np.ascontiguousarray(saq.load_fvecs(PCA), dtype=np.float32)
    col = np.ascontiguousarray(X[:, 0])                  # a head dim (heavy signal)
    n = col.size; MB = 8
    print(f"  column: n={n}, max_bits={MB} (k up to {1<<MB})")

    # engine histogram-DP (converged num_bins) + Lloyd
    t = time.perf_counter(); dp = saq.build_codebook_dp(col, max_bits=MB, num_bins=80*(1<<MB)); t_dp = time.perf_counter()-t
    t = time.perf_counter(); ll = saq.build_codebook_lloyd(col, saq.LloydOpts()); t_ll = time.perf_counter()-t
    # kmeans1d per bit-level (its API solves one k; a port would do one pass)
    t = time.perf_counter()
    km_cen = {}
    for b in range(1, MB+1):
        _, cen = kmeans1d.cluster(col.astype(np.float64).tolist(), 1 << b); km_cen[b] = np.array(cen)
    t_km = time.perf_counter()-t

    print(f"\n  {'bits':>4} {'kmeans1d':>11} {'engine-DP':>11} {'Lloyd':>11}   (MSE on raw)")
    for b in range(1, MB+1):
        m_km = mse_to_centroids(col, km_cen[b])
        m_dp = saq.codebook_mse(col, dp.codebooks[b])
        m_ll = saq.codebook_mse(col, ll.codebooks[b])
        print(f"  {b:>4} {m_km:>11.4e} {m_dp:>11.4e} {m_ll:>11.4e}   Lloyd/opt={m_ll/m_km:.4f} DP/opt={m_dp/m_km:.4f}")
    print(f"\n  time: kmeans1d(all bits)={t_km:.3f}s  engine-DP(1 call,all bits)={t_dp:.3f}s  Lloyd={t_ll:.3f}s")
    print("DONE_PROTO")


if __name__ == "__main__":
    main()
