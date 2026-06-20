"""Experiment 2 (professor): how close is greedy bit allocation to optimal?

Two references ("both", per design decision):
  1. OPTIMAL given the exact empirical per-dim per-bit costs -> a min-plus DP over
     the cost table (the true optimum greedy is approximating).
  2. The analytic Bennett allocation (var/2^b, the paper/DP model), RE-SCORED under
     the same empirical costs -> "how good is the analytic allocation on real costs."

All allocations are per-dim (the professor's method has a codebook per dimension)
and scored on ONE empirical cost table C[d][b] = MSE of dim d at b bits (b=0 ->
variance), built from build_codebook_lloyd.

IMPORTANT (WSL safety): the engine BitAllocatorDP at dim_padding_size=1 over ~1500
dims blows its internal segmentation-DP table to multi-GB and crashes WSL. So the
allocations are done in numpy here (tiny arrays). The engine greedy (safe at size
1) is used only as an optional small-D cross-check (CROSS_CHECK=1, D<=CC_MAX_D).

Env knobs: MAX_BITS, AVG_BITS_LIST, DIM_LIMIT, LLOYD_SAMPLE, CROSS_CHECK.

Outputs:
  approx_alloc.csv  per avg_bits: distortion + optimality ratio for each allocator.
"""
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import saq

BASE = os.environ.get(
    "VQ_DATA_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k")
PCA_FILE = os.environ.get("VQ_PCA_FILE", f"{BASE}/vectors_pca.fvecs")
OUT = Path(os.environ.get(
    "VQ_OUT_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/approx_alloc"))

MAX_BITS = int(os.environ.get("MAX_BITS", 8))
AVG_BITS_LIST = [float(x) for x in os.environ.get("AVG_BITS_LIST", "1,2,3,4,5,6,8").split(",")]
_dl = os.environ.get("DIM_LIMIT", "")
DIM_LIMIT = int(_dl) if _dl else None
LLOYD_SAMPLE = int(os.environ.get("LLOYD_SAMPLE", 0))
CROSS_CHECK = os.environ.get("CROSS_CHECK", "") == "1"
CC_MAX_D = 128                      # only cross-check engine greedy at small D (size-1 is safe but slow)


def build_cost_table(X, max_bits, sample):
    """C[d][b] = empirical MSE of dim d at b bits; C[d][0] = variance."""
    n, D = X.shape
    C = np.zeros((D, max_bits + 1), dtype=np.float64)
    opts = saq.LloydOpts()
    opts.max_bits = max_bits
    opts.sample_size = sample
    for d in range(D):
        col = np.ascontiguousarray(X[:, d])
        ll = saq.build_codebook_lloyd(col, opts)
        costs = np.asarray(ll.costs, dtype=np.float64)
        C[d, :] = costs[: max_bits + 1]
    return C


def numpy_greedy(C, budget, max_bits):
    """Per-dim marginal-gain greedy: repeatedly give one bit to the dim with the
    largest MSE drop. Identical rule to the engine BitAllocatorGreedy at size 1.
    Returns the per-dim bit array."""
    D = C.shape[0]
    bits = np.zeros(D, dtype=np.int64)
    # gain[d] = C[d, bits[d]] - C[d, bits[d]+1]; -inf at the cap.
    gain = C[:, 0] - C[:, 1]
    for _ in range(int(budget)):
        d = int(np.argmax(gain))
        if not np.isfinite(gain[d]) or gain[d] <= 0:
            break                                  # no further reduction possible
        bits[d] += 1
        if bits[d] >= max_bits:
            gain[d] = -np.inf
        else:
            gain[d] = C[d, bits[d]] - C[d, bits[d] + 1]
    return bits


def optimal_alloc_cost(C, budget, max_bits):
    """Exact min sum_d C[d][b_d] s.t. sum_d b_d <= budget, 0<=b_d<=max_bits.
    Min-plus DP, vectorized over the bit axis."""
    D = C.shape[0]
    dp = np.full(budget + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    for d in range(D):
        nd = np.full(budget + 1, np.inf, dtype=np.float64)
        for b in range(0, max_bits + 1):
            if b > budget:
                break
            cand = dp[: budget + 1 - b] + C[d, b]
            nd[b:] = np.minimum(nd[b:], cand)
        dp = nd
    return float(np.min(dp))


def rescore(C, bits):
    mb = C.shape[1] - 1
    bb = np.clip(bits, 0, mb)
    return float(C[np.arange(C.shape[0]), bb].sum())


def engine_greedy_bits(C, budget, max_bits, D):
    """Engine BitAllocatorGreedy at dim_padding_size=1 (safe: O(budget*D), no big alloc)."""
    cfg = saq.JointAllocationConfig()
    cfg.total_bits = int(budget)
    cfg.max_bits_per_dim = max_bits
    cfg.dim_padding_size = 1
    cfg.num_dim_padded = D
    cfg.num_bit_factors = 0
    res = saq.allocate_greedy(C.astype(np.float32), cfg)
    bits = np.zeros(D, dtype=np.int64)
    off = 0
    for dim_len, b in res.quant_plan:
        bits[off: off + dim_len] = b
        off += dim_len
    return bits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading {PCA_FILE} ...", flush=True)
    X = np.ascontiguousarray(saq.load_fvecs(PCA_FILE), dtype=np.float32)
    if DIM_LIMIT:
        X = X[:, :DIM_LIMIT]
    n, D = X.shape
    print(f"X={X.shape}  max_bits={MAX_BITS}", flush=True)

    t0 = time.time()
    C = build_cost_table(X, MAX_BITS, LLOYD_SAMPLE)
    var = C[:, 0]
    # Analytic Bennett cost curve: D(b) = var / 2^b (engine BitAllocatorDP model).
    C_ben = var[:, None] / (2.0 ** np.arange(MAX_BITS + 1))[None, :]
    print(f"cost table built ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    for ab in AVG_BITS_LIST:
        budget = int(round(D * ab))
        greedy_bits = numpy_greedy(C, budget, MAX_BITS)
        bennett_bits = numpy_greedy(C_ben, budget, MAX_BITS)   # greedy on convex Bennett == optimal-Bennett

        if CROSS_CHECK and D <= CC_MAX_D:
            eb = engine_greedy_bits(C, budget, MAX_BITS, D)
            match = bool(np.array_equal(eb, greedy_bits))
            print(f"  [cross-check ab={ab}] engine-greedy == numpy-greedy: {match}", flush=True)

        d_greedy = rescore(C, greedy_bits)
        d_bennett = rescore(C, bennett_bits)
        d_opt = optimal_alloc_cost(C, budget, MAX_BITS)

        rows.append({
            "avg_bits": ab, "budget": budget,
            "dist_optimal": d_opt, "dist_greedy": d_greedy, "dist_bennett": d_bennett,
            "ratio_greedy_opt": d_greedy / d_opt,
            "ratio_bennett_opt": d_bennett / d_opt,
            "greedy_bits_used": int(greedy_bits.sum()),
            "bennett_bits_used": int(bennett_bits.sum()),
        })
        print(f"  avg_bits={ab}: greedy/opt={d_greedy/d_opt:.5f}  "
              f"bennett/opt={d_bennett/d_opt:.4f}  (opt={d_opt:.4e})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "approx_alloc.csv", index=False)
    print("\n=== EXP 2: greedy vs optimal allocation (dbpedia PCA, per-dim) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print("\nratio = allocator_distortion / optimal_distortion  (1.0 = optimal)", flush=True)
    print("DONE_APPROX_ALLOC", flush=True)


if __name__ == "__main__":
    main()
