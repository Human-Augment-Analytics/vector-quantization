"""Experiment 3 (professor): how close is FFD byte-packing to optimal, and what
does the 'rewind' rule cost?

The method packs each per-dim code (b_d bits, b_d in 1..8) wholly into one byte
(bins of capacity 8, so dims may share a byte, e.g. 5+3 or 4+3+1). FFD =
first-fit-decreasing bin packing.

Part A - packing optimality. For greedy allocations across an avg_bits sweep:
  ffd_bytes vs LB = ceil(sum_bits / 8). OPT in [LB, ffd_bytes], so ffd_bytes==LB
  PROVES FFD optimal; ffd_bytes/LB upper-bounds FFD/OPT. Any ffd_bytes>LB is
  resolved exactly by a small branch-and-bound (instances are tiny gaps).

Part B - rewind rule. Given a byte budget (the packing target), if FFD needs more
  bytes than the budget, rewind: drop the bit budget by 1, re-allocate (greedy),
  re-pack; repeat until it fits. Measure bits sacrificed + distortion increase.

Cost table C[d][b] is cached to approx_packing/cost_table.npy (63s to build once).

Env knobs: MAX_BITS, AVG_BITS_LIST, DIM_LIMIT.
Outputs: approx_packing.csv (Part A + B per avg_bits).
"""
from pathlib import Path
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
import saq

# All byte-configurations: multisets of widths 1..CAP summing to <= CAP (66 for CAP=8).
_CAP = 8
_CONFIGS = []
def _gen(start, rem, cur):
    if cur:
        _CONFIGS.append(tuple(cur))
    for s in range(start, rem + 1):
        cur.append(s); _gen(s, rem - s, cur); cur.pop()
_gen(1, _CAP, [])


def opt_bytes(widths, cap=_CAP):
    """Exact minimum bytes via the cutting-stock config ILP (OPT, not a bound)."""
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
    demand = np.array([counts[s] for s in sizes])
    res = milp(c=np.ones(len(_CONFIGS)),
               constraints=LinearConstraint(A, lb=demand, ub=np.inf),
               integrality=np.ones(len(_CONFIGS)), bounds=Bounds(lb=0, ub=np.inf))
    return int(round(res.fun))

BASE = os.environ.get(
    "VQ_DATA_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k")
PCA_FILE = os.environ.get("VQ_PCA_FILE", f"{BASE}/vectors_pca.fvecs")
OUT = Path(os.environ.get(
    "VQ_OUT_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/approx_packing"))
CACHE = OUT / "cost_table.npy"

MAX_BITS = int(os.environ.get("MAX_BITS", 8))
AVG_BITS_LIST = [float(x) for x in os.environ.get(
    "AVG_BITS_LIST", "1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7").split(",")]
_dl = os.environ.get("DIM_LIMIT", "")
DIM_LIMIT = int(_dl) if _dl else None


def get_cost_table():
    if CACHE.exists():
        C = np.load(CACHE)
        print(f"loaded cached cost table {C.shape}", flush=True)
        return C
    print(f"building cost table from {PCA_FILE} ...", flush=True)
    X = np.ascontiguousarray(saq.load_fvecs(PCA_FILE), dtype=np.float32)
    if DIM_LIMIT:
        X = X[:, :DIM_LIMIT]
    n, D = X.shape
    C = np.zeros((D, MAX_BITS + 1), dtype=np.float64)
    opts = saq.LloydOpts(); opts.max_bits = MAX_BITS
    t0 = time.time()
    for d in range(D):
        ll = saq.build_codebook_lloyd(np.ascontiguousarray(X[:, d]), opts)
        C[d, :] = np.asarray(ll.costs, dtype=np.float64)[: MAX_BITS + 1]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, C)
    print(f"cost table {C.shape} built+cached ({time.time()-t0:.1f}s)", flush=True)
    return C


def numpy_greedy(C, budget, max_bits):
    D = C.shape[0]
    bits = np.zeros(D, dtype=np.int64)
    gain = C[:, 0] - C[:, 1]
    for _ in range(int(budget)):
        d = int(np.argmax(gain))
        if not np.isfinite(gain[d]) or gain[d] <= 0:
            break
        bits[d] += 1
        gain[d] = -np.inf if bits[d] >= max_bits else C[d, bits[d]] - C[d, bits[d] + 1]
    return bits


def ffd_bytes(widths, cap=8):
    """First-fit-decreasing bin count for item sizes `widths` into bins of `cap`."""
    bins = []  # remaining capacity per open bin
    for w in sorted(widths, reverse=True):
        placed = False
        for i in range(len(bins)):
            if bins[i] >= w:
                bins[i] -= w
                placed = True
                break
        if not placed:
            bins.append(cap - w)
    return len(bins)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = get_cost_table()
    D = C.shape[0]
    cap = 8

    rows = []
    for ab in AVG_BITS_LIST:
        budget = int(round(D * ab))
        bits = numpy_greedy(C, budget, MAX_BITS)
        widths = bits[bits > 0].tolist()
        sum_bits = int(sum(widths))
        n_items = len(widths)

        fb = ffd_bytes(widths, cap)
        lb = math.ceil(sum_bits / cap)   # perfect-packing LB (often unreachable at high bpd)
        ob = opt_bytes(widths, cap)      # exact optimal bytes

        # Part B: byte budget = OPT of the target allocation. If FFD needs more than
        # OPT, rewind (drop the bit budget) to the largest budget whose greedy
        # allocation FFD-packs into <= ob bytes. Binary search (FFD bytes rise with
        # the bit budget). Measures the bits/distortion lost to FFD's suboptimality.
        byte_budget = ob
        if fb <= byte_budget:
            rw_bits, rw_budget = bits, budget
        else:
            lo, hi = 0, budget
            while lo < hi:
                mid = (lo + hi + 1) // 2
                mb = numpy_greedy(C, mid, MAX_BITS)
                if ffd_bytes(mb[mb > 0].tolist(), cap) <= byte_budget:
                    lo = mid
                else:
                    hi = mid - 1
            rw_budget = lo
            rw_bits = numpy_greedy(C, rw_budget, MAX_BITS)
        bits_sacrificed = budget - int(rw_bits.sum())
        d_before = float(C[np.arange(D), np.clip(bits, 0, MAX_BITS)].sum())
        d_after = float(C[np.arange(D), np.clip(rw_bits, 0, MAX_BITS)].sum())

        rows.append({
            "avg_bits": ab, "budget": budget, "n_items": n_items, "sum_bits": sum_bits,
            "ffd_bytes": fb, "opt_bytes": ob, "lb_bytes": lb,
            "ffd_over_opt": fb / ob, "ffd_minus_opt": fb - ob,
            "bits_sacrificed": bits_sacrificed,
            "distortion_inflation": (d_after / d_before) if d_before > 0 else 1.0,
        })
        print(f"  ab={ab}: ffd={fb} opt={ob} lb={lb} (ffd/opt={fb/ob:.5f}, +{fb-ob}B)  "
              f"rewind_bits_lost={bits_sacrificed} dist_infl={d_after/d_before:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "approx_packing.csv", index=False)
    print("\n=== EXP 3: FFD vs optimal byte packing + rewind (dbpedia PCA) ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print("\nffd_over_opt = FFD bytes / exact-optimal bytes (1.0 = FFD optimal).", flush=True)
    print("lb = perfect-packing bound (unreachable at high bpd when items are large).", flush=True)
    print("DONE_APPROX_PACKING", flush=True)


if __name__ == "__main__":
    main()
