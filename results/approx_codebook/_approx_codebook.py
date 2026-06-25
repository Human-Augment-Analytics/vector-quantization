"""Experiment 1 (professor): how close is cumsum-kmeans (Lloyd) to the DP-optimal
per-dimension codebook?

For every PCA dimension of the (engine-matched) PCA-transformed database, build
both codebooks at b=1..MAX_BITS and score BOTH on the raw column values via
codebook_mse (exact nearest-centroid MSE), so the comparison is not polluted by
the DP builder's histogram-binned internal cost.

Calibration (validated in-session): build_codebook_dp bins the column into a
histogram; to be a faithful optimal reference at 2^b levels we need
num_bins >> 2^MAX_BITS. We use num_bins = NB_PER_LEVEL * 2^MAX_BITS. Under-
resolution makes DP look *worse* (raises dp_mse), so it conservatively understates
Lloyd's gap. DP cost is ~O(num_bins^2) and dominates; Lloyd/mse are negligible.

Parallel across dims (the DP binding holds the GIL -> use processes). Parent loads
X once and ships column-slices to workers to avoid N*600MB loads (WSL safety).

Env knobs: MAX_BITS, NB_PER_LEVEL, DIM_LIMIT, LLOYD_SAMPLE, WORKERS, CHUNK.

Outputs:
  approx_codebook.csv          per (dim, bits): dp_mse, lloyd_mse, ratio, var_d
  approx_codebook_summary.csv  per bits: mean/median/p95 ratio (unweighted + var-weighted)
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import time

import numpy as np
import pandas as pd

# Paths are env-driven so the same script runs locally and on PACE.
#   VQ_DATA_DIR : dataset dir containing vectors_pca.fvecs (default: local dbpedia_100k)
#   VQ_OUT_DIR  : output dir for CSVs (default: local results/approx_codebook)
BASE = os.environ.get(
    "VQ_DATA_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k")
PCA_FILE = os.environ.get("VQ_PCA_FILE", f"{BASE}/vectors_pca.fvecs")
OUT = Path(os.environ.get(
    "VQ_OUT_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/approx_codebook"))

MAX_BITS = int(os.environ.get("MAX_BITS", 8))           # DP-optimal valid for max_bits <= 8
NB_PER_LEVEL = int(os.environ.get("NB_PER_LEVEL", 40))  # num_bins = NB_PER_LEVEL * 2^MAX_BITS
NUM_BINS = NB_PER_LEVEL * (1 << MAX_BITS)
_dl = os.environ.get("DIM_LIMIT", "")                   # "" = all dims
DIM_LIMIT = int(_dl) if _dl else None
LLOYD_SAMPLE = int(os.environ.get("LLOYD_SAMPLE", 0))   # 0 = full data
WORKERS = int(os.environ.get("WORKERS", min(14, (os.cpu_count() or 4) - 2)))
CHUNK = int(os.environ.get("CHUNK", 32))                # dims per task


def _process_chunk(args):
    """Worker: cols is (n, len(dims)) float32; returns list of row dicts."""
    import saq
    dims, cols, vars_ = args
    opts = saq.LloydOpts()
    opts.max_bits = MAX_BITS
    opts.sample_size = LLOYD_SAMPLE
    out = []
    for j, d in enumerate(dims):
        col = np.ascontiguousarray(cols[:, j])
        # Exact (globally optimal) 1-D k-means reference — no histogram/num_bins,
        # ~100x faster than build_codebook_dp and truly optimal (Wu/Gronlund DP).
        dp = saq.build_codebook_exact(col, max_bits=MAX_BITS)
        ll = saq.build_codebook_lloyd(col, opts)
        for b in range(1, MAX_BITS + 1):
            dp_mse = saq.codebook_mse(col, dp.codebooks[b])
            ll_mse = saq.codebook_mse(col, ll.codebooks[b])
            out.append({
                "dim": int(d), "bits": b, "var": float(vars_[j]),
                "dp_mse": float(dp_mse), "lloyd_mse": float(ll_mse),
                "ratio": float(ll_mse / dp_mse) if dp_mse > 0 else np.nan,
                "k_dp": int(dp.codebooks[b].num_entries),
                "k_lloyd": int(ll.codebooks[b].num_entries),
            })
    return out


def main():
    import saq
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading {PCA_FILE} ...", flush=True)
    X = np.ascontiguousarray(saq.load_fvecs(PCA_FILE), dtype=np.float32)
    n, D = X.shape
    var = X.var(axis=0)
    ndim = D if DIM_LIMIT is None else min(DIM_LIMIT, D)
    print(f"X={X.shape}  num_bins={NUM_BINS}  dims={ndim}  max_bits={MAX_BITS}  "
          f"workers={WORKERS}  chunk={CHUNK}", flush=True)

    tasks = []
    for start in range(0, ndim, CHUNK):
        dims = list(range(start, min(start + CHUNK, ndim)))
        cols = np.ascontiguousarray(X[:, dims])         # (n, len(dims))
        tasks.append((dims, cols, var[dims]))

    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for chunk_rows in ex.map(_process_chunk, tasks):
            rows.extend(chunk_rows)
            done += 1
            dt = time.time() - t0
            print(f"  chunk {done}/{len(tasks)}  ({dt:.1f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["dim", "bits"]).reset_index(drop=True)
    df.to_csv(OUT / "approx_codebook.csv", index=False)

    summ = []
    for b in range(1, MAX_BITS + 1):
        sub = df[df.bits == b]
        r = sub["ratio"].to_numpy()
        w = sub["var"].to_numpy()
        summ.append({
            "bits": b,
            "ratio_mean": float(np.nanmean(r)),
            "ratio_median": float(np.nanmedian(r)),
            "ratio_p95": float(np.nanpercentile(r, 95)),
            "ratio_max": float(np.nanmax(r)),
            "ratio_varweighted": float(np.nansum(r * w) / np.nansum(w)),
            "frac_dims_lloyd_worse_1pct": float(np.mean(r > 1.01)),
        })
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "approx_codebook_summary.csv", index=False)

    print("\n=== EXP 1: Lloyd (cumsum k-means) vs DP-optimal codebook (dbpedia PCA) ===", flush=True)
    print(sdf.to_string(index=False), flush=True)
    print(f"\ntotal {time.time()-t0:.1f}s  |  ratio = lloyd_mse / dp_mse (1.0=optimal)", flush=True)
    print("DONE_APPROX_CODEBOOK", flush=True)


if __name__ == "__main__":
    main()
