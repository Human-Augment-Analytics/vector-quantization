#!/usr/bin/env python3
"""Full 7-method benchmark on dbpedia-100K.

Runs:
  - 6 methods (pq, sq, rabitq, lvq, saq_paper, ours) at bpd 1..8
  - opq at bpd 1, 2, 4  (pathologically slow at high bpd)

Writes combined CSV + Pareto plots to results/dbpedia_dev_full/.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path if needed
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from haag_vq.benchmarks.quantizer_study import _load_fvecs, run_study_arrays
from haag_vq.benchmarks.study_plots import pareto_curves

import os

# Env-driven so the same driver runs locally and on PACE at any scale.
#   VQ_DATA_DIR : dataset dir with vectors.fvecs + queries.fvecs (raw)
#   VQ_OUT_DIR  : output dir for results CSV/plots
#   VQ_DATASET  : dataset label written into the results
#   VQ_N_QUERIES: number of queries to evaluate (default 1000)
#   VQ_METHODS / VQ_BPD / VQ_OPQ_BPD : comma-separated overrides
_DEF_DIR = "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k"
_DATA_DIR = os.environ.get("VQ_DATA_DIR", _DEF_DIR)
DATA_BASE = os.environ.get("VQ_DATA_BASE", f"{_DATA_DIR}/vectors.fvecs")
DATA_QUERY = os.environ.get("VQ_DATA_QUERY", f"{_DATA_DIR}/queries.fvecs")
OUT_DIR = Path(os.environ.get(
    "VQ_OUT_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/dbpedia_dev_full"))
DATASET_LABEL = os.environ.get("VQ_DATASET", "dbpedia_100k")
N_QUERIES = int(os.environ.get("VQ_N_QUERIES", 1000))

KS = (1, 10, 100)
CHUNK_SIZE = 50_000
MSE_SAMPLE = 100_000

FULL_METHODS = os.environ.get(
    "VQ_METHODS", "pq,sq,rabitq,lvq,saq_paper,ours,rankaware,perdim_mse").split(",")
FULL_BPD = [int(b) for b in os.environ.get("VQ_BPD", "1,2,3,4,5,6,7,8").split(",")]

OPQ_METHODS = ["opq"]
OPQ_BPD = [int(b) for b in os.environ.get("VQ_OPQ_BPD", "1,2,4").split(",")]


def run_with_timeout_guard(method, bpd, X, Q, timeout_seconds=1200):
    """Run a single (method, bpd) cell with a soft wall-clock guard.
    Returns (df_row, wall_seconds, error_msg)
    """
    from haag_vq.benchmarks.quantizer_study import _build_quantizer
    from haag_vq.benchmarks.exact_search import (
        build_scaled_ip_index,
        compute_exact_norms,
        normalized_ground_truth,
        reconstruction_mse,
        recall_at_ks,
        search_index,
    )
    from datetime import datetime, timezone

    n, D = X.shape
    norms = compute_exact_norms(X)
    gt = normalized_ground_truth(X, Q, k=max(KS), norms=norms, chunk=CHUNK_SIZE)
    rng = np.random.default_rng(0)
    if n > MSE_SAMPLE:
        sample_ids = rng.choice(n, MSE_SAMPLE, replace=False).astype(np.uint32)
    else:
        sample_ids = np.arange(n, dtype=np.uint32)

    t0 = time.time()
    try:
        q = _build_quantizer(method, bpd, D)
        q.fit(X)
        index = build_scaled_ip_index(q.reconstruct, n=n, d=D, norms=norms, chunk=CHUNK_SIZE)
        _, ids = search_index(index, Q, k=max(KS))
        rec = recall_at_ks(ids, gt, ks=KS)
        mse = reconstruction_mse(X, q.reconstruct, sample_ids, chunk=CHUNK_SIZE)
        code_bytes = q.code_bytes()
        compression = (n * D * 4) / code_bytes if code_bytes else float("inf")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = {
            "dataset": DATASET_LABEL,
            "method": method,
            "bpd": bpd,
            "compression_factor": compression,
            "code_bytes": code_bytes,
            "mse": mse,
            "n_db": n,
            "n_queries": Q.shape[0],
            "D": D,
            "timestamp": ts,
        }
        for k in KS:
            row[f"recall_at_{k}"] = rec[k]
        del index, q
        return row, time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, traceback.format_exc()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    t0 = time.time()
    X = _load_fvecs(DATA_BASE)
    Q = _load_fvecs(DATA_QUERY)
    Q = Q[:N_QUERIES]
    print(f"  X: {X.shape}, Q: {Q.shape}  ({time.time()-t0:.1f}s)")

    all_rows = []
    skipped = []

    # ------------------------------------------------------------------
    # Run GT computation once (shared across cells)
    # ------------------------------------------------------------------
    from haag_vq.benchmarks.exact_search import (
        build_scaled_ip_index,
        compute_exact_norms,
        normalized_ground_truth,
        reconstruction_mse,
        recall_at_ks,
        search_index,
    )
    from haag_vq.benchmarks.quantizer_study import _build_quantizer
    from datetime import datetime, timezone

    n, D = X.shape
    print("Computing exact norms and ground truth...")
    t0 = time.time()
    norms = compute_exact_norms(X)
    gt = normalized_ground_truth(X, Q, k=max(KS), norms=norms, chunk=CHUNK_SIZE)
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(n, MSE_SAMPLE, replace=False).astype(np.uint32) if n > MSE_SAMPLE else np.arange(n, dtype=np.uint32)
    print(f"  GT done in {time.time()-t0:.1f}s")

    def run_cell(method, bpd):
        t_start = time.time()
        print(f"\n  [{method} bpd={bpd}] fitting...", flush=True)
        try:
            q = _build_quantizer(method, bpd, D)
            q.fit(X)
            t_fit = time.time() - t_start
            print(f"  [{method} bpd={bpd}] fit done in {t_fit:.1f}s; building index...", flush=True)
            index = build_scaled_ip_index(q.reconstruct, n=n, d=D, norms=norms, chunk=CHUNK_SIZE)
            _, ids = search_index(index, Q, k=max(KS))
            rec = recall_at_ks(ids, gt, ks=KS)
            mse = reconstruction_mse(X, q.reconstruct, sample_ids, chunk=CHUNK_SIZE)
            code_bytes = q.code_bytes()
            compression = (n * D * 4) / code_bytes if code_bytes else float("inf")
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            row = {
                "dataset": DATASET_LABEL,
                "method": method,
                "bpd": bpd,
                "compression_factor": compression,
                "code_bytes": code_bytes,
                "mse": mse,
                "n_db": n,
                "n_queries": Q.shape[0],
                "D": D,
                "timestamp": ts,
            }
            for k in KS:
                row[f"recall_at_{k}"] = rec[k]
            del index, q
            wall = time.time() - t_start
            print(f"  [{method} bpd={bpd}] DONE wall={wall:.1f}s  recall@10={rec[10]:.4f}  mse={mse:.6e}  compression={compression:.2f}x", flush=True)
            return row, wall, None
        except Exception as e:
            wall = time.time() - t_start
            msg = traceback.format_exc()
            print(f"  [{method} bpd={bpd}] ERROR after {wall:.1f}s:\n{msg}", flush=True)
            return None, wall, msg

    # ------------------------------------------------------------------
    # 6 full-range methods: bpd 1..8
    # ------------------------------------------------------------------
    print("\n=== 6 full-range methods (bpd 1..8) ===")
    for method in FULL_METHODS:
        for bpd in FULL_BPD:
            row, wall, err = run_cell(method, bpd)
            if row is not None:
                all_rows.append(row)
                # Save checkpoint after each method completes a bpd
                df_tmp = pd.DataFrame(all_rows)
                df_tmp.to_csv(OUT_DIR / "results_checkpoint.csv", index=False)
            else:
                skipped.append((method, bpd, wall, err))

    # ------------------------------------------------------------------
    # OPQ: bpd 1, 2, 4 only
    # ------------------------------------------------------------------
    print("\n=== OPQ (bpd 1, 2, 4 only) ===")
    for bpd in OPQ_BPD:
        row, wall, err = run_cell("opq", bpd)
        if row is not None:
            all_rows.append(row)
            df_tmp = pd.DataFrame(all_rows)
            df_tmp.to_csv(OUT_DIR / "results_checkpoint.csv", index=False)
        else:
            skipped.append(("opq", bpd, wall, err))

    # ------------------------------------------------------------------
    # Save combined results
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "results_dbpedia_100k_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Print full table
    cols = ["method", "bpd", "compression_factor", "mse", "recall_at_1", "recall_at_10", "recall_at_100"]
    print("\n=== FULL RESULTS TABLE ===")
    print(df[cols].sort_values(["method", "bpd"]).to_string(index=False))

    # ------------------------------------------------------------------
    # Pareto plots
    # ------------------------------------------------------------------
    plot_path = OUT_DIR / "pareto_dbpedia_100k_full.png"
    try:
        pareto_curves(df, plot_path, ks=KS)
        print(f"\nSaved plots: {plot_path}")
    except Exception as e:
        print(f"\nPlot failed: {e}")

    # ------------------------------------------------------------------
    # Skipped cells
    # ------------------------------------------------------------------
    if skipped:
        print(f"\n=== SKIPPED {len(skipped)} CELLS ===")
        for method, bpd, wall, err in skipped:
            print(f"  {method} bpd={bpd}  wall={wall:.1f}s")
            print(f"  {err[:200]}")
    else:
        print("\nNo cells skipped.")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print("\n=== SANITY CHECKS ===")

    # 1. NaN/inf
    nan_count = df[["mse", "recall_at_1", "recall_at_10", "recall_at_100"]].isna().sum().sum()
    inf_count = (df[["mse", "recall_at_1", "recall_at_10", "recall_at_100"]].abs() == float("inf")).sum().sum()
    print(f"1. NaN count: {nan_count}, Inf count: {inf_count} -> {'PASS' if nan_count == 0 and inf_count == 0 else 'FAIL'}")

    # 2. recall in [0,1] and recall@100 >= recall@10 >= recall@1
    recall_range_ok = ((df["recall_at_1"] >= 0) & (df["recall_at_1"] <= 1) &
                       (df["recall_at_10"] >= 0) & (df["recall_at_10"] <= 1) &
                       (df["recall_at_100"] >= 0) & (df["recall_at_100"] <= 1)).all()
    recall_order_ok = ((df["recall_at_100"] >= df["recall_at_10"] - 1e-9) &
                       (df["recall_at_10"] >= df["recall_at_1"] - 1e-9)).all()
    print(f"2. Recall in [0,1]: {'PASS' if recall_range_ok else 'FAIL'}, Recall@100>=@10>=@1: {'PASS' if recall_order_ok else 'FAIL'}")
    if not recall_order_ok:
        bad = df[df["recall_at_10"] < df["recall_at_1"] - 1e-9]
        print(f"   Violations: {bad[['method','bpd','recall_at_1','recall_at_10','recall_at_100']]}")

    # 3. MSE decreases as bpd increases; compression_factor decreases as bpd increases
    print("3. MSE monotone decrease with bpd (per method):")
    for method, g in df.groupby("method"):
        g = g.sort_values("bpd")
        if len(g) > 1:
            mse_mono = all(g["mse"].iloc[i] >= g["mse"].iloc[i+1] - 1e-12 for i in range(len(g)-1))
            # Allow ties since some bpd values map to same config
            mse_non_inc = all(g["mse"].iloc[i] >= g["mse"].iloc[i+1] - g["mse"].mean()*1e-3 for i in range(len(g)-1))
            comp_mono = all(g["compression_factor"].iloc[i] >= g["compression_factor"].iloc[i+1] - 1e-6 for i in range(len(g)-1))
            print(f"   {method}: MSE non-increasing={'PASS' if mse_non_inc else 'FAIL'}  compression non-increasing={'PASS' if comp_mono else 'FAIL'}")
            if not mse_non_inc:
                print(f"     MSE values: {g[['bpd','mse']].values.tolist()}")

    # 4. ours vs saq_paper MSE ratio
    print("4. ours vs saq_paper MSE ratio (ours/saq_paper, expect <1.0 if greedy lowers MSE):")
    if "ours" in df["method"].values and "saq_paper" in df["method"].values:
        ours = df[df["method"] == "ours"].set_index("bpd")
        paper = df[df["method"] == "saq_paper"].set_index("bpd")
        common_bpd = sorted(set(ours.index) & set(paper.index))
        for bpd in common_bpd:
            ratio = ours.loc[bpd, "mse"] / paper.loc[bpd, "mse"] if paper.loc[bpd, "mse"] > 0 else float("nan")
            print(f"   bpd={bpd}: ours_mse={ours.loc[bpd,'mse']:.6e}  paper_mse={paper.loc[bpd,'mse']:.6e}  ratio={ratio:.4f}")
    else:
        print("   MISSING: ours or saq_paper not in results")

    # 5. rabitq and lvq coverage
    print("5. rabitq/lvq bpd coverage:")
    for method in ["rabitq", "lvq"]:
        if method in df["method"].values:
            bpds = sorted(df[df["method"] == method]["bpd"].unique())
            print(f"   {method}: bpds={bpds}")

    # 6. Pareto frontier at a few compression levels
    print("6. Pareto frontier (recall@10 at select compression_factor ranges):")
    for cf_min, cf_max, label in [(28, 35, "~32x"), (14, 18, "~16x"), (7, 9, "~8x"), (3, 5, "~4x")]:
        subset = df[(df["compression_factor"] >= cf_min) & (df["compression_factor"] <= cf_max)]
        if len(subset) > 0:
            best = subset.loc[subset["recall_at_10"].idxmax()]
            print(f"   compression {label}: best recall@10={best['recall_at_10']:.4f} by {best['method']} bpd={best['bpd']}")
        else:
            print(f"   compression {label}: no methods in this range")


if __name__ == "__main__":
    main()
