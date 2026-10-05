"""Audit of the Gaussian scaling law as the LP's cost table (gap-2 experiment).

Controlled A/B on msmarco_500k: lp_ra1 with cost source = scaling law
(w_j * lambda_j * Dg(l)) vs empirical (w_j * measured per-dim Lloyd losses on
the 200K codebook sample), everything else fixed (weights, codebooks, budget,
eval). Reports allocation overlap, MSE, recall per bpd.

Run (WSL):  python results/lp_joint_alloc/_cost_source_ab.py
Writes cost_source_ab.csv next to this file.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
DATA = REPO.parent / "SAQ" / "data" / "datasets" / "msmarco_500k"

from haag_vq.benchmarks.method_registry import build_quantizer
from haag_vq.benchmarks.quantizer_study import _load_fvecs
from haag_vq.benchmarks.exact_search import (
    recall_at_ks, reconstruction_mse, stream_search_scaled_ip)

os.environ["VQ_MMAP"] = "1"
BPDS = [1.0, 2.0, 4.0, 8.0]
KS = (1, 10, 100)
N_QUERIES = 5000
CHUNK = 50_000


def main():
    X = _load_fvecs(str(DATA / "base.fvecs"))
    Q = _load_fvecs(str(DATA / "query.fvecs"))[:N_QUERIES]
    n, D = X.shape
    z = np.load(DATA / f"gt_cache_n{n}_q{N_QUERIES}_k100_mse100000.npz")
    norms, gt, sample_ids = z["norms"], z["gt"], z["sample_ids"]
    print(f"X {X.shape}, GT loaded")

    rows, bits_store = [], {}
    for bpd in BPDS:
        for src in ("scaling", "empirical"):
            os.environ["VQ_COST_SOURCE"] = src
            t0 = time.time()
            q = build_quantizer("lp_ra1", bpd, D)
            q.fit(X)
            bits = np.asarray(q._q.bits).copy()
            bits_store[(bpd, src)] = bits
            _, ids = stream_search_scaled_ip(q.reconstruct, n=n, d=D, norms=norms,
                                             Q=Q, k=max(KS), chunk=CHUNK)
            rec = recall_at_ks(ids, gt, ks=KS)
            mse = reconstruction_mse(X, q.reconstruct, sample_ids, chunk=CHUNK)
            row = {"bpd": bpd, "cost_source": src, "mse": mse,
                   "bytes_per_vec": q.code_bytes() / n,
                   **{f"recall_at_{k}": rec[k] for k in KS}}
            if src == "empirical":
                b0 = bits_store[(bpd, "scaling")]
                row["alloc_overlap"] = float((bits == b0).mean())
                row["mean_abs_dbits"] = float(np.abs(bits - b0).mean())
                row["max_abs_dbits"] = int(np.abs(bits - b0).max())
            rows.append(row)
            print(f"[bpd={bpd} {src}] mse={mse:.4e} r@1={rec[1]:.4f} "
                  f"r@10={rec[10]:.4f} ({time.time()-t0:.0f}s)"
                  + (f"  overlap={row.get('alloc_overlap'):.4f}" if src == "empirical" else ""),
                  flush=True)
            del q
            pd.DataFrame(rows).to_csv(HERE / "cost_source_ab.csv", index=False)

    print("wrote", HERE / "cost_source_ab.csv")


if __name__ == "__main__":
    main()
