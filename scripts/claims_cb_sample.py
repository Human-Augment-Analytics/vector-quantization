"""Claim 1 (k-means half): codebooks trained on a finite sample don't hurt.

Sweeps the Lloyd codebook training-sample size (VQ_CB_SAMPLE, read inside
RankAwareQuantizer) and measures MSE + recall at fixed allocation/bpd. The
expected chart: both flat down to small samples (quality saturates well
before full N; the production default is 200K).

Env (run_full_benchmark conventions):
    VQ_DATA_DIR   dataset dir (vectors.fvecs, queries.fvecs)   [required]
    VQ_OUT_DIR    output dir                                   [required]
    VQ_CB_SAMPLE  codebook training-sample size for this run   [required]
    VQ_METHODS    comma list                                   [default lp_ra1]
    VQ_BPD        comma list                                   [default 2,4]
    VQ_N_QUERIES                                               [default 20000]
    VQ_GT_CACHE / VQ_MMAP / VQ_PCA_CACHE as usual.

Output: cb_sample.csv (one row per method x bpd at this sample size).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from haag_vq.benchmarks.quantizer_study import _load_fvecs, _build_quantizer
from haag_vq.benchmarks.exact_search import (
    compute_exact_norms, normalized_ground_truth, recall_at_ks,
    reconstruction_mse, stream_search_scaled_ip,
)

DATA_DIR = os.environ["VQ_DATA_DIR"]
OUT_DIR = Path(os.environ["VQ_OUT_DIR"])
CB_SAMPLE = int(os.environ["VQ_CB_SAMPLE"])
METHODS = os.environ.get("VQ_METHODS", "lp_ra1").split(",")
BPDS = [float(b) for b in os.environ.get("VQ_BPD", "2,4").split(",")]
N_QUERIES = int(os.environ.get("VQ_N_QUERIES", "20000"))
KS = (1, 10, 100)
CHUNK = 50_000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VQ_CB_SAMPLE"] = str(CB_SAMPLE)  # consumed by RankAwareQuantizer
    X = _load_fvecs(f"{DATA_DIR}/vectors.fvecs")
    Q = _load_fvecs(f"{DATA_DIR}/queries.fvecs")[:N_QUERIES]
    n, D = X.shape

    gt_cache = os.environ.get("VQ_GT_CACHE") or os.path.join(
        DATA_DIR, f"gt_cache_n{n}_q{N_QUERIES}_k{max(KS)}_mse100000.npz")
    assert os.path.exists(gt_cache), f"GT cache required at scale: {gt_cache}"
    z = np.load(gt_cache)
    norms, gt, sample_ids = z["norms"], z["gt"], z["sample_ids"]

    rows = []
    for method in METHODS:
        for bpd in BPDS:
            t0 = time.time()
            q = _build_quantizer(method, bpd, D)
            q.fit(X)
            _, ids = stream_search_scaled_ip(q.reconstruct, n=n, d=D, norms=norms,
                                             Q=Q, k=max(KS), chunk=CHUNK)
            rec = recall_at_ks(ids, gt, ks=KS)
            mse = reconstruction_mse(X, q.reconstruct, sample_ids, chunk=CHUNK)
            rows.append({"dataset": os.path.basename(DATA_DIR.rstrip("/")),
                         "method": method, "bpd": bpd, "cb_sample": CB_SAMPLE,
                         "mse": mse,
                         **{f"recall_at_{k}": rec[k] for k in KS},
                         "n_db": n, "n_queries": Q.shape[0]})
            print(f"[{method} bpd={bpd} cb={CB_SAMPLE}] mse={mse:.4e} "
                  f"r@10={rec[10]:.4f} ({time.time()-t0:.1f}s)", flush=True)
            del q
            pd.DataFrame(rows).to_csv(OUT_DIR / "cb_sample.csv", index=False)

    print(f"wrote {OUT_DIR}/cb_sample.csv")


if __name__ == "__main__":
    main()
