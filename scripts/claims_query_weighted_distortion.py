"""Query-weighted distortion column for the main method table.

For each (method, bpd) computes, in the ORIGINAL input space (comparable across
methods regardless of their internal rotations):

    mse_d  = mean_i (X[i,d] - X_hat[i,d])^2       per-dim MSE on a row sample
    wdist  = sum_d E[q_d^2] * mse_d               recall-noise proxy
    mse_sum = sum_d mse_d                         (= D x the usual per-dim MSE;
                                                  cross-check vs main table)

E[q_d^2] is the uncentered second moment over the query set — the same
query-energy quantity validated in the beta study (E[q_d^2] ~ var_d^beta).

Env (run_full_benchmark conventions):
    VQ_DATA_DIR   dataset dir (vectors.fvecs, queries.fvecs)   [required]
    VQ_OUT_DIR    output dir                                   [required]
    VQ_METHODS    comma list       [default pq,sq,rabitq,lvq,lp_mse,lp_ra05,lp_ra1]
    VQ_BPD        comma list                                   [default 1,2,4,8]
    VQ_N_QUERIES  queries for E[q^2]                           [default 20000]
    VQ_SAMPLE_N   rows for per-dim MSE                         [default 200000]
    VQ_MMAP=1, VQ_PCA_CACHE supported as usual.

Output: query_weighted_distortion.csv
"""
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from haag_vq.benchmarks.quantizer_study import _load_fvecs, _build_quantizer

DATA_DIR = os.environ["VQ_DATA_DIR"]
OUT_DIR = Path(os.environ["VQ_OUT_DIR"])
METHODS = os.environ.get("VQ_METHODS", "pq,sq,rabitq,lvq,lp_mse,lp_ra05,lp_ra1").split(",")
BPDS = [float(b) for b in os.environ.get("VQ_BPD", "1,2,4,8").split(",")]
N_QUERIES = int(os.environ.get("VQ_N_QUERIES", "20000"))
SAMPLE_N = int(os.environ.get("VQ_SAMPLE_N", "200000"))
CHUNK = 50_000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = _load_fvecs(f"{DATA_DIR}/vectors.fvecs")
    Q = _load_fvecs(f"{DATA_DIR}/queries.fvecs")[:N_QUERIES]
    n, D = X.shape

    Eq2 = (np.asarray(Q, dtype=np.float64) ** 2).mean(axis=0)  # (D,)
    print(f"X {X.shape}  Q {Q.shape}  sum E[q^2]={Eq2.sum():.4f}")

    rng = np.random.default_rng(0)
    sample_ids = (rng.choice(n, SAMPLE_N, replace=False) if n > SAMPLE_N
                  else np.arange(n))
    sample_ids = np.sort(sample_ids)

    rows = []
    for method in METHODS:
        for bpd in BPDS:
            t0 = time.time()
            try:
                q = _build_quantizer(method, bpd, D)
                q.fit(X)
                sq_sum = np.zeros(D, dtype=np.float64)
                for lo in range(0, len(sample_ids), CHUNK):
                    ids = sample_ids[lo: lo + CHUNK]
                    diff = np.asarray(X[ids], dtype=np.float32) - q.reconstruct(ids)
                    sq_sum += (diff.astype(np.float64) ** 2).sum(axis=0)
                mse_d = sq_sum / len(sample_ids)
                wdist = float((Eq2 * mse_d).sum())
                rows.append({
                    "dataset": os.path.basename(DATA_DIR.rstrip("/")),
                    "method": method, "bpd": bpd,
                    "wdist": wdist,
                    "mse_sum": float(mse_d.sum()),
                    "wdist_over_uniform": wdist / (Eq2.mean() * mse_d.sum()),
                    "code_bytes_per_vec": q.code_bytes() / n,
                    "n_sample": len(sample_ids), "n_queries": Q.shape[0], "D": D,
                })
                print(f"[{method} bpd={bpd}] wdist={wdist:.4e} mse_sum={mse_d.sum():.4e} "
                      f"({time.time()-t0:.1f}s)", flush=True)
                del q
            except Exception:
                print(f"[{method} bpd={bpd}] ERROR\n{traceback.format_exc()}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / "query_weighted_distortion.csv", index=False)

    print(f"wrote {OUT_DIR}/query_weighted_distortion.csv")


if __name__ == "__main__":
    main()
