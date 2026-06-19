"""MSMARCO-500K confirmation run, exact search, 10K held-out queries.
Checkpoints after every (method, bpd) cell so a late failure loses nothing.
Computes GT + norms once; finding-critical methods first."""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import _load_fvecs
from haag_vq.benchmarks.method_registry import build_quantizer
from haag_vq.benchmarks import exact_search as es

OUT = Path("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/msmarco_500k")
OUT.mkdir(parents=True, exist_ok=True)
D_DIR = "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/msmarco_500k"
KS = (1, 10, 100)
CHUNK = 50000

X = _load_fvecs(f"{D_DIR}/base.fvecs")
Q = _load_fvecs(f"{D_DIR}/query.fvecs")
n, D = X.shape
print(f"loaded base={X.shape} query={Q.shape}", flush=True)

norms = es.compute_exact_norms(X)
gt = es.normalized_ground_truth(X, Q, k=max(KS), norms=norms, chunk=CHUNK)
print("GT computed", flush=True)
sample = (np.arange(n, dtype=np.uint32) if n <= 100000
          else np.random.default_rng(0).choice(n, 100000, replace=False).astype(np.uint32))

# finding-critical first (fix + divergence), baselines last
METHODS = ["rankaware", "perdim_mse", "saq_paper", "ours", "rabitq", "pq", "sq"]
BPDS = [1, 2, 3, 4, 5, 6]
ckpt = OUT / "results_checkpoint.csv"
rows = []

for method in METHODS:
    for bpd in BPDS:
        print(f"[{method} bpd={bpd}] ...", flush=True)
        try:
            q = build_quantizer(method, bpd, D)
            q.fit(X)
            idx = es.build_scaled_ip_index(q.reconstruct, n, D, norms, chunk=CHUNK)
            _, ids = es.search_index(idx, Q, max(KS))
            rec = es.recall_at_ks(ids, gt, ks=KS)
            mse = es.reconstruction_mse(X, q.reconstruct, sample, chunk=CHUNK)
            cb = int(q.code_bytes())
            row = {"dataset": "msmarco_500k", "method": method, "bpd": bpd,
                   "compression_factor": (n * D * 4) / cb, "code_bytes": cb, "mse": mse,
                   "n_db": n, "n_queries": Q.shape[0], "D": D,
                   **{f"recall_at_{k}": rec[k] for k in KS}}
            rows.append(row)
            pd.DataFrame(rows).to_csv(ckpt, index=False)  # checkpoint each cell
            print(f"  done r@10={rec[10]:.4f} mse={mse:.3e} comp={row['compression_factor']:.2f}x", flush=True)
            del idx, q
        except Exception as e:
            print(f"  CELL FAILED {method} bpd={bpd}: {repr(e)[:200]}", flush=True)

df = pd.DataFrame(rows)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
df.to_csv(OUT / f"results_full_{stamp}.csv", index=False)
try:
    from haag_vq.benchmarks.study_plots import pareto_curves
    pareto_curves(df, OUT / f"pareto_{stamp}.png", ks=KS)
except Exception as e:
    print("plot failed:", repr(e)[:150], flush=True)
print("\n=== MSMARCO-500K RESULTS ===", flush=True)
print(df[["method", "bpd", "compression_factor", "mse", "recall_at_1", "recall_at_10", "recall_at_100"]].to_string(index=False), flush=True)
print("DONE_MSMARCO_500K", flush=True)
