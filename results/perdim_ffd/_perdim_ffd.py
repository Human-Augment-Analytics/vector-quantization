"""Per-dim + FFD (professor's method) on dbpedia-100K, exact search.
Sweeps avg_bits x alpha, packing='ffd', reporting recall + REAL FFD compression
(incl. the +4B/vec exact-norm side-channel, matching the 7-method benchmark)."""
from pathlib import Path

import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import _load_fvecs
from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
from haag_vq.benchmarks import exact_search as es

OUT = Path("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/perdim_ffd")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k"

X = _load_fvecs(f"{BASE}/vectors.fvecs")
Q = _load_fvecs(f"{BASE}/queries.fvecs")[:1000]
n, D = X.shape
ks = (1, 10, 100)
NORM_B = 4  # exact-norm side-channel, charged to every method
print(f"loaded X={X.shape} Q={Q.shape}", flush=True)

norms = es.compute_exact_norms(X)
gt = es.normalized_ground_truth(X, Q, k=max(ks), norms=norms, chunk=50000)
sample = (np.arange(n, dtype=np.uint32) if n <= 100000
          else np.random.default_rng(0).choice(n, 100000, replace=False).astype(np.uint32))

rows = []
for avg_bits in [1, 2, 3, 4]:
    for alpha in [0.0, 0.5]:
        print(f"fitting avg_bits={avg_bits} alpha={alpha} packing=ffd ...", flush=True)
        q = RankAwareQuantizer(avg_bits=avg_bits, alpha=alpha, max_bits=8, packing="ffd")
        q.fit(X)
        codes = q.compress(X)
        ffd_bytes = int(codes.shape[1])
        dense_bytes = int(np.ceil(q.bits.sum() / 8))

        def recon(ids, _c=codes, _q=q):
            return _q.decompress(_c[ids])

        idx = es.build_scaled_ip_index(recon, n, D, norms, chunk=50000)
        _, ids = es.search_index(idx, Q, max(ks))
        rec = es.recall_at_ks(ids, gt, ks)
        mse = es.reconstruction_mse(X, recon, sample, chunk=50000)
        comp_ffd = D * 4 / (ffd_bytes + NORM_B)
        comp_dense = D * 4 / (dense_bytes + NORM_B)
        row = {"avg_bits": avg_bits, "alpha": alpha, "mse": mse,
               **{f"r@{k}": rec[k] for k in ks},
               "bits_sum": int(q.bits.sum()), "dense_bytes": dense_bytes, "ffd_bytes": ffd_bytes,
               "ffd_overhead_pct": 100.0 * (ffd_bytes - dense_bytes) / dense_bytes,
               "comp_ffd": comp_ffd, "comp_dense": comp_dense}
        rows.append(row)
        print(f"  -> r@10={rec[10]:.4f} mse={mse:.3e} ffd_bytes={ffd_bytes} "
              f"(dense {dense_bytes}, +{row['ffd_overhead_pct']:.1f}%) comp_ffd={comp_ffd:.2f}x", flush=True)
        del idx, codes, q

df = pd.DataFrame(rows)
df.to_csv(OUT / "perdim_ffd.csv", index=False)
print("\n=== PER-DIM + FFD (dbpedia, exact, 1000q) ===", flush=True)
print(df[["avg_bits", "alpha", "r@10", "mse", "bits_sum", "ffd_bytes",
          "ffd_overhead_pct", "comp_ffd"]].to_string(index=False), flush=True)
print("\nReference per-block engine (from 7-method run): "
      "ours r@10 0.922@~15.8x(b2) 0.969@~8.0x(b4); saq_paper 0.939@15.8x 0.981@8.0x", flush=True)
print("DONE_PERDIM_FFD", flush=True)
