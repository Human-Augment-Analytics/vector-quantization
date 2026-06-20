"""alpha-sweep of the rank-aware quantizer on dbpedia-100K, exact search.
Tests whether concentrating bits on the high-variance head (higher alpha) buys
recall that the MSE-optimal allocation (alpha=0) leaves on the table."""
from pathlib import Path

import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import _load_fvecs
from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
from haag_vq.benchmarks import exact_search as es

OUT = Path("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/rankaware_sweep")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k"

X = _load_fvecs(f"{BASE}/vectors.fvecs")
Q = _load_fvecs(f"{BASE}/queries.fvecs")[:1000]
n, D = X.shape
ks = (1, 10, 100)
print(f"loaded X={X.shape} Q={Q.shape}", flush=True)

norms = es.compute_exact_norms(X)
gt = es.normalized_ground_truth(X, Q, k=max(ks), norms=norms, chunk=50000)
sample = (np.arange(n, dtype=np.uint32) if n <= 100000
          else np.random.default_rng(0).choice(n, 100000, replace=False).astype(np.uint32))

rows = []
for bpd in [2, 4]:
    for alpha in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"fitting bpd={bpd} alpha={alpha} ...", flush=True)
        q = RankAwareQuantizer(avg_bits=bpd, alpha=alpha)
        q.fit(X)
        codes = q.compress(X)

        def recon(ids, _codes=codes, _q=q):
            return _q.decompress(_codes[ids])

        idx = es.build_scaled_ip_index(recon, n, D, norms, chunk=50000)
        _, ids = es.search_index(idx, Q, max(ks))
        rec = es.recall_at_ks(ids, gt, ks)
        mse = es.reconstruction_mse(X, recon, sample, chunk=50000)
        head = int(q.bits[:D // 2].sum()); tail = int(q.bits[D // 2:].sum())
        code_bytes = int(np.ceil(q.bits.sum() / 8))
        row = {"bpd": bpd, "alpha": alpha, "mse": mse,
               **{f"r@{k}": rec[k] for k in ks},
               "bits_sum": int(q.bits.sum()), "head_bits": head, "tail_bits": tail,
               "code_bytes": code_bytes, "compression": D * 4 / code_bytes}
        rows.append(row)
        print(f"  -> mse={mse:.3e} r@10={rec[10]:.4f} head/tail={head}/{tail}", flush=True)
        del idx, codes, q

df = pd.DataFrame(rows)
df.to_csv(OUT / "rankaware_sweep.csv", index=False)
print("\n=== RANK-AWARE ALPHA SWEEP (dbpedia-100K, exact, 1000q) ===", flush=True)
print(df[["bpd", "alpha", "mse", "r@1", "r@10", "r@100", "head_bits", "tail_bits"]].to_string(index=False), flush=True)
for bpd in [2, 4]:
    d = df[df.bpd == bpd].sort_values("alpha")
    best = d.loc[d["r@10"].idxmax()]
    a0 = d[d.alpha == 0.0].iloc[0]
    print(f"\nbpd={bpd}: r@10 by alpha -> " +
          ", ".join(f"a{a}:{r:.4f}" for a, r in zip(d.alpha, d['r@10'])), flush=True)
    print(f"  best alpha={best.alpha} (r@10={best['r@10']:.4f}) vs alpha=0 (r@10={a0['r@10']:.4f}); "
          f"delta={best['r@10']-a0['r@10']:+.4f}; mse a0={a0.mse:.3e} -> best={best.mse:.3e}", flush=True)
print("\nDONE_SWEEP", flush=True)
