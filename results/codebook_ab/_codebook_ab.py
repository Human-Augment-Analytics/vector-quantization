"""Phase 2 A/B: does an EXACT 1-D codebook beat Lloyd (and analytic Gaussian) on
RECALL, holding per-dim allocation + FFD packing fixed?

Per-dim quantizer (alpha fixed) with codebook in {gaussian, lloyd, exact}; the only
thing that changes is the per-dim centroids. Exact-search recall@{1,10,100} + MSE on
dbpedia. This isolates the codebook's effect on recall (the Phase-2 question: the
exact codebook's lower MSE — 6.6% at 8bpd — does it move recall?).
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import _load_fvecs
from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
from haag_vq.benchmarks import exact_search as es

OUT = Path(os.environ.get("VQ_OUT_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/codebook_ab"))
BASE = os.environ.get("VQ_DATA_DIR",
    "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k")
ALPHA = float(os.environ.get("ALPHA", 0.5))
BPDS = [int(b) for b in os.environ.get("BPDS", "2,4,6").split(",")]
NQ = int(os.environ.get("NQ", 1000))
OUT.mkdir(parents=True, exist_ok=True)

X = _load_fvecs(f"{BASE}/vectors.fvecs")
Q = _load_fvecs(f"{BASE}/queries.fvecs")[:NQ]
n, D = X.shape
ks = (1, 10, 100)
print(f"X={X.shape} Q={Q.shape} alpha={ALPHA} bpds={BPDS}", flush=True)

norms = es.compute_exact_norms(X)
gt = es.normalized_ground_truth(X, Q, k=max(ks), norms=norms, chunk=50000)
sample = (np.arange(n, dtype=np.uint32) if n <= 100000
          else np.random.default_rng(0).choice(n, 100000, replace=False).astype(np.uint32))

rows = []
for avg_bits in BPDS:
    for codebook in ("gaussian", "lloyd", "exact"):
        import time; t = time.time()
        q = RankAwareQuantizer(avg_bits=avg_bits, alpha=ALPHA, max_bits=8,
                               packing="ffd", codebook=codebook)
        q.fit(X)
        codes = q.compress(X)
        recon = lambda ids, _c=codes, _q=q: _q.decompress(_c[ids])
        idx = es.build_scaled_ip_index(recon, n, D, norms, chunk=50000)
        _, ids = es.search_index(idx, Q, max(ks))
        rec = es.recall_at_ks(ids, gt, ks)
        mse = es.reconstruction_mse(X, recon, sample, chunk=50000)
        comp = D * 4 / (int(codes.shape[1]) + 4)
        rows.append({"avg_bits": avg_bits, "codebook": codebook, "mse": mse,
                     **{f"r@{k}": rec[k] for k in ks}, "comp": comp,
                     "bits_sum": int(q.bits.sum()), "fit_s": time.time() - t})
        print(f"  bpd={avg_bits} {codebook:8s}: r@10={rec[10]:.4f} r@1={rec[1]:.4f} "
              f"mse={mse:.3e} comp={comp:.2f}x ({time.time()-t:.0f}s)", flush=True)
        del idx, codes, q

df = pd.DataFrame(rows)
df.to_csv(OUT / "codebook_ab.csv", index=False)
print("\n=== Phase 2 A/B: codebook effect on recall (dbpedia, per-dim+FFD) ===", flush=True)
print(df[["avg_bits", "codebook", "mse", "r@1", "r@10", "r@100"]].to_string(index=False), flush=True)
# deltas vs gaussian
for ab in BPDS:
    sub = df[df.avg_bits == ab].set_index("codebook")
    g = sub.loc["gaussian"]
    print(f"  bpd={ab}: exact-vs-lloyd r@10 {sub.loc['exact','r@10']-sub.loc['lloyd','r@10']:+.4f}, "
          f"exact-vs-gaussian r@10 {sub.loc['exact','r@10']-g['r@10']:+.4f}", flush=True)
print("DONE_CODEBOOK_AB", flush=True)
