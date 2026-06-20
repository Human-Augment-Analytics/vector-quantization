"""2x2 ablation: {DP, greedy} x {default-CB, kmeans-CB} on dbpedia-100K, exact
search, to attribute the ours-vs-saq_paper recall gap to codebook vs allocation.
Bit-allocation plans go to stderr (SAQ glog), tagged by the markers we print."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import _load_fvecs
from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter
from haag_vq.benchmarks import exact_search as es

OUT = Path("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/ablation_2x2")
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

configs = {
    "DP+default":     dict(greedy=False, derive_codebooks=False),
    "DP+kmeans":      dict(greedy=False, derive_codebooks=True),
    "greedy+default": dict(greedy=True,  derive_codebooks=False),
    "greedy+kmeans":  dict(greedy=True,  derive_codebooks=True),
}

rows = []
for bpd in [2, 4]:
    for name, cfg in configs.items():
        # marker on stderr so it interleaves with the engine's glog plan dump
        print(f"@@@ PLAN MARKER: {name} bpd={bpd}", file=sys.stderr, flush=True)
        print(f"running {name} bpd={bpd} ...", flush=True)
        q = SaqEngineAdapter(quant_type="CAQ", avg_bits=bpd, **cfg)
        q.fit(X)
        idx = es.build_scaled_ip_index(q.reconstruct, n, D, norms, chunk=50000)
        _, ids = es.search_index(idx, Q, max(ks))
        rec = es.recall_at_ks(ids, gt, ks)
        mse = es.reconstruction_mse(X, q.reconstruct, sample, chunk=50000)
        row = {"config": name, "bpd": bpd,
               "alloc": name.split("+")[0], "codebook": name.split("+")[1],
               "mse": mse, **{f"r@{k}": rec[k] for k in ks}}
        rows.append(row)
        print(f"  -> mse={mse:.3e} r@10={rec[10]:.4f}", flush=True)
        del idx, q

df = pd.DataFrame(rows)
df.to_csv(OUT / "ablation_2x2.csv", index=False)
print("\n=== 2x2 ABLATION TABLE ===", flush=True)
print(df.to_string(index=False), flush=True)

for bpd in [2, 4]:
    d = df[df.bpd == bpd].set_index("config")
    print(f"\n--- attribution @ bpd={bpd} (delta r@10) ---", flush=True)
    print(f"  codebook (kmeans-default) | DP:     {d.loc['DP+kmeans','r@10']-d.loc['DP+default','r@10']:+.4f}", flush=True)
    print(f"  codebook (kmeans-default) | greedy: {d.loc['greedy+kmeans','r@10']-d.loc['greedy+default','r@10']:+.4f}", flush=True)
    print(f"  alloc    (greedy-DP)      | default:{d.loc['greedy+default','r@10']-d.loc['DP+default','r@10']:+.4f}", flush=True)
    print(f"  alloc    (greedy-DP)      | kmeans: {d.loc['greedy+kmeans','r@10']-d.loc['DP+kmeans','r@10']:+.4f}", flush=True)
    print(f"  total    (ours-saq_paper):          {d.loc['greedy+kmeans','r@10']-d.loc['DP+default','r@10']:+.4f}", flush=True)
print("\nDONE_ABLATION", flush=True)
