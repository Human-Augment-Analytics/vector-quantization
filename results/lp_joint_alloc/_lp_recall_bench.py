"""Recall-bridge test of the joint allocation+packing LP (follow-up to
_lp_joint_alloc.py): does the LP's MSE win over byte-constrained greedy at
4-6 bpd survive into recall on dbpedia-100K (exact reconstruct+search)?

Variants at byte budget round(bpd*D/8):
  allocator=greedy_bytes : rank-aware greedy, largest bit budget that packs
                           into the byte budget (budget-fair 2-stage baseline)
  allocator=lp           : joint LP + rounding at the byte budget
  allocator=greedy       : plain bit-budget greedy (packed bytes fall where
                           they fall — context for what extra bytes buy)
alpha in {0, 0.5} (0 = plain MSE objective; 0.5 = validated rank-aware weight).

Runs on Windows Python (large allocations crash WSL). SMOKE=1 subsamples for a
fast plumbing check. Writes lp_recall_bench.csv next to this file.
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
DATA = REPO.parent / "SAQ" / "data" / "datasets" / "dbpedia_100k"

from haag_vq.benchmarks import exact_search as es


def _load_fvecs(path: str) -> np.ndarray:
    # inlined from quantizer_study to avoid its yaml/config import chain on Windows
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    d = int(data[0].view(np.int32))
    return data.reshape(-1, d + 1)[:, 1:].copy()

from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
from haag_vq.methods.lp_alloc import lp_joint_alloc

SMOKE = os.environ.get("SMOKE") == "1"
DATASET = os.environ.get("VQ_DATASET", "dbpedia")   # dbpedia | msmarco500k | sift10m

if DATASET == "sift10m":
    import h5py
    MAT = REPO.parent / "SAQ" / "data" / "datasets" / "sift10m" / "SIFT10M" / "SIFT10Mfeatures.mat"
    with h5py.File(str(MAT), "r") as f:
        fea = f["fea"]                       # (11164866, 128) uint8
        n_all = 6000 if SMOKE else fea.shape[0]
        raw = fea[:n_all]                    # lazy slice read
    rng = np.random.default_rng(0)
    q_ids = rng.choice(n_all, 1000, replace=False)
    q_mask = np.zeros(n_all, dtype=bool); q_mask[q_ids] = True
    Q = raw[q_ids].astype(np.float32)
    X = raw[~q_mask].astype(np.float32)      # held-out queries, disjoint base
    del raw
    BPDS, ALPHAS = (4, 6), (0.0, 0.5)
elif DATASET == "msmarco500k":
    DATA = REPO.parent / "SAQ" / "data" / "datasets" / "msmarco_500k"
    X = _load_fvecs(str(DATA / "base.fvecs"))
    Q = _load_fvecs(str(DATA / "query.fvecs"))[:1000]
    BPDS, ALPHAS = (4, 6), (0.0, 0.5)               # trimmed grid at 500K scale
else:
    X = _load_fvecs(str(DATA / "vectors.fvecs"))
    Q = _load_fvecs(str(DATA / "queries.fvecs"))[:1000]
    BPDS, ALPHAS = (2, 4, 6), (0.0, 0.5)
if SMOKE:
    X = X[:5000]
    Q = Q[:50]
if os.environ.get("VQ_ALPHAS"):
    ALPHAS = tuple(float(a) for a in os.environ["VQ_ALPHAS"].split(","))
n, D = X.shape
ks = (1, 10, 100)
print(f"loaded {DATASET}: X={X.shape} Q={Q.shape} SMOKE={SMOKE}", flush=True)

norms = es.compute_exact_norms(X)
gt = es.normalized_ground_truth(X, Q, k=max(ks), norms=norms, chunk=50000)
sample = (np.arange(n, dtype=np.uint32) if n <= 100000
          else np.random.default_rng(0).choice(n, 100000, replace=False).astype(np.uint32))

rows = []
for bpd in BPDS:
    for alpha in ALPHAS:
        allocs = ("greedy_bytes", "lp") + (("greedy",) if alpha == 0.5 else ())
        for alloc in allocs:
            t0 = time.time()
            q = RankAwareQuantizer(avg_bits=bpd, alpha=alpha, packing="ffd",
                                   codebook="gaussian", allocator=alloc)
            q.fit(X)
            fit_s = time.time() - t0
            codes = q.compress(X)

            def recon(ids, _codes=codes, _q=q):
                return _q.decompress(_codes[ids])

            idx = es.build_scaled_ip_index(recon, n, D, norms, chunk=50000)
            _, ids = es.search_index(idx, Q, max(ks))
            rec = es.recall_at_ks(ids, gt, ks)
            mse = es.reconstruction_mse(X, recon, sample, chunk=50000)
            bits_sum = int(q.bits.sum())
            code_bytes = int(q.code_size)
            util = bits_sum / (8 * code_bytes)
            hist = {j: int((q.bits == j).sum()) for j in range(9) if (q.bits == j).any()}
            row = {"bpd": bpd, "alpha": alpha, "allocator": alloc,
                   "code_bytes": code_bytes, "bits_sum": bits_sum,
                   "bits_per_byte": round(8 * util, 3), "mse": mse,
                   **{f"r@{k}": rec[k] for k in ks},
                   "fit_s": round(fit_s, 1), "widths": str(hist)}
            rows.append(row)
            print(f"bpd={bpd} alpha={alpha} alloc={alloc:12s} bytes={code_bytes} "
                  f"bits={bits_sum} ({8*util:.2f}b/B) mse={mse:.4e} "
                  f"r@10={rec[10]:.4f} fit={fit_s:.0f}s", flush=True)
            del idx, codes, q

df = pd.DataFrame(rows)
suffix = ("_smoke" if SMOKE else "") + ("" if DATASET == "dbpedia" else f"_{DATASET}")
if os.environ.get("VQ_ALPHAS"):
    suffix += "_a" + os.environ["VQ_ALPHAS"].replace(",", "_")
out = HERE / f"lp_recall_bench{suffix}.csv"
df.to_csv(out, index=False)
print(f"\nwrote {out}", flush=True)
print(df[["bpd", "alpha", "allocator", "code_bytes", "bits_per_byte", "mse",
          "r@1", "r@10", "r@100"]].to_string(index=False), flush=True)

# LP runtime at MSMARCO scale (D=1024): the other follow-up.
rng = np.random.default_rng(0)
var = np.sort(rng.lognormal(-2, 2, 1024))[::-1]
Dg = np.array([1.0, 0.3634, 0.1175, 0.03454, 0.009497, 0.002499, 0.000640, 0.000162, 4.1e-05])
cost = var[:, None] * Dg[None, :]
for B in (256, 512, 768):
    t0 = time.time()
    bits, nnz = lp_joint_alloc(cost, B)
    print(f"LP runtime D=1024 B={B}: {time.time()-t0:.2f}s (nnz_z={nnz})", flush=True)
