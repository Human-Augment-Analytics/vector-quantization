"""Claim: "cosine similarity requires normalized vectors, and quantized vectors
are almost normalized."

For each (method, bpd): reconstruct the corpus, measure the distribution of
||x_hat|| (inputs are unit-norm, so ~1 means the norm side channel is nearly
redundant), and compare recall under three scoring rules against the same
exact-vector ground truth:

    exactnorm : q . x_hat / ||x||_exact   (the paper's current eval — needs the
                stored 4 B/vec norm)
    selfnorm  : q . x_hat / ||x_hat||     (deployable cosine, NO stored norm)
    rawip     : q . x_hat                 (plain IP, no normalization at all)

If selfnorm ~= exactnorm, Table 5's +4 B/vec norm channel can be dropped.

Env (run_full_benchmark conventions):
    VQ_DATA_DIR   dataset dir (vectors.fvecs, queries.fvecs)   [required]
    VQ_OUT_DIR    output dir                                   [required]
    VQ_METHODS    comma list                                   [default lp_ra1]
    VQ_BPD        comma list                                   [default 1,2,4,8]
    VQ_N_QUERIES                                               [default 1000]
    VQ_GT_CACHE   gt cache npz (else default path / computed)
    VQ_MMAP=1, VQ_PCA_CACHE supported as usual.

Outputs: norm_hist_summary.csv (stats + recalls), norm_hist_bins.csv.
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
    stream_search_scaled_ip,
)

DATA_DIR = os.environ["VQ_DATA_DIR"]
OUT_DIR = Path(os.environ["VQ_OUT_DIR"])
METHODS = os.environ.get("VQ_METHODS", "lp_ra1").split(",")
BPDS = [float(b) for b in os.environ.get("VQ_BPD", "1,2,4,8").split(",")]
N_QUERIES = int(os.environ.get("VQ_N_QUERIES", "1000"))
KS = (1, 10, 100)
CHUNK = 50_000
BIN_EDGES = np.linspace(0.0, 1.5, 76)  # 75 bins of width 0.02


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = _load_fvecs(f"{DATA_DIR}/vectors.fvecs")
    Q = _load_fvecs(f"{DATA_DIR}/queries.fvecs")[:N_QUERIES]
    n, D = X.shape
    print(f"X {X.shape}  Q {Q.shape}")

    gt_cache = os.environ.get("VQ_GT_CACHE") or os.path.join(
        DATA_DIR, f"gt_cache_n{n}_q{N_QUERIES}_k{max(KS)}_mse100000.npz")
    if os.path.exists(gt_cache):
        z = np.load(gt_cache)
        norms, gt = z["norms"], z["gt"]
        print(f"loaded GT cache {gt_cache}")
    else:
        print("computing GT (no cache found)...")
        norms = compute_exact_norms(X)
        gt = normalized_ground_truth(X, Q, k=max(KS), norms=norms, chunk=CHUNK)
        rng = np.random.default_rng(0)
        sample_ids = (rng.choice(X.shape[0], 100_000, replace=False).astype(np.uint32)
                      if X.shape[0] > 100_000 else np.arange(X.shape[0], dtype=np.uint32))
        tmp = f"{gt_cache}.tmp.{os.getpid()}.npz"
        np.savez(tmp, norms=norms, gt=gt, sample_ids=sample_ids)
        os.replace(tmp, gt_cache)
        print(f"cached GT -> {gt_cache}")

    summary, bins = [], []
    for method in METHODS:
        for bpd in BPDS:
            t0 = time.time()
            q = _build_quantizer(method, bpd, D)
            q.fit(X)
            print(f"[{method} bpd={bpd}] fit {time.time()-t0:.1f}s", flush=True)

            # ||x_hat|| over the whole corpus, chunked
            rnorms = np.empty(n, dtype=np.float32)
            for lo in range(0, n, CHUNK):
                ids = np.arange(lo, min(lo + CHUNK, n))
                rnorms[ids] = np.linalg.norm(q.reconstruct(ids), axis=1)
            stats = {
                "mean": float(rnorms.mean()), "sd": float(rnorms.std()),
                **{f"p{p}": float(np.percentile(rnorms, p)) for p in (1, 5, 50, 95, 99)},
                "min": float(rnorms.min()), "max": float(rnorms.max()),
            }
            counts, _ = np.histogram(rnorms, bins=BIN_EDGES)
            for lo_e, c in zip(BIN_EDGES[:-1], counts):
                bins.append({"method": method, "bpd": bpd,
                             "bin_lo": round(float(lo_e), 3), "count": int(c)})

            # Three scoring rules, same reconstructions, same GT
            recs = {}
            for label, denom in (("exactnorm", norms),
                                 ("selfnorm", np.maximum(rnorms, 1e-12)),
                                 ("rawip", np.ones(n, dtype=np.float32))):
                _, ids = stream_search_scaled_ip(q.reconstruct, n=n, d=D, norms=denom,
                                                 Q=Q, k=max(KS), chunk=CHUNK)
                r = recall_at_ks(ids, gt, ks=KS)
                for k in KS:
                    recs[f"{label}_r{k}"] = r[k]
                print(f"  {label}: " + " ".join(f"r@{k}={r[k]:.4f}" for k in KS), flush=True)

            summary.append({"method": method, "bpd": bpd, "n": n, **stats, **recs})
            pd.DataFrame(summary).to_csv(OUT_DIR / "norm_hist_summary.csv", index=False)
            pd.DataFrame(bins).to_csv(OUT_DIR / "norm_hist_bins.csv", index=False)
            del q

    print(f"wrote {OUT_DIR}/norm_hist_summary.csv (+bins)")


if __name__ == "__main__":
    main()
