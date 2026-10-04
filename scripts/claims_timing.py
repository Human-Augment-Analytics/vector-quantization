"""Quantization-time comparison: full index-build wall time per method/bpd at
two corpus sizes, from which a fixed-train vs per-vector-encode split follows:

    t(N) ~= T_train + N / encode_vps
    encode_vps = (N2 - N1) / (t2 - t1),  T_train = t1 - N1/encode_vps

Two-point measurement deliberately avoids touching quantizer internals (fit()
trains AND encodes in every implementation here). Threads are whatever the
node provides — record cpus in the row and keep it constant across methods
(submit all methods with the same --cpus-per-task).

Env (run_full_benchmark conventions):
    VQ_DATA_DIR   dataset dir (vectors.fvecs)                 [required]
    VQ_OUT_DIR    output dir                                  [required]
    VQ_METHODS    comma list                                  [default lp_ra1]
    VQ_BPD        comma list                                  [default 1,2,4,8]
    VQ_TIMING_NS  comma list of corpus sizes                  [default 200000,2000000]
    VQ_MMAP=1 supported as usual.

Output: timing.csv (one row per method x bpd x N, plus derived columns on the
larger-N rows when the pair exists).
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
METHODS = os.environ.get("VQ_METHODS", "lp_ra1").split(",")
BPDS = [float(b) for b in os.environ.get("VQ_BPD", "1,2,4,8").split(",")]
NS = [int(n) for n in os.environ.get("VQ_TIMING_NS", "200000,2000000").split(",")]
CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = _load_fvecs(f"{DATA_DIR}/vectors.fvecs")
    D = X.shape[1]
    assert max(NS) <= X.shape[0]

    rows = []
    for method in METHODS:
        for bpd in BPDS:
            fits = {}
            for n in sorted(NS):
                try:
                    q = _build_quantizer(method, bpd, D)
                    t0 = time.time()
                    q.fit(X[:n])
                    fits[n] = time.time() - t0
                    code_bytes = q.code_bytes()
                    del q
                    row = {"dataset": os.path.basename(DATA_DIR.rstrip("/")),
                           "method": method, "bpd": bpd, "n": n,
                           "fit_s": round(fits[n], 2),
                           "bytes_per_vec": code_bytes / n, "cpus": CPUS, "D": D}
                    # Derived split once both sizes are in
                    if len(fits) == 2:
                        (n1, t1), (n2, t2) = sorted(fits.items())
                        if t2 > t1:
                            vps = (n2 - n1) / (t2 - t1)
                            row["encode_vps"] = round(vps)
                            row["train_s_est"] = round(t1 - n1 / vps, 2)
                    rows.append(row)
                    print(f"[{method} bpd={bpd} n={n}] fit {fits[n]:.1f}s", flush=True)
                except Exception:
                    print(f"[{method} bpd={bpd} n={n}] ERROR\n{traceback.format_exc()}",
                          flush=True)
                pd.DataFrame(rows).to_csv(OUT_DIR / "timing.csv", index=False)

    print(f"wrote {OUT_DIR}/timing.csv")


if __name__ == "__main__":
    main()
