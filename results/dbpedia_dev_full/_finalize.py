"""Finish the 6 missing cells (ours 6,7,8 + opq 1,2,4), merge with the
checkpoint, write the final combined CSV + Pareto plots, print sanity."""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import run_study_arrays, _load_fvecs
from haag_vq.benchmarks.study_plots import pareto_curves

BASE = "/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/data/datasets/dbpedia_100k"
OUT = Path("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/vector-quantization/results/dbpedia_dev_full")

X = _load_fvecs(f"{BASE}/vectors.fvecs")
Q = _load_fvecs(f"{BASE}/queries.fvecs")[:1000]
print(f"loaded X={X.shape} Q={Q.shape}", flush=True)


def run(methods, bpds):
    df = run_study_arrays(X, Q, methods=methods, bpd_values=bpds,
                          ks=(1, 10, 100), chunk_size=50000, mse_sample=100000)
    df.insert(0, "dataset", "dbpedia_100k")
    return df


parts = []
ckpt = OUT / "results_checkpoint.csv"
if ckpt.exists():
    parts.append(pd.read_csv(ckpt))
    print(f"checkpoint rows: {len(parts[0])}", flush=True)

print("running ours [6,7,8] ...", flush=True)
d_ours = run(["ours"], [6, 7, 8])
d_ours.to_csv(OUT / "_progress_ours678.csv", index=False)
parts.append(d_ours)
print("ours done", flush=True)

print("running opq [1,2,4] ...", flush=True)
d_opq = run(["opq"], [1, 2, 4])
parts.append(d_opq)
print("opq done", flush=True)

df = pd.concat(parts, ignore_index=True)
df = df.drop_duplicates(subset=["dataset", "method", "bpd"], keep="last")
df = df.sort_values(["method", "bpd"]).reset_index(drop=True)

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
csv = OUT / f"results_full_{stamp}.csv"
df.to_csv(csv, index=False)
pareto_curves(df, OUT / f"pareto_full_{stamp}.png", ks=(1, 10, 100))
print(f"WROTE {csv}", flush=True)

cols = ["method", "bpd", "compression_factor", "mse",
        "recall_at_1", "recall_at_10", "recall_at_100"]
print(df[cols].to_string(index=False), flush=True)
print(f"NaNs present: {bool(df.isna().any().any())}", flush=True)

piv = df.pivot_table(index="bpd", columns="method", values="mse")
for b in (2, 4, 6, 8):
    if b in piv.index and {"ours", "saq_paper"} <= set(piv.columns):
        o, s = piv.loc[b, "ours"], piv.loc[b, "saq_paper"]
        if pd.notna(o) and pd.notna(s):
            print(f"bpd={b}: ours/saq_paper MSE ratio = {o / s:.3f}", flush=True)
print("DONE_FINALIZE", flush=True)
