"""LP rounding battery, MSMARCO edition (paper tab:lp-rounding redo).

Changes vs _lp_joint_alloc.py / _lp_claims_larged.py:
- Real cost curves = MSMARCO 200K per-dim k-means losses (d=1024), not the
  1536-d corpus.
- Synthetic curves: monotone random per-bit reduction factors drawn uniform
  from [0.2, 1], sorted decreasing -- NO non-convexity swap trick.
- Exact MILP reference solved for EVERY instance (spawn pool; fork after
  HiGHS deadlocks).

Prints the table rows: instances, fractional b_{j,l} after fixing integer
r_p, max nonzero r_p, max byte overshoot, LP bound == integer optimum,
within 0.1%, worst gap. Writes lp_rounding_msmarco.csv next to this script.
"""
import argparse
import importlib.util
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MSMARCO_CURVES = ("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/research-hub/"
                  "vq-results/msmarco_200k/approx_codebook/approx_codebook.csv")

spec = importlib.util.spec_from_file_location("lpmod", HERE / "_lp_joint_alloc.py")
lpmod = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, spec.origin, "exec"), lpmod.__dict__)


def load_msmarco_curves(path=MSMARCO_CURVES):
    df = pd.read_csv(path)
    d = int(df["dim"].max()) + 1
    mse = np.zeros((d, 9))
    mse[:, 0] = df.groupby("dim")["var"].first().values
    for _, r in df.iterrows():
        mse[int(r["dim"]), int(r["bits"])] = r["lloyd_mse"]
    return mse


def synth_curves_plain(rng, d):
    """Monotone curves: per-bit reduction factors uniform [0.2, 1], sorted
    decreasing (largest reduction from the first bit). No swap trick."""
    var = rng.lognormal(-2, 1.5, d)
    mse = np.zeros((d, 9))
    mse[:, 0] = var
    for i in range(d):
        v = var[i]
        drops = np.sort(rng.uniform(0.2, 1.0, 8))[::-1]
        for j in range(1, 9):
            v = v * drops[j - 1]
            mse[i, j] = v
    return mse


def solve_one(args):
    key, mse, B = args
    obj, ok = lpmod.solve_milp(mse, B, 300.0)
    return key, obj, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-small", type=int, default=200)
    ap.add_argument("--trials-large", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--bpd-max", type=float, default=8.0)
    args = ap.parse_args()

    instances = []  # (name, mse, B)
    mse_real = load_msmarco_curves()
    d = mse_real.shape[0]
    for bpd in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        instances.append((f"real_bpd{bpd}", mse_real, int(d * bpd / 8)))

    rng = np.random.default_rng(0)
    for t in range(args.trials_small):
        d_s = int(rng.integers(16, 129))
        m = synth_curves_plain(rng, d_s)
        bpd = float(rng.uniform(0.5, args.bpd_max))
        instances.append((f"synth_small{t}", m, max(1, int(d_s * bpd / 8))))
    for t in range(args.trials_large):
        d_s = int(rng.integers(1024, 1537))
        m = synth_curves_plain(rng, d_s)
        bpd = float(rng.uniform(0.5, args.bpd_max))
        instances.append((f"synth_large{t}", m, max(1, int(d_s * bpd / 8))))

    rows = {}
    t0 = time.time()
    for name, m, B in instances:
        res = lpmod.lp_round(m, B)
        rows[name] = dict(name=name, d=m.shape[0], B=B,
                          lp_bound=res["lp_bound"], n_nonzero_z=res["n_nonzero_z"],
                          n_frac_x=res["n_frac_x"],
                          bytes_over=res["bytes_used"] - B)
    print(f"LP+round on {len(instances)} instances in {time.time()-t0:.0f}s", flush=True)

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        key, obj, ok = ex.submit(solve_one, (instances[0][0], instances[0][1], instances[0][2])).result(timeout=300)
        rows[key].update(milp_obj=obj, milp_ok=ok)
        print(f"smoke: {key} ok={ok}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = {ex.submit(solve_one, (n, m, B)): n for n, m, B in instances[1:]}
        for i, f in enumerate(as_completed(futs), 1):
            key, obj, ok = f.result()
            rows[key].update(milp_obj=obj, milp_ok=ok)
            if i % 50 == 0 or i == len(futs):
                print(f"milp {i}/{len(futs)}  {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows.values())
    df.to_csv(HERE / "lp_rounding_msmarco.csv", index=False)

    for grp, sub in (("real", df[df.name.str.startswith("real")]),
                     ("synthetic", df[df.name.str.startswith("synth")])):
        ok = sub[sub.milp_ok == True]
        rel = (ok.lp_bound - ok.milp_obj).abs() / ok.milp_obj
        print(f"\n=== {grp}: {len(sub)} instances (MILP solved {len(ok)}) ===")
        print(f"fractional b after fixing r: {int(sub.n_frac_x.sum())}")
        print(f"max nonzero r_p: {int(sub.n_nonzero_z.max())}")
        print(f"max byte overshoot: {int(sub.bytes_over.max())}  "
              f"dist {sub.bytes_over.value_counts().sort_index().to_dict()}")
        print(f"LP bound == integer optimum (<1e-9): {(rel < 1e-9).sum()}/{len(ok)}")
        print(f"within 0.1%: {(rel < 1e-3).sum()}/{len(ok)}")
        print(f"worst gap: {rel.max():.3e}")


if __name__ == "__main__":
    main()
