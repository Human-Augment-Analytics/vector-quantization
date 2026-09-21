"""Claim battery at large d: the 200-trial synthetic non-convex test from
_lp_joint_alloc.py, but with d drawn from 1024..1536 (paper-scale dims).
Checks Claim 1 (x integral given integer z), Claim 2 (nnz(z) <= 8), rounding
byte overhead, and — on a subset — LP+round vs exact MILP. Prints a summary."""
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lpmod", HERE / "_lp_joint_alloc.py")
lpmod = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, spec.origin, "exec"), lpmod.__dict__)

RNG = np.random.default_rng(1)
lpmod.RNG = RNG   # synth_curves uses the module RNG

trials = 200
rows = []
c1_viol = c2_viol = 0
t_start = time.time()
for t in range(trials):
    d = int(RNG.integers(1024, 1537))
    mse = lpmod.synth_curves(d, nonconvex_frac=0.6)
    bpd = float(RNG.uniform(0.5, 7.5))
    B = max(1, int(d * bpd / 8))
    t0 = time.time()
    r = lpmod.run_instance(mse, B, do_milp=(t < 20), milp_tl=120.0)
    r.update(d=d, bpd=bpd, solve_s=round(time.time() - t0, 2))
    rows.append(r)
    if r["n_frac_x"] > 0: c1_viol += 1
    if r["n_nonzero_z"] > 8: c2_viol += 1
    if (t + 1) % 25 == 0:
        print(f"{t+1}/{trials}  elapsed {time.time()-t_start:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "lp_claims_larged.csv", index=False)
milp_rows = df[df.milp_ok]
gap = (df.lp_round_obj - df.lp_bound) / df.lp_bound
print(f"\n=== large-d claim battery: {trials} trials, d in [1024,1536], 95%-non-convex curves ===")
print(f"Claim 1 (x integral given integer z): {c1_viol} violations  "
      f"(max fractional entries in any trial: {int(df.n_frac_x.max())})")
print(f"Claim 2 (nnz(z) <= 8 at LP vertex):  {c2_viol} violations  "
      f"(max nnz seen: {int(df.n_nonzero_z.max())})")
print(f"Rounding byte overhead: max +{int(df.bytes_over.max())} "
      f"(median +{int(df.bytes_over.median())})")
print(f"MILP subset ({len(milp_rows)} solved of 20 attempted): "
      f"max |lp_round_obj - milp|/milp = "
      f"{(abs(milp_rows.lp_round_obj - milp_rows.milp_obj) / milp_rows.milp_obj).max():.2e}")
print(f"LP+round vs LP bound: max rel gap {gap.max():.2e} (negative = beats exact-B optimum "
      f"using the rounded extra bytes; min {gap.min():.2e})")
print(f"solve time per instance: median {df.solve_s.median():.1f}s  max {df.solve_s.max():.1f}s")
