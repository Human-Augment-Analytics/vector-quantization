"""Complete the MILP reference for the 340 synthetic LP instances that
_lp_joint_alloc.py (seed 0, t>=40) and _lp_claims_larged.py (seed 1, t>=20)
skipped. Replays the seeded instance streams, verifies each reproduced
instance against the existing CSVs (d, B exact; lp_bound to 1e-6 rel) before
solving, then fans the MILPs over a process pool.

Writes milp_fill.csv next to this script and prints final combined counts.
"""
import importlib.util
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LP_DIR = Path(__file__).resolve().parent
HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("lpmod", LP_DIR / "_lp_joint_alloc.py")
lpmod = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, spec.origin, "exec"), lpmod.__dict__)

MILP_TL = 300.0


def gen_stream(seed: int, trials: int, d_lo: int, d_hi: int):
    """Replay one script's instance stream; yields (t, d, bpd, B, mse)."""
    rng = np.random.default_rng(seed)
    lpmod.RNG = rng
    for t in range(trials):
        d = int(rng.integers(d_lo, d_hi))
        mse = lpmod.synth_curves(d, nonconvex_frac=0.6)
        bpd = float(rng.uniform(0.5, 7.5))
        B = max(1, int(d * bpd / 8))
        yield t, d, bpd, B, mse


def solve_one(args):
    key, mse, B = args
    t0 = time.time()
    obj, ok = lpmod.solve_milp(mse, B, MILP_TL)
    return key, obj, ok, time.time() - t0


def main():
    small = pd.read_csv(LP_DIR / "lp_joint_alloc.csv")
    small = small[small.dataset.str.startswith("synth")].reset_index(drop=True)
    large = pd.read_csv(LP_DIR / "lp_claims_larged.csv").reset_index(drop=True)
    assert len(small) == 200 and len(large) == 200

    jobs = []
    for name, seed, lo, hi, df, skip in (("small", 0, 16, 129, small, 40),
                                         ("large", 1, 1024, 1537, large, 20)):
        n_checked = 0
        for t, d, bpd, B, mse in gen_stream(seed, 200, lo, hi):
            row = df.iloc[t]
            assert int(row.d) == d and int(row.B) == B, \
                f"{name} t={t}: replay mismatch d {d} vs {row.d}, B {B} vs {row.B}"
            if t < skip:
                continue  # already has a MILP reference
            # lp_bound check on a sample (LP at large d costs ~1s; check every
            # 10th large instance, every small instance)
            if name == "small" or t % 10 == 0:
                lp_obj, _, _ = lpmod.solve_lp(mse, B)
                rel = abs(lp_obj - row.lp_bound) / abs(row.lp_bound)
                assert rel < 1e-6, f"{name} t={t}: lp_bound rel diff {rel:.2e}"
                n_checked += 1
            jobs.append(((name, t), mse, B))
        print(f"{name}: replay verified (d,B all 200; lp_bound on {n_checked}), "
              f"{sum(1 for j in jobs if j[0][0] == name)} MILPs queued", flush=True)

    # spawn, not fork: the parent has run HiGHS (verification LPs), and forked
    # children inherit its thread state and spin forever (observed: 14 workers
    # at 100% CPU for an hour with zero completions)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        key, obj, ok, secs = ex.submit(solve_one, jobs[0]).result(timeout=120)
        print(f"smoke test: {key} solved={ok} obj={obj:.6g} in {secs:.1f}s", flush=True)
    out.append(dict(src=jobs[0][0][0], t=jobs[0][0][1], milp_obj=obj, milp_ok=ok,
                    solve_s=round(secs, 2)))
    jobs = jobs[1:]
    with ProcessPoolExecutor(max_workers=14, mp_context=ctx) as ex:
        futs = {ex.submit(solve_one, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            key, obj, ok, secs = f.result()
            out.append(dict(src=key[0], t=key[1], milp_obj=obj, milp_ok=ok,
                            solve_s=round(secs, 2)))
            if i % 20 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}  elapsed {time.time()-t0:.0f}s", flush=True)

    fill = pd.DataFrame(out).sort_values(["src", "t"]).reset_index(drop=True)
    fill.to_csv(HERE / "milp_fill.csv", index=False)
    print(f"wrote {HERE/'milp_fill.csv'}  "
          f"(timeouts: {(~fill.milp_ok).sum()}, max solve {fill.solve_s.max():.0f}s)")

    # combined final counts over all 400 synthetic instances
    small["src"], large["src"] = "small", "large"
    small["t"] = range(200)
    large["t"] = range(200)
    allrows = pd.concat([small, large], ignore_index=True)
    allrows = allrows.merge(fill, on=["src", "t"], how="left", suffixes=("", "_new"))
    allrows["milp_obj"] = allrows.milp_obj_new.combine_first(allrows.milp_obj)
    allrows["milp_ok"] = allrows.milp_ok.astype(bool) | allrows.milp_ok_new.fillna(False).astype(bool)
    ok = allrows[allrows.milp_ok]
    rel = (ok.lp_bound - ok.milp_obj).abs() / ok.milp_obj
    print(f"\n=== synthetic, all {len(allrows)} instances ===")
    print(f"MILP solved: {len(ok)}/{len(allrows)}")
    print(f"lp_bound == milp (rel < 1e-9): {(rel < 1e-9).sum()}/{len(ok)}  "
          f"worst rel gap {rel.max():.3e}")
    for thr in (1e-6, 1e-4, 1e-3):
        print(f"  within {thr:g}: {(rel < thr).sum()}")


if __name__ == "__main__":
    main()
