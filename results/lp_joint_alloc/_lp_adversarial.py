"""Adversarial search against the joint-LP claims (follow-up to _lp_joint_alloc.py).

Targets, in order of attackability:
  A. Rank precheck: can 9 z-columns even be simultaneously basic? The vertex
     counting bound allows nnz(z) <= 9 (8 slot rows + 1 budget row); Claim 2
     says <= 8. If no 9 partition-vectors (A[.,c], 1) are linearly independent,
     9 is impossible and Claim 2 is proved; otherwise it's a search target.
  B. Claim 2 hunt: hill-climb over instances (per-dim retention curves, budget)
     maximizing nnz(z) at the highs-ds vertex optimum, seeded with random
     instances and constructed "all 8 widths in use" instances. Sideways moves
     allowed; secondary score = number of tight non-assignment rows.
  C. Claim 1 degeneracy battery: instances built for maximal ties (costs on a
     coarse grid, duplicated dims) run through the full LP->round->x-LP
     pipeline; count fractional x in what the solver actually returns.

Also tracks the max rounding byte-overshoot seen anywhere.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lpmod", HERE / "_lp_joint_alloc.py")
lpmod = importlib.util.module_from_spec(spec)
src = open(spec.origin).read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(src, spec.origin, "exec"), lpmod.__dict__)

RNG = np.random.default_rng(7)
TOL = 1e-6
MAXB = 8
A, PARTS = lpmod.A, lpmod.PARTS   # (8, 22) slot counts, partitions of 8


# ---------------------------------------------------------------- A: rank check
def rank_precheck():
    V = np.vstack([A, np.ones(A.shape[1])])          # (9, 22): columns (A[.,c], 1)
    r = np.linalg.matrix_rank(V)
    print(f"[A] rank of the 22 partition columns in the 9 tight-row space: {r}")
    if r < 9:
        print("[A] -> 9 simultaneously-basic z's are IMPOSSIBLE; Claim 2 (<=8) is proved.")
    else:
        print("[A] -> 9 independent columns exist; nnz(z)=9 is not ruled out. Hunting.")
    return r


# --------------------------------------------------------- instance machinery
def curves_from(var, drops):
    d = var.shape[0]
    mse = np.empty((d, 9))
    mse[:, 0] = var
    mse[:, 1:] = var[:, None] * np.cumprod(drops, axis=1)
    return mse

def rand_instance(d):
    var = RNG.lognormal(-2, 2, d)
    drops = RNG.uniform(0.1, 0.98, (d, MAXB))
    B = max(1, int(d * RNG.uniform(0.5, 7.5) / 8))
    return var, drops, B

def all_widths_instance(d):
    """Tiered spectrum: ~d/8 dims per tier, tier g tuned to want ~g bits, so the
    optimum spreads over many widths and many slot rows bind at once."""
    var = np.repeat(np.geomspace(1.0, 1e-6, 8), d // 8)
    var = np.pad(var, (0, d - var.size), constant_values=var[-1])
    var *= RNG.lognormal(0, 0.05, d)                 # break exact ties slightly
    drops = np.tile(RNG.uniform(0.28, 0.45, (1, MAXB)), (d, 1))
    drops *= RNG.lognormal(0, 0.03, (d, MAXB))
    np.clip(drops, 0.05, 0.99, out=drops)
    B = max(1, int(d * RNG.uniform(2.5, 5.5) / 8))
    return var, drops, B

def score(mse, B):
    """(nnz_z, n_tight_rows) at the highs-ds vertex optimum."""
    _, x, z = lpmod.solve_lp(mse, B)
    nnz = int((z > TOL).sum())
    usage = x[:, 1:].sum(axis=0)                      # demand per width
    cap = A @ z
    tight = int((np.abs(usage - cap) < 1e-7).sum()) + int(abs(z.sum() - B) < 1e-7)
    return nnz, tight


# ------------------------------------------------------------- B: claim-2 hunt
def hunt_nnz(restarts=40, steps=60, d=48):
    best = (0, 0)
    best_desc = None
    t0 = time.time()
    for r in range(restarts):
        var, drops, B = (all_widths_instance(d) if r % 2 else rand_instance(d))
        cur = score(curves_from(var, drops), B)
        for _ in range(steps):
            nv, nd, nB = var.copy(), drops.copy(), B
            move = RNG.integers(4)
            if move == 0:
                k = RNG.integers(1, max(2, d // 8))
                idx = RNG.choice(d, k, replace=False)
                nd[idx] = RNG.uniform(0.1, 0.98, (k, MAXB))
            elif move == 1:
                nd *= RNG.lognormal(0, 0.08, nd.shape)
                np.clip(nd, 0.05, 0.99, out=nd)
            elif move == 2:
                nB = max(1, B + int(RNG.integers(-2, 3)))
            else:
                idx = RNG.choice(d, max(1, d // 10), replace=False)
                nv[idx] *= RNG.lognormal(0, 0.5, idx.size)
            s = score(curves_from(nv, nd), nB)
            if s >= cur:                              # greedy with sideways moves
                var, drops, B, cur = nv, nd, nB, s
            if cur > best:
                best, best_desc = cur, (var.copy(), drops.copy(), B)
                if best[0] >= 9:
                    print(f"[B] *** nnz(z) = {best[0]} FOUND (restart {r}) ***")
                    np.savez(HERE / "claim2_counterexample.npz",
                             var=var, drops=drops, B=B)
                    return best
        if (r + 1) % 10 == 0:
            print(f"[B] restart {r+1}/{restarts}: best nnz={best[0]} "
                  f"(tight rows {best[1]}/9)  {time.time()-t0:.0f}s", flush=True)
    return best


# ------------------------------------------------------ C: claim-1 tie battery
def tie_battery(trials=400):
    viol = 0
    max_over = 0
    max_frac_seen = 0.0
    for t in range(trials):
        d = int(RNG.integers(16, 65))
        kind = t % 4
        if kind == 0:      # coarse-grid costs: many exact ties
            mse = np.sort(RNG.integers(0, 12, (d, 9)), axis=1)[:, ::-1] / 8.0
            mse[:, 0] += 1.0
        elif kind == 1:    # duplicated dims: identical cost rows
            base = curves_from(*rand_instance(max(4, d // 4))[:2])
            mse = np.repeat(base, 4, axis=0)[:d]
        elif kind == 2:    # power-of-two curves: deltas coincide across dims
            mse = np.outer(2.0 ** -RNG.integers(0, 4, d), 2.0 ** -np.arange(9))
        else:              # random non-convex (control)
            var, drops, _ = rand_instance(d)
            mse = curves_from(var, drops)
        B = max(1, int(d * RNG.uniform(0.5, 7.5) / 8))
        r = lpmod.lp_round(mse, B)
        if r["n_frac_x"] > 0:
            viol += 1
            print(f"[C] fractional x! trial {t} kind={kind} d={d} B={B} "
                  f"n_frac={r['n_frac_x']}")
        max_over = max(max_over, r["bytes_used"] - B)
        if (t + 1) % 100 == 0:
            print(f"[C] {t+1}/{trials}: {viol} violations, max overshoot +{max_over}",
                  flush=True)
    return viol, max_over


if __name__ == "__main__":
    t0 = time.time()
    rank = rank_precheck()
    if rank >= 9:
        best = hunt_nnz()
        print(f"[B] hunt done: max nnz(z) = {best[0]} (tight rows {best[1]}/9)")
    viol, over = tie_battery()
    print(f"[C] tie battery: {viol} fractional-x violations; "
          f"max rounding overshoot +{over} bytes")
    print(f"total {time.time()-t0:.0f}s")
