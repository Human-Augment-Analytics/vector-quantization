"""Joint bit-allocation + byte-packing LP (research-group note, 2026-09-10).

The professor's proposal: one LP does allocation AND packing jointly, then rounding
gives an integral solution within <= 8 bytes of the optimal allocation, beating the
2-stage (allocate -> pack) pipeline's potential suboptimality.

    vars:  x[i,j] >= 0  (dim i assigned j bits, j = 0..8)
           z[c]   >= 0  (number of bytes with bit-partition c; c over partitions of 8)
    s.t.   sum_c z[c] <= B                                (byte budget)
           sum_i x[i,j] <= sum_c A[j,c] z[c]   (j=1..8)   (slots for j-bit groups)
           sum_j x[i,j] == 1                              (each dim gets one level)
    min    sum_{i,j} x[i,j] * mse[i][j]

(The note says "maximize"; with MSE coefficients the intended sense is minimize —
equivalently maximize MSE reduction vs 0 bits. The note counts 21 byte partitions;
enumeration gives p(8) = 22 — all 22 are used here.)

Experiments:
  E1  convexity scan of the real per-dim MSE curves (the note's premise).
  E2  Claim 2: #nonzero z at an LP vertex optimum <= 8.
  E3  Claim 1: with z fixed integer, vertex-optimal x is integral.
  E4  end-to-end: LP+rounding vs 2-stage (greedy alloc -> production FFD pack)
      vs exact MILP, on real dbpedia curves across byte budgets, plus synthetic
      non-convex instances for the claim checks.

Real MSE curves: results/approx_codebook/approx_codebook.csv (1536 dims; 'var' is
the 0-bit MSE, 'lloyd_mse' the k-means MSE at 1..8 bits).

Writes results/lp_joint_alloc/lp_joint_alloc.csv and prints a summary.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, hstack, eye

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from haag_vq.methods.ffd_packing import ffd_layout  # production packer

RNG = np.random.default_rng(0)
TOL = 1e-6
MAXB = 8


# ---------------------------------------------------------------------------
# Byte partitions of 8 and the A[j,c] slot-count matrix
# ---------------------------------------------------------------------------
def partitions_of(n: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    def rec(rem, mx, cur):
        if rem == 0:
            out.append(tuple(cur)); return
        for p in range(min(rem, mx), 0, -1):
            cur.append(p); rec(rem - p, p, cur); cur.pop()
    rec(n, n, [])
    return out

PARTS = partitions_of(8)
C = len(PARTS)
A = np.zeros((MAXB, C), dtype=np.int64)          # A[j-1, c] = count of j-bit groups in c
for c, part in enumerate(PARTS):
    for p in part:
        A[p - 1, c] += 1


# ---------------------------------------------------------------------------
# LP construction. Variable order: x (d*9, index i*9+j), then z (C).
# ---------------------------------------------------------------------------
def build_lp(mse: np.ndarray, B: int):
    d = mse.shape[0]
    nx, nz = d * 9, C
    cost = np.concatenate([mse.reshape(-1), np.zeros(nz)])

    # A_eq: each dim picks one level
    rows = np.repeat(np.arange(d), 9)
    cols = np.arange(nx)
    A_eq = csr_matrix((np.ones(nx), (rows, cols)), shape=(d, nx + nz))
    b_eq = np.ones(d)

    # A_ub: slots per bit level (8 rows) + budget (1 row)
    ub_rows, ub_cols, ub_vals = [], [], []
    for j in range(1, MAXB + 1):
        for i in range(d):
            ub_rows.append(j - 1); ub_cols.append(i * 9 + j); ub_vals.append(1.0)
        for c in range(C):
            if A[j - 1, c]:
                ub_rows.append(j - 1); ub_cols.append(nx + c); ub_vals.append(-float(A[j - 1, c]))
    for c in range(C):
        ub_rows.append(MAXB); ub_cols.append(nx + c); ub_vals.append(1.0)
    A_ub = csr_matrix((ub_vals, (ub_rows, ub_cols)), shape=(MAXB + 1, nx + nz))
    b_ub = np.concatenate([np.zeros(MAXB), [float(B)]])
    return cost, A_ub, b_ub, A_eq, b_eq, nx, nz


def _obj_scale(mse: np.ndarray) -> float:
    """MSE entries are ~1e-7; HiGHS tolerances then terminate early (seen: 1.7%
    off at 6 bpd). Normalize the objective to O(1) and unscale on return."""
    m = float(np.abs(mse).mean())
    return 1.0 / m if m > 0 else 1.0


def solve_lp(mse: np.ndarray, B: int):
    s = _obj_scale(mse)
    cost, A_ub, b_ub, A_eq, b_eq, nx, nz = build_lp(mse * s, B)
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=(0, None), method="highs-ds")   # dual simplex -> vertex solution
    assert r.status == 0, f"LP failed: {r.message}"
    x = r.x[:nx].reshape(-1, 9)
    z = r.x[nx:]
    return r.fun / s, x, z


def solve_x_given_slots(mse: np.ndarray, slots: np.ndarray):
    """Fix integer z (as per-level slot counts); re-solve the x-only LP (Claim 1)."""
    d = mse.shape[0]
    nx = d * 9
    rows = np.repeat(np.arange(d), 9)
    A_eq = csr_matrix((np.ones(nx), (rows, np.arange(nx))), shape=(d, nx))
    ub_rows, ub_cols = [], []
    for j in range(1, MAXB + 1):
        for i in range(d):
            ub_rows.append(j - 1); ub_cols.append(i * 9 + j)
    A_ub = csr_matrix((np.ones(len(ub_rows)), (ub_rows, ub_cols)), shape=(MAXB, nx))
    s = _obj_scale(mse)
    r = linprog(mse.reshape(-1) * s, A_ub=A_ub, b_ub=slots.astype(float),
                A_eq=A_eq, b_eq=np.ones(d), bounds=(0, None), method="highs-ds")
    assert r.status == 0, f"x-LP failed: {r.message}"
    return r.fun / s, r.x.reshape(-1, 9)


def lp_round(mse: np.ndarray, B: int):
    """The note's algorithm: LP -> ceil nonzero z -> re-solve x with z fixed."""
    lp_obj, _, z = solve_lp(mse, B)
    nz_mask = z > TOL
    n_nonzero_z = int(nz_mask.sum())
    z_int = np.where(nz_mask, np.ceil(z - TOL), 0.0).astype(np.int64)
    bytes_used = int(z_int.sum())
    slots = A @ z_int
    obj, x = solve_x_given_slots(mse, slots)
    frac = np.abs(x - np.round(x))
    n_frac_x = int((frac > 1e-7).sum())
    bits = np.argmax(np.round(x), axis=1)   # integral assignment (valid if n_frac_x == 0)
    return dict(lp_bound=lp_obj, obj=obj, bits=bits, bytes_used=bytes_used,
                n_nonzero_z=n_nonzero_z, n_frac_x=n_frac_x)


def solve_milp(mse: np.ndarray, B: int, time_limit: float = 300.0):
    s = _obj_scale(mse)
    cost, A_ub, b_ub, A_eq, b_eq, nx, nz = build_lp(mse * s, B)
    cons = [LinearConstraint(A_ub, ub=b_ub), LinearConstraint(A_eq, lb=b_eq, ub=b_eq)]
    integ = np.ones(nx + nz)
    r = milp(c=cost, constraints=cons, integrality=integ,
             bounds=Bounds(0, np.inf), options={"time_limit": time_limit})
    ok = r.status == 0
    return (r.fun / s if ok else np.nan), ok


# ---------------------------------------------------------------------------
# 2-stage baseline: greedy bit allocation (bit budget) -> production FFD pack;
# binary-search the largest bit budget whose packed size fits the byte budget.
# ---------------------------------------------------------------------------
def greedy_alloc(mse: np.ndarray, bit_budget: int) -> np.ndarray:
    d = mse.shape[0]
    b = np.zeros(d, dtype=np.int64)
    gain = mse[np.arange(d), 0] - mse[np.arange(d), 1]
    for _ in range(int(bit_budget)):
        i = int(np.argmax(gain))
        if gain[i] <= -np.inf or b[i] >= MAXB:
            break
        b[i] += 1
        gain[i] = (mse[i, b[i]] - mse[i, b[i] + 1]) if b[i] < MAXB else -np.inf
    return b

def packed_bytes(bits: np.ndarray) -> int:
    w = bits[bits > 0].astype(np.int64)
    return int(ffd_layout(w)[2]) if w.size else 0

def two_stage(mse: np.ndarray, B: int):
    lo, hi = 0, MAXB * mse.shape[0]
    best_bits, best_obj = np.zeros(mse.shape[0], dtype=np.int64), float(mse[:, 0].sum())
    while lo <= hi:
        mid = (lo + hi) // 2
        bits = greedy_alloc(mse, mid)
        if packed_bytes(bits) <= B:
            obj = float(mse[np.arange(len(bits)), bits].sum())
            if obj < best_obj:
                best_obj, best_bits = obj, bits
            lo = mid + 1
        else:
            hi = mid - 1
    return best_obj, best_bits, packed_bytes(best_bits)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_real_curves(col: str = "lloyd_mse") -> np.ndarray:
    df = pd.read_csv(REPO / "results" / "approx_codebook" / "approx_codebook.csv")
    d = int(df["dim"].max()) + 1
    mse = np.zeros((d, 9))
    var = df.groupby("dim")["var"].first()
    mse[:, 0] = var.values
    for _, r in df.iterrows():
        mse[int(r["dim"]), int(r["bits"])] = r[col]
    return mse

def synth_curves(d: int, nonconvex_frac: float = 0.5) -> np.ndarray:
    """Monotone-decreasing MSE curves; a fraction get a deliberate convexity
    violation (2nd bit drops more than the 1st, per the note's example)."""
    var = RNG.lognormal(-2, 1.5, d)
    mse = np.zeros((d, 9)); mse[:, 0] = var
    for i in range(d):
        v = var[i]
        drops = np.sort(RNG.uniform(0.3, 0.95, MAXB))[::-1]   # convex-ish by default
        if RNG.random() < nonconvex_frac:
            drops[0], drops[1] = drops[1] * 0.5, drops[0]     # bit 2 gains > bit 1
        for j in range(1, 9):
            v = v * drops[j - 1]
            mse[i, j] = v
    return mse


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def convexity_scan(mse: np.ndarray, name: str):
    deltas = mse[:, :-1] - mse[:, 1:]              # marginal drop of bit j (col j-1)
    viol = deltas[:, 1:] > deltas[:, :-1] + 1e-12  # drop increases -> non-convex
    n_dims = int((viol.any(axis=1)).sum())
    second_vs_first = deltas[:, 1] / np.maximum(deltas[:, 0], 1e-30)
    print(f"[E1:{name}] dims with any convexity violation: {n_dims}/{mse.shape[0]}"
          f" | max delta2/delta1 = {second_vs_first.max():.3f}"
          f" (note's example would be 2.0)")
    return n_dims


def run_instance(mse: np.ndarray, B: int, do_milp: bool, milp_tl: float):
    res = lp_round(mse, B)
    ts_obj, _, ts_bytes = two_stage(mse, B)
    milp_obj, milp_ok = solve_milp(mse, B, milp_tl) if do_milp else (np.nan, False)
    return dict(B=B, lp_bound=res["lp_bound"], lp_round_obj=res["obj"],
                lp_round_bytes=res["bytes_used"], bytes_over=res["bytes_used"] - B,
                n_nonzero_z=res["n_nonzero_z"], n_frac_x=res["n_frac_x"],
                two_stage_obj=ts_obj, two_stage_bytes=ts_bytes,
                milp_obj=milp_obj, milp_ok=milp_ok)


def main():
    print(f"partitions of 8: {C} (note says 21)")
    rows = []

    # ---- real curves ----
    mse = load_real_curves()
    d = mse.shape[0]
    convexity_scan(mse, "dbpedia-lloyd")
    print(f"[real] d={d}")
    for bpd in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        B = int(d * bpd / 8)
        r = run_instance(mse, B, do_milp=True, milp_tl=120.0)
        r.update(dataset="dbpedia", d=d, bpd=bpd)
        rows.append(r)
        print(f"[E4:real bpd={bpd}] B={B}  lp_bound={r['lp_bound']:.6g}  "
              f"lp+round={r['lp_round_obj']:.6g} ({r['lp_round_bytes']}B, +{r['bytes_over']})  "
              f"2stage={r['two_stage_obj']:.6g} ({r['two_stage_bytes']}B)  "
              f"milp={r['milp_obj']:.6g}{'' if r['milp_ok'] else ' (TIMEOUT)'}  "
              f"nnz_z={r['n_nonzero_z']}  frac_x={r['n_frac_x']}")

    # ---- synthetic claim trials (small d so the MILP is exact) ----
    n_claim2_viol = n_claim1_viol = 0
    trials = 200
    for t in range(trials):
        d_s = int(RNG.integers(16, 129))
        mse_s = synth_curves(d_s, nonconvex_frac=0.6)
        bpd = float(RNG.uniform(0.5, 7.5))
        B = max(1, int(d_s * bpd / 8))
        r = run_instance(mse_s, B, do_milp=(t < 40), milp_tl=30.0)
        r.update(dataset=f"synth{t}", d=d_s, bpd=bpd)
        rows.append(r)
        if r["n_nonzero_z"] > 8: n_claim2_viol += 1
        if r["n_frac_x"] > 0: n_claim1_viol += 1
    sub = pd.DataFrame(rows)
    synth = sub[sub.dataset.str.startswith("synth")]
    print(f"\n[E2] Claim 2 (nnz(z) <= 8 at LP vertex): violations {n_claim2_viol}/{trials}"
          f"  (max nnz seen: {int(synth.n_nonzero_z.max())},"
          f" real-data max: {int(sub[sub.dataset=='dbpedia'].n_nonzero_z.max())})")
    print(f"[E3] Claim 1 (x integral given integer z): violations {n_claim1_viol}/{trials}"
          f"  (real data: {int(sub[sub.dataset=='dbpedia'].n_frac_x.max())} fractional)")
    print(f"[E4] byte overhead of rounding, all instances: max +{int(sub.bytes_over.max())}"
          f" bytes (guarantee: <= 8)")

    out = HERE / "lp_joint_alloc.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
