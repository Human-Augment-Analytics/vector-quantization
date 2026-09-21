"""Joint bit-allocation + byte-packing LP allocator (research-group note, 2026-09-10).

Solves, for per-dim cost curves cost[d][b] (b = 0..8 bits) and a byte budget B:

    min  sum_{d,b} x[d,b] * cost[d,b]
    s.t. sum_b x[d,b] == 1                      (each dim gets one bit level)
         sum_d x[d,b] <= sum_c A[b,c] z[c]      (slots for b-bit groups, b>=1)
         sum_c z[c] <= B                        (byte budget; z[c] = #bytes with
                                                 bit-partition c, c over the 22
                                                 partitions of 8)

then rounds: ceil the nonzero z (adds <= #nonzero <= 8 bytes), re-solves x with
slots fixed — the x-polytope is a transportation polytope, so a vertex optimum is
integral (verified experimentally in results/lp_joint_alloc/, 0 violations).

Objective is normalized to O(1) internally: raw MSE entries ~1e-7 make HiGHS
terminate early (seen 1.7% off at 6 bpd).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

MAXB = 8


def _partitions_of(n: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    def rec(rem, mx, cur):
        if rem == 0:
            out.append(tuple(cur)); return
        for p in range(min(rem, mx), 0, -1):
            cur.append(p); rec(rem - p, p, cur); cur.pop()
    rec(n, n, [])
    return out

PARTS = _partitions_of(MAXB)
_C = len(PARTS)
_A = np.zeros((MAXB, _C), dtype=np.int64)
for _c, _part in enumerate(PARTS):
    for _p in _part:
        _A[_p - 1, _c] += 1


def lp_joint_alloc(cost: np.ndarray, byte_budget: int) -> tuple[np.ndarray, int]:
    """Return (bits, n_nonzero_z). bits[d] in 0..8; slot capacities certify the
    allocation packs into <= ceil'd-z bytes (usually byte_budget..byte_budget+2)."""
    d, levels = cost.shape
    assert levels == MAXB + 1
    s = 1.0 / max(float(np.abs(cost).mean()), 1e-300)
    nx, nz = d * 9, _C

    c_vec = np.concatenate([(cost * s).reshape(-1), np.zeros(nz)])
    rows = np.repeat(np.arange(d), 9)
    A_eq = csr_matrix((np.ones(nx), (rows, np.arange(nx))), shape=(d, nx + nz))

    ub_r, ub_c, ub_v = [], [], []
    for j in range(1, MAXB + 1):
        for i in range(d):
            ub_r.append(j - 1); ub_c.append(i * 9 + j); ub_v.append(1.0)
        for c in range(_C):
            if _A[j - 1, c]:
                ub_r.append(j - 1); ub_c.append(nx + c); ub_v.append(-float(_A[j - 1, c]))
    for c in range(_C):
        ub_r.append(MAXB); ub_c.append(nx + c); ub_v.append(1.0)
    A_ub = csr_matrix((ub_v, (ub_r, ub_c)), shape=(MAXB + 1, nx + nz))
    b_ub = np.concatenate([np.zeros(MAXB), [float(byte_budget)]])

    r = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(d),
                bounds=(0, None), method="highs-ds")
    if r.status != 0:
        raise RuntimeError(f"joint LP failed: {r.message}")
    z = r.x[nx:]
    nz_mask = z > 1e-6
    z_int = np.where(nz_mask, np.ceil(z - 1e-6), 0.0).astype(np.int64)
    slots = (_A @ z_int).astype(float)

    # x-only re-solve with integer slots (integral at a vertex).
    ub_r2, ub_c2 = [], []
    for j in range(1, MAXB + 1):
        for i in range(d):
            ub_r2.append(j - 1); ub_c2.append(i * 9 + j)
    A_ub2 = csr_matrix((np.ones(len(ub_r2)), (ub_r2, ub_c2)), shape=(MAXB, nx))
    A_eq2 = csr_matrix((np.ones(nx), (rows, np.arange(nx))), shape=(d, nx))
    r2 = linprog((cost * s).reshape(-1), A_ub=A_ub2, b_ub=slots,
                 A_eq=A_eq2, b_eq=np.ones(d), bounds=(0, None), method="highs-ds")
    if r2.status != 0:
        raise RuntimeError(f"x-LP failed: {r2.message}")
    x = r2.x.reshape(d, 9)
    frac = np.abs(x - np.round(x)).max()
    if frac > 1e-6:
        raise RuntimeError(f"x not integral at vertex (max frac {frac}) — Claim 1 violated?")
    bits = np.argmax(np.round(x), axis=1).astype(np.int64)
    return bits, int(nz_mask.sum())
