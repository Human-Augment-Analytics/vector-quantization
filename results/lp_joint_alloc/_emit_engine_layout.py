"""Emit a per-dimension byte LAYOUT for the engine's heterogeneous-segment
fused decode (stage 2), from the harness LP solution directly.

Unlike _emit_engine_plan.py (block-64 DP -> uniform-width segments, an
approximation the engine's segment format required), this runs the paper's
actual allocator -- joint allocation+packing LP with query-covariance weights
and empirical cost tables, plus the tighten-retry and greedy top-up of
RankAwareQuantizer -- and materializes the byte packing with ffd_layout
(verified exactly optimal on realized allocations). No permutation, no
segment merging: dims stay in PCA order and the layout maps each dim to
(byte index, bit offset from MSB).

Output <out-dir>/layout.txt:
    line 1:  D n_bytes
    lines 2..D+1:  width byte_idx bit_off      (-1 -1 for 0-width dims)

Also prints the model-cost comparison vs the block-64 plan (fidelity gain)
and the effective byte-pattern histogram (which decode branches stage 2
needs).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from haag_vq.methods.lp_alloc import lp_joint_alloc
from haag_vq.methods.ffd_packing import ffd_layout

import importlib.util
_spec = importlib.util.spec_from_file_location("planmod", HERE / "_emit_engine_plan.py")
planmod = importlib.util.module_from_spec(_spec)
_src = open(_spec.origin).read().replace('if __name__ == "__main__":\n    main()', '')
exec(compile(_src, _spec.origin, "exec"), planmod.__dict__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bpd", required=True, type=float)
    ap.add_argument("--max-bits", default=8, type=int)
    ap.add_argument("--sample-n", default=200_000, type=int)
    args = ap.parse_args()
    src, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    var_raw = np.fromfile(src / "variances_pca.fvecs", dtype=np.float32)
    D = int(var_raw[:1].view(np.int32)[0])
    var = var_raw[1 : 1 + D].astype(np.float64)

    def _read_sample(path: Path, max_rows: int) -> np.ndarray:
        d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
        assert d == D
        n_rows = path.stat().st_size // (4 * (d + 1))
        take = min(max_rows, n_rows)
        raw = np.fromfile(path, dtype=np.float32, count=take * (d + 1))
        return raw.reshape(take, d + 1)[:, 1:].astype(np.float64)

    # Query-covariance weights and empirical cost table, as in the harness
    # (lp_q_emp) and _emit_engine_plan --weights query --cost-source empirical.
    Qs = _read_sample(src / "queries_pca.fvecs", 50_000)
    w = (Qs ** 2).mean(axis=0)
    print(f"weights=query from {Qs.shape[0]} queries")
    Xs = _read_sample(src / "vectors_pca.fvecs", args.sample_n)
    print(f"empirical costs from {Xs.shape[0]} rows")
    f = np.empty((D, args.max_bits + 1))
    for j in range(D):
        col = np.sort(Xs[:, j])
        cs = np.concatenate(([0.0], np.cumsum(col)))
        cs2 = np.concatenate(([0.0], np.cumsum(col * col)))
        n_s = len(col)
        f[j, 0] = cs2[-1] / n_s - (cs[-1] / n_s) ** 2
        for l in range(1, args.max_bits + 1):
            k = 2 ** l
            qq = (np.arange(k) + 0.5) / k
            c = col[np.minimum((qq * n_s).astype(int), n_s - 1)].copy()
            for _ in range(100_000):
                bnd = 0.5 * (c[:-1] + c[1:])
                ii = np.searchsorted(col, bnd)
                lo = np.concatenate(([0], ii)); hi = np.concatenate((ii, [n_s]))
                cnt = hi - lo
                new = np.sort(np.where(cnt > 0,
                                       (cs[hi] - cs[lo]) / np.maximum(cnt, 1), c))
                if float(np.max(np.abs(new - c))) < 1e-10:
                    c = new; break
                c = new
            bnd = 0.5 * (c[:-1] + c[1:])
            ii = np.searchsorted(col, bnd)
            lo = np.concatenate(([0], ii)); hi = np.concatenate((ii, [n_s]))
            cnt = hi - lo
            f[j, l] = float(np.sum((cs2[hi] - cs2[lo])
                                   - 2 * c * (cs[hi] - cs[lo]) + c * c * cnt)) / n_s

    cost = w[:, None] * f
    byte_budget = int(round(args.bpd * D / 8))

    # Allocation exactly as RankAwareQuantizer (allocator="lp"): LP, tighten
    # on rounding overshoot, then greedy top-up of stranded capacity.
    B = byte_budget
    while B > 0:
        bits, _ = lp_joint_alloc(cost, B)
        if int(ffd_layout(bits)[2]) <= byte_budget:
            break
        B -= 1
    while True:
        nb_ = np.minimum(bits + 1, args.max_bits)
        g = cost[np.arange(D), bits] - cost[np.arange(D), nb_]
        order = np.argsort(-np.where(bits < args.max_bits, g, -np.inf))
        added = False
        for d_star in order:
            if bits[d_star] >= args.max_bits:
                break
            bits[d_star] += 1
            if int(ffd_layout(bits)[2]) <= byte_budget:
                added = True
                break
            bits[d_star] -= 1
        if not added:
            break

    byte_idx, bit_off, n_bytes = ffd_layout(bits)
    model_cost = float(cost[np.arange(D), bits].sum())

    # Fidelity comparison vs the block-64 plan at the same inputs.
    BLOCK = planmod.BLOCK
    block_cost = cost.reshape(D // BLOCK, BLOCK, args.max_bits + 1).sum(axis=1)
    bbits, bcost, _ = planmod.solve_block_allocation(block_cost, args.bpd, args.max_bits)
    print(f"bits/vec {int(bits.sum())} in {n_bytes} bytes (budget {byte_budget}; "
          f"{bits.sum() / max(n_bytes, 1):.2f} bits/byte)")
    print(f"model cost per-dim LP {model_cost:.6g} vs block-64 plan {bcost:.6g} "
          f"({100 * (bcost - model_cost) / model_cost:+.2f}% block penalty)")

    # Effective byte patterns (sorted width tuples) -> the decode branches needed.
    pats = Counter()
    for t in range(n_bytes):
        widths = tuple(sorted((int(b) for b, bi in zip(bits, byte_idx) if bi == t),
                              reverse=True))
        pats[widths] += 1
    print(f"effective patterns ({len(pats)}): "
          + "  ".join(f"{'+'.join(map(str, p))}x{c}" for p, c in pats.most_common()))

    with open(out / "layout.txt", "w") as fh:
        fh.write(f"{D} {n_bytes}\n")
        for j in range(D):
            fh.write(f"{int(bits[j])} {int(byte_idx[j])} {int(bit_off[j])}\n")
    print(f"wrote {out/'layout.txt'}")


if __name__ == "__main__":
    main()
