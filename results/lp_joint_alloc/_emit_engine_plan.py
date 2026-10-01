"""Emit an engine-injectable quantization plan (block-64 rank-aware allocation)
plus a width-sorted, dimension-permuted copy of a SAQ engine dataset.

The SAQ engine expresses plans as contiguous (dim_len, bits) segments with
dim_len a multiple of 64 (kDimPaddingSize) — so 64*b bits per block is always
whole bytes and the packing dimension of the joint LP vanishes: the
engine-expressible optimum is a pure allocation over 64-dim blocks. This
script solves that exactly (small DP knapsack = the block-LP's integral
optimum), using the same rank-aware cost model as the harness allocators:

    cost(block i at b bits) = sum_{d in block i} var_d^(1+alpha) * Dg(b)

with Dg the normalized Lloyd-Max MSE of N(0,1) (rank_aware_quantization).
Budget mirrors the engine DP's convention: round(bpd*D) + 64 total bits, and
every nonzero-bit segment pays 64 factor bits (0-bit segments pay none).
Segment count = #distinct widths after the width-sort, handled by an outer
loop over the segment-overhead charge.

Caveat (modelling, judged by the A/B): the engine rotates each segment, mixing
per-dim variances within it; the per-dim cost model prices the plan in
unrotated coordinates, same as the engine's own DP prices var/2^b.

Usage (Windows or WSL python; IO is chunked so WSL is safe):
    python results/lp_joint_alloc/_emit_engine_plan.py \
        --data-dir ../SAQ/data/datasets/dbpedia_100k \
        --out-dir  ../SAQ/data/datasets/dbpedia_100k_lp2a1 \
        --bpd 2.0 [--alpha 1.0] [--max-bits 8]

Outputs in --out-dir:
    plan.txt        one "dim_len bits" pair per line (feed via SAQ_QUANT_PLAN
                    or IVF/GpuIVF.set_quant_plan)
    perm_dims.txt   permutation: line j = original dim index at new position j
    *_pca.fvecs     permuted copies (vectors, queries, centroids, variances)
    cluster_ids_*.ivecs, groundtruth.ivecs, metadata.txt   copied verbatim
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from haag_vq.methods.rank_aware_quantization import _lloyd_1d_normal

BLOCK = 64          # engine kDimPaddingSize
FACTOR_BITS = 64    # engine kNumShortFactors * 32 bits per (nonzero) segment


def read_fvecs_header(path: Path) -> tuple[int, int]:
    """Return (dim, num_rows) of an fvecs file."""
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    row_bytes = 4 * (d + 1)
    size = path.stat().st_size
    assert size % row_bytes == 0, f"{path}: size {size} not a multiple of row {row_bytes}"
    return d, size // row_bytes


def permute_fvecs(src: Path, dst: Path, perm: np.ndarray, chunk_rows: int = 8192) -> int:
    """Stream-permute the columns of an fvecs file: out[:, j] = in[:, perm[j]]."""
    d, n = read_fvecs_header(src)
    assert d == len(perm), f"{src}: dim {d} != permutation length {len(perm)}"
    done = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while done < n:
            take = min(chunk_rows, n - done)
            raw = np.fromfile(fi, dtype=np.float32, count=take * (d + 1))
            rows = raw.reshape(take, d + 1)
            hdr = rows[:, :1].view(np.int32)
            assert np.all(hdr == d), f"{src}: corrupt fvecs header in rows {done}..{done+take}"
            out = np.empty_like(rows)
            out[:, :1] = rows[:, :1]
            out[:, 1:] = rows[:, 1:][:, perm]
            out.tofile(fo)
            done += take
    return n


def solve_block_allocation(block_cost: np.ndarray, bpd: float, max_bits: int):
    """Exact optimum over the engine-expressible plan space: per-block bits with
    a 64-bit factor charge per distinct nonzero width (= per segment after the
    width-sort; 0-bit segments are free, matching the engine DP's accounting).

    Exhaustive over allowed-width subsets W of {1..max_bits} (0 bits always
    allowed, DP restricted to W u {0}, budget charged 64*|W|). A subset's
    solution may use fewer widths than |W| — then it is merely re-found more
    cheaply at its exact subset, so the scan remains exact. ~2^max_bits DPs of
    (nb blocks x budget/64 states): milliseconds at nb=24.
    """
    nb = block_cost.shape[0]
    budget = int(round(bpd * nb * BLOCK)) + FACTOR_BITS

    best = None  # (total_cost, bits_per_block)
    all_widths = list(range(1, max_bits + 1))
    for mask in range(1, 1 << max_bits):
        W = [w for w in all_widths if mask & (1 << (w - 1))]
        cap = (budget - FACTOR_BITS * len(W)) // BLOCK  # max sum of per-block bits
        if cap < 0:
            continue
        cap = min(cap, nb * max_bits)
        INF = np.inf
        f = np.full(cap + 1, INF)
        f[0] = 0.0
        choice = np.zeros((nb, cap + 1), dtype=np.int8)
        for i in range(nb):
            g = np.full(cap + 1, INF)
            for b in [0] + W:
                if b > cap:
                    continue
                cand = np.full(cap + 1, INF)
                cand[b:] = f[: cap + 1 - b] + block_cost[i, b]
                upd = cand < g
                g[upd] = cand[upd]
                choice[i, upd] = b
            f = g
        used = int(np.argmin(f))
        if not np.isfinite(f[used]):
            continue
        cost = float(f[used])
        if best is not None and cost >= best[0]:
            continue
        bits = np.zeros(nb, dtype=np.int64)
        u = used
        for i in range(nb - 1, -1, -1):
            b = int(choice[i, u])
            bits[i] = b
            u -= b
        distinct_nz = len(set(int(b) for b in bits if b > 0))
        total_bits = BLOCK * int(bits.sum()) + FACTOR_BITS * distinct_nz
        assert total_bits <= budget
        best = (cost, bits)
    assert best is not None, "no feasible allocation (budget too small?)"
    cost, bits = best
    return bits, cost, budget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bpd", required=True, type=float)
    ap.add_argument("--alpha", default=1.0, type=float)
    ap.add_argument("--max-bits", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int, help="Dg table seed (harness default 0)")
    ap.add_argument("--plan-only", action="store_true",
                    help="Write plan.txt + perm_dims.txt only; skip the permuted "
                         "dataset copies. Refuses a non-identity permutation (the "
                         "plan would be wrong against unpermuted data).")
    args = ap.parse_args()

    src, out = args.data_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Per-dim variance (1 x D fvecs)
    var_raw = np.fromfile(src / "variances_pca.fvecs", dtype=np.float32)
    D = int(var_raw[:1].view(np.int32)[0])
    var = var_raw[1 : 1 + D].astype(np.float64)
    assert D % BLOCK == 0, f"D={D} not a multiple of {BLOCK}"
    nb = D // BLOCK

    # Dg table — identical construction to RankAwareQuantizer (seed + b per level).
    Dg = np.empty(args.max_bits + 1)
    for b in range(args.max_bits + 1):
        _, Dg[b] = _lloyd_1d_normal(2 ** b, seed=args.seed + b)
    Dg[0] = 1.0

    var_pow = var ** (1.0 + args.alpha)
    block_w = var_pow.reshape(nb, BLOCK).sum(axis=1)          # (nb,)
    block_cost = block_w[:, None] * Dg[None, :]               # (nb, max_bits+1)

    bits, cost, budget = solve_block_allocation(block_cost, args.bpd, args.max_bits)

    # Width-sort permutation (stable: preserves PCA order within equal widths).
    order = np.argsort(-bits, kind="stable")                  # block order, widest first
    perm = (order[:, None] * BLOCK + np.arange(BLOCK)[None, :]).reshape(-1)

    # Contiguous runs of equal width -> plan segments.
    sorted_bits = bits[order]
    plan: list[tuple[int, int]] = []
    for b in sorted_bits:
        if plan and plan[-1][1] == b:
            plan[-1] = (plan[-1][0] + BLOCK, int(b))
        else:
            plan.append((BLOCK, int(b)))

    with open(out / "plan.txt", "w") as f:
        for dim_len, b in plan:
            f.write(f"{dim_len} {b}\n")
    np.savetxt(out / "perm_dims.txt", perm, fmt="%d")

    identity = bool((perm == np.arange(len(perm))).all())
    if args.plan_only:
        assert identity, ("--plan-only but the width-sort permutation is NOT the "
                          "identity; rerun without --plan-only to emit permuted data")
        print("plan-only: permutation is identity, no dataset copies needed")
    else:
        # Permute the coordinate files; copy the coordinate-free ones.
        for name in ["vectors_pca.fvecs", "queries_pca.fvecs", "variances_pca.fvecs"]:
            n = permute_fvecs(src / name, out / name, perm)
            print(f"permuted {name}: {n} rows")
        for p in sorted(src.glob("centroids_*_pca.fvecs")):
            n = permute_fvecs(p, out / p.name, perm)
            print(f"permuted {p.name}: {n} rows")
        for p in sorted(src.glob("cluster_ids_*.ivecs")) + [src / "groundtruth.ivecs"]:
            shutil.copyfile(p, out / p.name)
            print(f"copied   {p.name}")
        if (src / "metadata.txt").exists():
            shutil.copyfile(src / "metadata.txt", out / "metadata.txt")

    nz_segments = sum(1 for _, b in plan if b > 0)
    total_bits = sum(dl * b for dl, b in plan) + FACTOR_BITS * nz_segments
    plan_str = " ".join(f"{dl}d/{b}b" for dl, b in plan)
    print(f"\nplan ({len(plan)} segments, {nz_segments} nonzero): {plan_str}")
    print(f"bits/vec {total_bits} vs budget {budget} (bpd {total_bits / D:.3f} nominal {args.bpd})")
    print(f"model cost {cost:.6g}  (alpha={args.alpha}, max_bits={args.max_bits})")
    print(f"wrote {out / 'plan.txt'}")


if __name__ == "__main__":
    main()
