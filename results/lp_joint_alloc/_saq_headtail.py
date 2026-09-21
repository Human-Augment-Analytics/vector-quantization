"""Head/tail bit split: SAQ's joint-DP plan vs our perdim-MSE (alpha=0) greedy,
both on the MSMARCO variance spectrum at 2 bpd (D=1024, head = first 512 PCA dims).

SAQ side: saq.allocate_dp with the engine's exact config (total_bits = avg*D + 64,
max 13 bits/dim, 64-dim padding) — the same BitAllocatorDP the benchmark ran.
Our side: the RankAwareQuantizer allocation replicated exactly (Dg table seeds 0..8,
greedy, total = round(avg*D), max 8). Spectrum: PCA eigenvalues of the first 200K
rows of msmarco_500k (same corpus prefix as the PACE msmarco_200k set). WSL run
(saq wheel is Linux-only); memmap + chunked covariance keeps memory small.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import saq
from haag_vq.methods.rank_aware_quantization import _lloyd_1d_normal

BASE = REPO.parent / "SAQ" / "data" / "datasets" / "msmarco_500k" / "base.fvecs"
import os
N, D, AVG_BITS = 200_000, 1024, float(os.environ.get("AVG_BITS", "2.0"))

# ---- variance spectrum (chunked, memmap; rows are [dim_i32][1024 f32]) ----
mm = np.memmap(BASE, dtype=np.float32, mode="r").reshape(-1, D + 1)
S = np.zeros(D); G = np.zeros((D, D))
for s in range(0, N, 20_000):
    blk = np.asarray(mm[s:s + 20_000, 1:], dtype=np.float64)
    S += blk.sum(axis=0); G += blk.T @ blk
mu = S / N
C = G / N - np.outer(mu, mu)
var = np.sort(np.clip(np.linalg.eigvalsh(C), 1e-12, None))[::-1]
print(f"spectrum: D={D} top eig={var[0]:.4e} median={np.median(var):.2e}")

# ---- SAQ joint DP plan (engine config) ----
cfg = saq.JointAllocationConfig()
cfg.num_dim_padded = D
cfg.dim_padding_size = 64
cfg.max_bits_per_dim = 13
cfg.num_bit_factors = 2 * 4 * 8            # kNumShortFactors * sizeof(float) * 8
cfg.total_bits = int(AVG_BITS * D) + cfg.num_bit_factors
r = saq.allocate_dp(np.asarray(var, dtype=np.float32), cfg)
assert r.ok(), r.error
plan = list(r.quant_plan)
bits_saq = np.concatenate([np.full(int(dl), int(b), dtype=np.int64) for dl, b in plan])
print(f"SAQ plan: {plan}  (total code bits {int(bits_saq.sum())}, "
      f"total_bits_used incl. factor overhead {r.total_bits_used})")

# ---- our perdim-MSE (alpha=0) greedy ----
Dg = np.array([_lloyd_1d_normal(2 ** b, seed=b)[1] for b in range(9)]); Dg[0] = 1.0
bits_ours = np.zeros(D, dtype=np.int64)
gain = var * (Dg[0] - Dg[1])
for _ in range(int(round(AVG_BITS * D))):
    i = int(np.argmax(gain))
    bits_ours[i] += 1
    gain[i] = var[i] * (Dg[bits_ours[i]] - Dg[bits_ours[i] + 1]) if bits_ours[i] < 8 else -np.inf

half = D // 2
for name, b in [("perdim-MSE(a=0)", bits_ours), ("SAQ-DP        ", bits_saq)]:
    print(f"{name}: head={int(b[:half].sum())}  tail={int(b[half:].sum())}  "
          f"total={int(b.sum())}  widths={ {int(j): int((b == j).sum()) for j in np.unique(b)} }")
