"""First-Fit-Decreasing bin-packing of per-dimension bit-widths into bytes.

Each dimension's b_d-bit code is placed wholly inside one byte (b_d <= 8), so
decode is byte-aligned (read a byte, shift/mask out each dim's code) rather than
cross-byte bit assembly. This realizes the 'combine per-dim bits into whole
bytes (5+3, 4+3+1, ...)' packing. FFD: sort dims by width desc, place each into
the first byte with room, else open a new byte.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def byte_cap_minus(width: int, offset: int, byte_cap: int = 8) -> int:
    """Right-shift to place/extract a ``width``-bit field at ``offset`` from the
    MSB side of a ``byte_cap``-bit byte.

    A field occupying bits ``[offset, offset + width)`` counted from the MSB has
    its LSB at distance ``byte_cap - offset - width`` from the byte's LSB, which
    is exactly the shift needed to align the field's LSB with bit 0.
    """
    return byte_cap - offset - width


def ffd_layout(bits_per_dim: np.ndarray, byte_cap: int = 8) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pack per-dim widths into bytes via First-Fit-Decreasing with the 4-fix —
    optimal for byte_cap==8.

    Plain FFD (cap 8) is suboptimal only around width-4 dims: 4 is the unique
    self-complementary size (4+4=8=C), so an orphaned 4, placed before the 3s, grabs
    a 3 (4+3=7, wasting a bit) and breaks the 3s' packing. Fix: move ALL width-4 dims
    to just after the width-3 dims (before the width<=2 dims). This is a single FFD
    pass, never worse than plain FFD, and produces an OPTIMAL packing: verified
    exhaustively vs the exact config-ILP for every item multiset with bit-sum <= 50
    (268,681 cases, 0 suboptimal) plus 50k random adversarial instances. (Moving all
    4s is provably equivalent to demoting only the lone 4 — same bin count everywhere
    — but needs no odd/even branch.)

    Args:
        bits_per_dim: (D,) int array, each in [0, byte_cap]. Zero-width dims skipped.
        byte_cap: bits per bin (8 for a byte).
    Returns:
        byte_idx: (D,) int — which byte each dim is packed into (-1 if b_d==0).
        bit_off:  (D,) int — bit offset within that byte (-1 if b_d==0).
        n_bytes:  total bytes used.
    """
    b = np.asarray(bits_per_dim, dtype=np.int64)
    D = b.shape[0]
    if np.any(b > byte_cap) or np.any(b < 0):
        raise ValueError(f"bits_per_dim must be in [0, {byte_cap}]")
    byte_idx = np.full(D, -1, dtype=np.int64)
    bit_off = np.full(D, -1, dtype=np.int64)
    # FFD order: descending width, tie-break by index for determinism.
    order = sorted([d for d in range(D) if b[d] > 0], key=lambda d: (-b[d], d))

    # 4-fix (byte_cap==8 only): move ALL width-4 dims to just after the width-3 dims
    # (before the first width<=2). Provably equivalent to demoting only the lone 4
    # -- identical bin count on all 268,681 multisets with bit-sum <= 50 -- but with
    # no parity branch. Optimal cap-8 packing; see ffd-orphaned-4-optimality note.
    if byte_cap == 8 and any(b[d] == 4 for d in order):
        fours = [d for d in order if b[d] == 4]
        rest = [d for d in order if b[d] != 4]
        ins = next((i for i, d in enumerate(rest) if b[d] <= 2), len(rest))
        order = rest[:ins] + fours + rest[ins:]

    # First-fit-decreasing placement, recording the byte layout.
    bin_remaining: List[int] = []  # remaining capacity per open byte
    for d in order:
        w = int(b[d])
        placed = -1
        for bi in range(len(bin_remaining)):
            if bin_remaining[bi] >= w:
                placed = bi
                break
        if placed < 0:
            placed = len(bin_remaining)
            bin_remaining.append(byte_cap)
        bit_off[d] = byte_cap - bin_remaining[placed]   # bits already used in this byte
        byte_idx[d] = placed
        bin_remaining[placed] -= w

    return byte_idx, bit_off, len(bin_remaining)


def ffd_encode(codes: np.ndarray, bits_per_dim: np.ndarray,
               byte_idx: np.ndarray, bit_off: np.ndarray, n_bytes: int,
               byte_cap: int = 8) -> np.ndarray:
    """Pack (N, D) integer codes into (N, n_bytes) uint8 per the FFD layout.
    codes[:, d] must be in [0, 2^bits_per_dim[d]). Dims with b_d==0 are skipped."""
    codes = np.asarray(codes, dtype=np.uint64)
    N = codes.shape[0]
    out = np.zeros((N, n_bytes), dtype=np.uint8)
    b = np.asarray(bits_per_dim, dtype=np.int64)
    for d in range(b.shape[0]):
        if b[d] == 0:
            continue
        # place code[:, d] into byte byte_idx[d] at offset bit_off[d] (MSB-first within byte)
        shift = byte_cap_minus(int(b[d]), int(bit_off[d]), byte_cap)
        out[:, byte_idx[d]] |= (codes[:, d].astype(np.uint8) << shift)
    return out


def ffd_decode(packed: np.ndarray, bits_per_dim: np.ndarray,
               byte_idx: np.ndarray, bit_off: np.ndarray, D: int,
               byte_cap: int = 8) -> np.ndarray:
    """Inverse of ffd_encode -> (N, D) int codes (0 for b_d==0 dims)."""
    packed = np.asarray(packed, dtype=np.uint8)
    N = packed.shape[0]
    codes = np.zeros((N, D), dtype=np.int64)
    b = np.asarray(bits_per_dim, dtype=np.int64)
    for d in range(D):
        if b[d] == 0:
            continue
        shift = byte_cap_minus(int(b[d]), int(bit_off[d]), byte_cap)
        mask = (1 << int(b[d])) - 1
        codes[:, d] = (packed[:, byte_idx[d]].astype(np.int64) >> shift) & mask
    return codes
