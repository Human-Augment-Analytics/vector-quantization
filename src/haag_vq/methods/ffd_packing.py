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


def _bfd_pack(order: List[int], b: np.ndarray, byte_cap: int,
              byte_idx: np.ndarray, bit_off: np.ndarray,
              first_byte: int = 0, prefill: List[int] = None) -> int:
    """Bucketed best-fit-decreasing packing of `order` (dims, descending width)
    into bytes, numbered from `first_byte`. O(n): capacities are bounded [0,byte_cap]
    so the best-fit search per item is <= byte_cap steps (O(1)).

    Writes byte_idx/bit_off in place; returns the number of bytes opened here
    (NOT counting `first_byte` pre-reserved bytes). `prefill[bi]` optionally seeds
    the remaining capacity of pre-reserved byte (first_byte-1-...)? -- see caller.
    `buckets[r]` holds byte indices whose remaining capacity is exactly r.
    """
    buckets: List[List[int]] = [[] for _ in range(byte_cap + 1)]
    # seed buckets with any pre-reserved bytes' leftover capacity
    if prefill:
        for bi, rem in prefill:
            if rem > 0:
                buckets[rem].append(bi)
    n_new = 0
    for d in order:
        w = int(b[d])
        r = next((rr for rr in range(w, byte_cap + 1) if buckets[rr]), -1)
        if r < 0:                                   # open a new byte
            bi = first_byte + n_new
            n_new += 1
            bit_off[d] = 0
            rem = byte_cap - w
        else:
            bi = buckets[r].pop()
            bit_off[d] = byte_cap - r               # = bits already used in this byte
            rem = r - w
        byte_idx[d] = bi
        buckets[rem].append(bi)
    return n_new


def ffd_layout(bits_per_dim: np.ndarray, byte_cap: int = 8) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pack per-dim widths into bytes — O(n) and optimal for byte_cap==8.

    Bucketed best-fit-decreasing (linear, since capacities are bounded), wrapped in
    the ``min(plain, 4+2+2-fix)`` rule: a lone (odd-count) width-4 dim that would
    waste a near-empty byte is co-located with two width-2 dims as {4,2,2}; we keep
    that variant only if it uses fewer bytes. Empirically optimal vs the exact
    config-ILP on all sampled cap-8 instances (see results/theory_counterexamples.py).

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
    nz = [d for d in range(D) if b[d] > 0]
    order = sorted(nz, key=lambda d: (-b[d], d))   # descending width, deterministic

    def plain():
        bi = np.full(D, -1, dtype=np.int64)
        bo = np.full(D, -1, dtype=np.int64)
        nb = _bfd_pack(order, b, byte_cap, bi, bo)
        return bi, bo, nb

    byte_idx, bit_off, n_bytes = plain()

    # 4+2+2 fix (byte_cap==8 only): when there is an odd number of width-4 dims and
    # >=2 width-2 dims, reserve byte 0 as {4,2,2} and best-fit-pack the rest after.
    if byte_cap == 8:
        fours = [d for d in order if b[d] == 4]
        twos = [d for d in order if b[d] == 2]
        if len(fours) % 2 == 1 and len(twos) >= 2:
            bi = np.full(D, -1, dtype=np.int64)
            bo = np.full(D, -1, dtype=np.int64)
            f, t0, t1 = fours[-1], twos[-2], twos[-1]   # the lone 4 + two 2s -> byte 0
            bi[f], bo[f] = 0, 0
            bi[t0], bo[t0] = 0, 4
            bi[t1], bo[t1] = 0, 6
            reserved = {f, t0, t1}
            rest = [d for d in order if d not in reserved]
            n_new = _bfd_pack(rest, b, byte_cap, bi, bo, first_byte=1)
            if 1 + n_new < n_bytes:
                byte_idx, bit_off, n_bytes = bi, bo, 1 + n_new

    return byte_idx, bit_off, n_bytes


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
