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
    """Compute an FFD packing of per-dim widths into bytes.

    Args:
        bits_per_dim: (D,) int array, each in [0, byte_cap]. Zero-width dims are
            skipped (no storage).
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
    # FFD: descending width; tie-break by original index for determinism.
    order = sorted([d for d in range(D) if b[d] > 0], key=lambda d: (-b[d], d))
    bin_remaining: List[int] = []  # remaining capacity per open byte
    for d in order:
        placed = False
        for bi in range(len(bin_remaining)):
            if bin_remaining[bi] >= b[d]:
                bit_off[d] = byte_cap - bin_remaining[bi]
                byte_idx[d] = bi
                bin_remaining[bi] -= b[d]
                placed = True
                break
        if not placed:
            bit_off[d] = 0
            byte_idx[d] = len(bin_remaining)
            bin_remaining.append(byte_cap - b[d])
    n_bytes = len(bin_remaining)
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
