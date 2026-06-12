"""
Locally-adaptive Vector Quantization (LVQ) — single-level, SVS-style.

Each vector gets its own per-vector scalar quantizer:
  1. Subtract a global mean learned during fit().
  2. For each residual vector, record its lo/hi range and compute a uniform step.
  3. Quantize each dimension to a B-bit integer index.
  4. Pack everything into a self-contained byte row:
       [ bit-packed B-bit indices (ceil(D*B/8) bytes) | lo float32 (4 bytes) | delta float32 (4 bytes) ]

Because lo and delta are stored inside each code row, decompress() is fully
row-independent — you can slice codes[ids] and decompress without touching the
original codes array or any instance state beyond self.mu.
"""

import math

import numpy as np

from .base_quantizer import BaseQuantizer


class LVQQuantizer(BaseQuantizer):
    """
    Single-level Locally-adaptive Vector Quantization.

    Args:
        num_bits: Bits per dimension (1–8). Controls quantization precision.
                  code_size = ceil(D * num_bits / 8) + 8  bytes per vector.
    """

    def __init__(self, num_bits: int = 8) -> None:
        if not (1 <= num_bits <= 8):
            raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")
        self.num_bits: int = num_bits
        self.mu: np.ndarray | None = None  # global mean, shape (D,)

    # ------------------------------------------------------------------
    # BaseQuantizer interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> None:
        """Compute global mean from training data X of shape (N, D)."""
        self.mu = X.mean(axis=0).astype(np.float32)

    def compress(self, X: np.ndarray) -> np.ndarray:
        """
        Encode each row of X into a self-contained byte code.

        Returns
        -------
        codes : uint8 array of shape (N, code_size)
            code_size = ceil(D * B / 8) + 8
        """
        if self.mu is None:
            raise RuntimeError("LVQQuantizer.fit() must be called before compress/decompress.")
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        B = self.num_bits
        levels = (1 << B) - 1          # 2^B - 1
        bits_per_row = D * B
        index_bytes = math.ceil(bits_per_row / 8)
        code_size = index_bytes + 8     # +4 for lo float32, +4 for delta float32

        codes = np.empty((N, code_size), dtype=np.uint8)

        # Residuals: subtract global mean
        R = X - self.mu                 # (N, D)

        lo = R.min(axis=1)              # (N,)
        hi = R.max(axis=1)              # (N,)
        span = hi - lo                  # (N,)

        # Guard against constant residual (all-zero vector or identical dims)
        eps = np.finfo(np.float32).tiny
        delta = np.where(span == 0.0, eps, span / levels)  # (N,)

        # Quantize: integer indices in [0, 2^B - 1]
        # shape (N, D) → float, then round and clip
        idx = np.round((R - lo[:, None]) / delta[:, None]).astype(np.int32)
        idx = np.clip(idx, 0, levels)  # (N, D)

        # Bit-pack each row's D indices (each B bits wide), MSB-first.
        # np.packbits operates on a flat bit stream. We construct a (N, D*B) bool
        # array by expanding each index into B bits, then packbits row-wise.
        #
        # Bit layout (MSB-first within each index):
        #   index for dim 0: bits [B-1 .. 0], index for dim 1: bits [B-1 .. 0], ...
        bit_shifts = np.arange(B - 1, -1, -1, dtype=np.int32)  # [B-1, B-2, ..., 0]
        # bits_2d shape: (N, D*B) — True where bit is set
        bits_2d = ((idx[:, :, None] >> bit_shifts[None, None, :]) & 1).reshape(N, D * B).astype(np.uint8)

        # packbits pads the last byte with zeros if D*B is not a multiple of 8
        packed = np.packbits(bits_2d, axis=1, bitorder='big')  # (N, index_bytes)
        codes[:, :index_bytes] = packed

        # Append lo and delta as float32 little-endian bytes
        lo_f32 = lo.astype(np.float32)
        delta_f32 = delta.astype(np.float32)
        # View as uint8: each float32 → 4 bytes
        codes[:, index_bytes:index_bytes + 4] = lo_f32.view(np.uint8).reshape(N, 4)
        codes[:, index_bytes + 4:index_bytes + 8] = delta_f32.view(np.uint8).reshape(N, 4)

        return codes

    def decompress(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode codes back to approximate float32 vectors.

        Fully self-contained per row: reads lo/delta from each code row,
        so decompress(codes[ids]) == decompress(codes)[ids].
        """
        if self.mu is None:
            raise RuntimeError("LVQQuantizer.fit() must be called before compress/decompress.")
        codes = np.asarray(codes, dtype=np.uint8)
        N = codes.shape[0]
        D = self.mu.shape[0]
        B = self.num_bits
        index_bytes = math.ceil(D * B / 8)

        # Recover lo and delta from the trailing 8 bytes of each row
        lo_bytes = codes[:, index_bytes:index_bytes + 4].copy()         # (N, 4)
        delta_bytes = codes[:, index_bytes + 4:index_bytes + 8].copy()  # (N, 4)
        lo = lo_bytes.view(np.float32).reshape(N)
        delta = delta_bytes.view(np.float32).reshape(N)

        # Unpack bit-packed indices
        packed = codes[:, :index_bytes]  # (N, index_bytes)
        bits_2d = np.unpackbits(packed, axis=1, bitorder='big')  # (N, index_bytes*8)

        # Slice exactly D*B bits (discard packbits padding)
        bits_2d = bits_2d[:, : D * B]  # (N, D*B)

        # Reconstruct integer indices from bit planes
        bit_shifts = np.arange(B - 1, -1, -1, dtype=np.int32)  # [B-1, ..., 0]
        # Reshape to (N, D, B) and combine bits
        bits_3d = bits_2d.reshape(N, D, B).astype(np.int32)
        idx = (bits_3d * (1 << bit_shifts)[None, None, :]).sum(axis=2)  # (N, D)

        # Reconstruct residuals and add mean
        r_hat = lo[:, None] + idx.astype(np.float32) * delta[:, None]
        return r_hat + self.mu[None, :]

    def get_compression_ratio(self, X: np.ndarray) -> float:
        """Original float32 bytes per dim vs. compressed bytes per dim."""
        D = X.shape[1]
        B = self.num_bits
        index_bytes = math.ceil(D * B / 8)
        code_size = index_bytes + 8
        original_bytes = D * 4  # float32
        return original_bytes / code_size
