import numpy as np

from .base_quantizer import BaseQuantizer


def _lloyd_1d_normal(num_levels: int, seed: int, n_samples: int = 200_000,
                     max_iter: int = 100, tol: float = 1e-7) -> np.ndarray:
    """Compute a 1-D Gaussian-optimal scalar codebook via Lloyd / k-means.

    After rotating a unit residual by a random orthogonal matrix and scaling by
    sqrt(D), each coordinate is ~ N(0, 1). So a single scalar codebook optimal
    for the standard normal serves every coordinate (this mirrors SAQ's
    k-means-codebook idea, applied to N(0,1)).

    Returns sorted centroid levels of length ``num_levels``.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)

    # Initialize levels at evenly-spaced quantiles of the sample so Lloyd
    # converges quickly and deterministically.
    qs = (np.arange(num_levels) + 0.5) / num_levels
    levels = np.quantile(samples, qs)

    for _ in range(max_iter):
        # Assign each sample to nearest level. Since levels are sorted, the
        # decision boundaries are the midpoints between consecutive levels.
        boundaries = 0.5 * (levels[:-1] + levels[1:])
        idx = np.searchsorted(boundaries, samples)

        new_levels = levels.copy()
        for k in range(num_levels):
            mask = idx == k
            if np.any(mask):
                new_levels[k] = samples[mask].mean()
            # Empty cell: keep previous centroid (rare for N(0,1) + quantile init).

        new_levels.sort()
        shift = float(np.max(np.abs(new_levels - levels)))
        levels = new_levels
        if shift < tol:
            break

    return levels.astype(np.float64)


class ExtendedRaBitQuantizer(BaseQuantizer):
    """Standalone multi-bit (Extended) RaBitQ quantizer in numpy.

    Gao, J., & Long, C. — Extended RaBitQ, the multi-bit generalization of
    RaBitQ. faiss's ``RaBitQuantizer`` only behaves at ``nb_bits=1`` (multi-bit
    corrupts memory), and the SAQ engine has no RaBitQ encoder, so this is a
    self-contained, reconstruction-faithful ``BaseQuantizer`` implementation.

    Model state learned in ``fit`` (data-independent, seeded):
      - centroid ``c`` = global mean of the training data.
      - orthogonal rotation ``P`` (D x D) from a seeded QR factorization.
      - a shared per-coordinate B-bit Gaussian-optimal scalar codebook
        ``levels`` (length 2^B), from 1-D Lloyd on a large N(0,1) sample.

    Per-vector code row layout (uint8):
      [ B-bit indices for D coords, bit-packed ]
      ++ [ nrm  as float32 (4 bytes) ]
      ++ [ t    as float32 (4 bytes) ]
    => code_size = ceil(D*B/8) + 8 bytes/vector.

    ``decompress`` parses each row independently: P, c, and levels are shared
    model state, but the per-vector norm ``nrm`` and rescale factor ``t`` come
    solely from the code row (so ``codes[ids]`` decodes correctly).

    NOTE on faithfulness: we include the optional per-vector rescale factor
    ``t = <s, ŝ> / <ŝ, ŝ>`` which minimizes ‖s - t·ŝ‖ and improves fidelity.
    """

    def __init__(self, num_bits: int = 4, seed: int = 0):
        if num_bits < 1 or num_bits > 8:
            raise ValueError("num_bits must be in [1, 8]")
        self.num_bits = int(num_bits)
        self.seed = int(seed)

        # Learned model state (set in fit).
        self.c: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.levels: np.ndarray | None = None
        self.D: int | None = None

        self._eps = 1e-12

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X)
        N, D = X.shape
        self.D = int(D)

        # Global centroid.
        self.c = X.mean(axis=0).astype(np.float64)

        # Random orthogonal rotation via QR. Do QR in float64 for a clean
        # orthogonality guarantee, then keep float64.
        rng = np.random.default_rng(self.seed)
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        Q = Q.astype(np.float64)
        self.P = Q

        # Shared Gaussian-optimal B-bit scalar codebook (1-D Lloyd on N(0,1)).
        self.levels = _lloyd_1d_normal(2 ** self.num_bits, seed=self.seed)

    # -------------------------------------------------------------- helpers
    @property
    def _index_bytes(self) -> int:
        """Number of bytes used to store the bit-packed indices per vector."""
        return (self.D * self.num_bits + 7) // 8

    @property
    def code_size(self) -> int:
        return self._index_bytes + 8  # +4 (nrm) +4 (t), both float32

    def _quantize_to_levels(self, s: np.ndarray) -> np.ndarray:
        """Nearest-level index for each coordinate. s: (N, D) -> idx: (N, D)."""
        boundaries = 0.5 * (self.levels[:-1] + self.levels[1:])
        # searchsorted over the (monotone) boundaries gives the nearest level.
        return np.searchsorted(boundaries, s).astype(np.int64)

    # ------------------------------------------------------------- compress
    def compress(self, X: np.ndarray) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("Quantizer must be fit before compress().")
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        if D != self.D:
            raise ValueError(f"compress() got D={D}, but fit() saw D={self.D}")

        # Residual, norm, and unit direction.
        r = X - self.c
        nrm = np.linalg.norm(r, axis=1)
        nrm_safe = np.maximum(nrm, self._eps)
        o = r / nrm_safe[:, None]

        # Rotate, then scale to unit variance per coordinate (~N(0,1)).
        s = (o @ self.P) * np.sqrt(D)

        # Quantize to nearest codebook level.
        idx = self._quantize_to_levels(s)          # (N, D) ints in [0, 2^B-1]
        s_hat = self.levels[idx]                    # dequantized (N, D)

        # Per-vector rescale factor t = <s, ŝ> / <ŝ, ŝ> minimizing ‖s - t·ŝ‖.
        num = np.einsum("nd,nd->n", s, s_hat)
        den = np.einsum("nd,nd->n", s_hat, s_hat)
        t = np.where(den > self._eps, num / den, 1.0)

        # --- Pack each row ---
        out = np.zeros((N, self.code_size), dtype=np.uint8)

        # Bit-pack indices. Expand each B-bit index into B bits (MSB-first),
        # flatten per row, then packbits. Verified round-trip in unpack.
        bit_positions = np.arange(self.num_bits - 1, -1, -1)  # MSB..LSB
        bits = ((idx[:, :, None] >> bit_positions) & 1).astype(np.uint8)  # (N, D, B)
        bits = bits.reshape(N, D * self.num_bits)
        packed = np.packbits(bits, axis=1)  # (N, _index_bytes)
        out[:, : self._index_bytes] = packed

        # Append nrm and t as float32 byte views.
        # nrm stored as float32: ~7 sig-figs, ample for unit-ish residuals; if a
        # future dataset has very large ||r|| (>1e4) this becomes the precision floor.
        nrm_f32 = nrm.astype(np.float32).view(np.uint8).reshape(N, 4)
        t_f32 = t.astype(np.float32).view(np.uint8).reshape(N, 4)
        out[:, self._index_bytes : self._index_bytes + 4] = nrm_f32
        out[:, self._index_bytes + 4 : self._index_bytes + 8] = t_f32

        return out

    # ----------------------------------------------------------- decompress
    def decompress(self, codes: np.ndarray) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("Quantizer must be fit before decompress().")
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        N = codes.shape[0]
        D = self.D
        B = self.num_bits

        # --- Unpack indices ---
        packed = codes[:, : self._index_bytes]
        bits = np.unpackbits(packed, axis=1)[:, : D * B]      # (N, D*B), MSB-first
        bits = bits.reshape(N, D, B).astype(np.int64)
        weights = (1 << np.arange(B - 1, -1, -1)).astype(np.int64)  # MSB..LSB
        idx = bits @ weights                                   # (N, D)

        # Dequantize and apply per-vector pieces (from the code row).
        s_hat = self.levels[idx]                               # (N, D)
        nrm = codes[:, self._index_bytes : self._index_bytes + 4].copy().view(np.float32).reshape(N).astype(np.float64)
        t = codes[:, self._index_bytes + 4 : self._index_bytes + 8].copy().view(np.float32).reshape(N).astype(np.float64)

        # Un-normalize variance, apply rescale factor.
        o_hat = (s_hat / np.sqrt(D)) * t[:, None]

        # Inverse rotation, restore norm, add centroid.
        x_hat = (o_hat @ self.P.T) * nrm[:, None] + self.c
        return x_hat.astype(np.float32)

    # ----------------------------------------------------- compression ratio
    def get_compression_ratio(self, X: np.ndarray) -> float:
        """Original bytes (D*4, float32) over compressed bytes (code_size)."""
        D = int(X.shape[1])
        return float(D * 4 / self.code_size)
