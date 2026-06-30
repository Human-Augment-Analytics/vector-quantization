import numpy as np

from .base_quantizer import BaseQuantizer
from .ffd_packing import ffd_layout, ffd_encode, ffd_decode


def _lloyd_1d_normal(num_levels: int, seed: int, n_samples: int = 200_000,
                     max_iter: int = 100, tol: float = 1e-7):
    """Compute a 1-D Gaussian-optimal scalar codebook (Lloyd-Max) for N(0,1).

    Returns ``(levels, dmse)`` where ``levels`` are the sorted centroids
    (length ``num_levels``) and ``dmse`` is the resulting normalized MSE on the
    sample (relative to the unquantized variance, i.e. 1.0). For ``num_levels``
    == 1 the optimal level for N(0,1) is 0.0 with normalized MSE 1.0.

    Mirrors the approach in ``extended_rabitq._lloyd_1d_normal``.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)
    var = float(np.mean(samples ** 2))  # ~1.0; used to normalize the MSE

    if num_levels == 1:
        levels = np.array([0.0], dtype=np.float64)
        dmse = float(np.mean((samples - 0.0) ** 2) / var)  # == var/var ~ 1.0
        return levels, dmse

    # Initialize levels at evenly-spaced quantiles of the sample for fast,
    # deterministic Lloyd convergence.
    qs = (np.arange(num_levels) + 0.5) / num_levels
    levels = np.quantile(samples, qs)

    for _ in range(max_iter):
        boundaries = 0.5 * (levels[:-1] + levels[1:])
        idx = np.searchsorted(boundaries, samples)

        new_levels = levels.copy()
        for k in range(num_levels):
            mask = idx == k
            if np.any(mask):
                new_levels[k] = samples[mask].mean()

        new_levels.sort()
        shift = float(np.max(np.abs(new_levels - levels)))
        levels = new_levels
        if shift < tol:
            break

    # Normalized MSE of the final quantizer on the sample.
    boundaries = 0.5 * (levels[:-1] + levels[1:])
    idx = np.searchsorted(boundaries, samples)
    s_hat = levels[idx]
    dmse = float(np.mean((samples - s_hat) ** 2) / var)
    return levels.astype(np.float64), dmse


class RankAwareQuantizer(BaseQuantizer):
    """Rank-aware (var^alpha-weighted) greedy bit-allocation quantizer.

    Tests the hypothesis that the rank-relevant error of a distance estimate
    ``q·x̂`` is dominated by ``Σ_d var_d · MSE_d`` rather than the reconstruction
    objective ``Σ_d MSE_d``. We allocate bits with a greedy whose marginal gain
    weights each PCA dim by ``var_d^alpha``:

      - alpha = 0  -> pure MSE objective (spreads bits, MSE-optimal).
      - alpha > 0  -> concentrates bits on the high-variance head.

    This is the scalar bit-allocation analog of anisotropic VQ (ScaNN).

    Pipeline: center -> PCA rotate -> per-dim Gaussian-optimal scalar codebook
    scaled by sqrt(var_d), with per-dim bit budgets chosen by the rank-aware
    greedy. Codes are per-vector bit-packed; ``decompress`` reconstructs each
    row independently from the global state (``mu, V, var, bits, cb``).
    """

    def __init__(self, avg_bits: float, alpha: float = 1.0,
                 max_bits: int = 8, seed: int = 0, packing: str = "dense",
                 codebook: str = "gaussian"):
        if max_bits < 1 or max_bits > 8:
            raise ValueError("max_bits must be in [1, 8]")
        if packing not in ("dense", "ffd"):
            raise ValueError("packing must be 'dense' or 'ffd'")
        if codebook not in ("gaussian", "lloyd", "exact"):
            raise ValueError("codebook must be 'gaussian', 'lloyd', or 'exact'")
        # codebook: per-dim centroids source (allocation + packing held fixed so this
        # isolates the codebook's effect). 'gaussian' = analytic Lloyd-Max(N(0,1))*sigma;
        # 'lloyd'/'exact' = data-fit on the projected column via the SAQ engine builders.
        self.codebook = codebook
        if packing == "ffd" and max_bits > 8:
            # Each dim's code must fit wholly inside one byte for FFD packing.
            raise ValueError("packing='ffd' requires max_bits <= 8")
        self.avg_bits = float(avg_bits)
        self.alpha = float(alpha)
        self.max_bits = int(max_bits)
        self.seed = int(seed)
        self.packing = packing

        # Learned state (set in fit).
        self.mu: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.bits: np.ndarray | None = None  # public per-dim allocation
        self.cb: list[np.ndarray] | None = None
        self.D: int | None = None
        self._offsets: np.ndarray | None = None  # bit start offset per dim
        self.code_size: int | None = None
        # FFD layout (set in fit when packing == "ffd").
        self._ffd_byte_idx: np.ndarray | None = None
        self._ffd_bit_off: np.ndarray | None = None
        self._ffd_n_bytes: int | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X)                 # keep caller dtype (float32); never upcast all N
        N, D = X.shape
        self.D = int(D)

        # 1+2. One-pass chunked mean + covariance via float64 D-sized accumulators
        #      (sum S, gram G = X^T X), so peak memory is ~one chunk rather than a
        #      full float64 copy of X and Xc. At 53M a float64 (N,D) is ~430GB each;
        #      the old `asarray(float64)` + `X - mu` OOMs. Cov identity is exact:
        #        Cov = E[xx^T] - mu mu^T = G/N - mu mu^T.
        CHUNK = 1_000_000
        S = np.zeros(D, dtype=np.float64)
        G = np.zeros((D, D), dtype=np.float64)
        for s in range(0, N, CHUNK):
            blk = np.asarray(X[s:s + CHUNK], dtype=np.float64)
            S += blk.sum(axis=0)
            G += blk.T @ blk
        self.mu = S / N
        C = G / N - np.outer(self.mu, self.mu)

        w, Vt = np.linalg.eigh(C)        # ascending eigenvalues, columns = eigvecs
        order = np.argsort(w)[::-1]      # descending
        self.var = np.clip(w[order], 1e-12, None)
        self.V = Vt[:, order]            # (D, D), columns = components

        # 3. Per-dim Gaussian-optimal scalar codebooks for b = 0..max_bits.
        #    levels[b]: centroids for a b-bit N(0,1) quantizer (2^b levels).
        #    Dg[b]: its normalized MSE (Dg[0] = 1.0, strictly decreasing).
        self._levels = [None] * (self.max_bits + 1)
        Dg = np.empty(self.max_bits + 1, dtype=np.float64)
        for b in range(self.max_bits + 1):
            lv, dmse = _lloyd_1d_normal(2 ** b, seed=self.seed + b)
            self._levels[b] = lv
            Dg[b] = dmse
        Dg[0] = 1.0  # enforce exact normalization for b=0
        self._Dg = Dg

        # 4. Rank-aware greedy allocation.
        total = int(round(self.avg_bits * D))
        bits = np.zeros(D, dtype=np.int64)

        # marginal weighted gain of dim d going from bits[d] -> bits[d]+1:
        #   var^alpha * (mse_d(b) - mse_d(b+1))
        #     = var^alpha * var * (Dg[b] - Dg[b+1])
        #     = var^(1+alpha) * (Dg[b] - Dg[b+1])
        var_pow = self.var ** (1.0 + self.alpha)  # (D,)

        def gain_at(b_arr):
            # delta MSE per extra bit for current level b_arr (set to -inf when
            # at the cap so those dims are never chosen).
            g = np.full(D, -np.inf, dtype=np.float64)
            ok = b_arr < self.max_bits
            db = b_arr[ok]
            g[ok] = var_pow[ok] * (Dg[db] - Dg[db + 1])
            return g

        gain = gain_at(bits)
        for _ in range(total):
            d_star = int(np.argmax(gain))
            if not np.isfinite(gain[d_star]):
                break  # all dims at cap; budget can't be fully spent
            bits[d_star] += 1
            # recompute only the touched dim's gain
            if bits[d_star] < self.max_bits:
                gain[d_star] = var_pow[d_star] * (Dg[bits[d_star]] - Dg[bits[d_star] + 1])
            else:
                gain[d_star] = -np.inf

        self.bits = bits.astype(np.int64)
        assert self.bits.sum() <= total

        # 5. Per-dim codebooks. 'gaussian' = analytic levels scaled by sqrt(var).
        #    'lloyd'/'exact' = data-fit centroids on the projected column at bits[d].
        scale = np.sqrt(self.var)
        if self.codebook == "gaussian":
            self.cb = [self._levels[int(b)] * scale[d] for d, b in enumerate(self.bits)]
        else:
            import saq
            # Per-dim data-fit codebook (Lloyd/exact). Codebook quality saturates
            # well before full N, so train on a fixed ~200k row sample — and
            # project only that sample (not all N) — so both the build and the
            # projection stay ~constant cost at scale (e.g. 53M). No-op at N<=
            # sample (e.g. 200k), so smaller-scale results are unchanged.
            CB_SAMPLE = 200_000
            N = X.shape[0]
            if N > CB_SAMPLE:
                _idx = np.random.default_rng(0).choice(N, CB_SAMPLE, replace=False)
                Xs = np.asarray(X[_idx], dtype=np.float64) - self.mu
                Y = np.ascontiguousarray(Xs @ self.V)         # (CB_SAMPLE, D)
            else:
                Y = (np.asarray(X, dtype=np.float64) - self.mu) @ self.V   # (N, D)
            self.cb = []
            for d, b in enumerate(self.bits):
                b = int(b)
                if b == 0:
                    self.cb.append(np.array([0.0], dtype=np.float64))
                    continue
                col = np.ascontiguousarray(Y[:, d], dtype=np.float32)
                if self.codebook == "exact":
                    r = saq.build_codebook_exact(col, b)
                else:                              # lloyd
                    o = saq.LloydOpts(); o.max_bits = b
                    r = saq.build_codebook_lloyd(col, o)
                cen = np.asarray(r.codebooks[b].centroids, dtype=np.float64)
                # the dense/ffd encoders need exactly 2^b sorted, strictly-usable levels
                self.cb.append(np.sort(cen))

        # Bit-packing layout: cumulative offsets and total code size.
        self._offsets = np.concatenate(([0], np.cumsum(self.bits))).astype(np.int64)
        total_bits = int(self.bits.sum())

        if self.packing == "ffd":
            # Pack each dim's code wholly inside one byte (byte-aligned decode).
            byte_idx, bit_off, n_bytes = ffd_layout(self.bits)
            self._ffd_byte_idx = byte_idx
            self._ffd_bit_off = bit_off
            self._ffd_n_bytes = int(n_bytes)
            self.code_size = int(n_bytes)
        else:
            self.code_size = (total_bits + 7) // 8

    # -------------------------------------------------------------- helpers
    def _quantize_dim(self, y_d: np.ndarray, d: int) -> np.ndarray:
        """Nearest codebook index for column d. bits[d]==0 -> all zeros."""
        if self.bits[d] == 0:
            return np.zeros(y_d.shape[0], dtype=np.int64)
        cbd = self.cb[d]
        boundaries = 0.5 * (cbd[:-1] + cbd[1:])
        return np.searchsorted(boundaries, y_d).astype(np.int64)

    # ------------------------------------------------------------- compress
    def compress(self, X: np.ndarray) -> np.ndarray:
        if self.V is None:
            raise RuntimeError("Quantizer must be fit before compress().")
        X = np.asarray(X)                 # keep dtype; project per chunk in float64
        N, D = X.shape
        if D != self.D:
            raise ValueError(f"compress() got D={D}, but fit() saw D={self.D}")

        # Chunk over rows so we never hold a full float64 (N,D) projection (~430GB
        # at 53M). Each block's code bytes are independent and concatenable.
        CHUNK = 1_000_000
        if N <= CHUNK:
            return self._compress_block((np.asarray(X, dtype=np.float64) - self.mu) @ self.V)
        out = []
        for s in range(0, N, CHUNK):
            Y = (np.asarray(X[s:s + CHUNK], dtype=np.float64) - self.mu) @ self.V
            out.append(self._compress_block(Y))
        return np.concatenate(out, axis=0)

    def _compress_block(self, Y: np.ndarray) -> np.ndarray:
        """Quantize + pack one block of projected coords Y (Nb, D) -> code bytes."""
        N, D = Y.shape
        if self.packing == "ffd":
            # Quantize each dim to its index, then pack via the FFD byte layout.
            code_mat = np.zeros((N, D), dtype=np.int64)
            for d in range(D):
                if int(self.bits[d]) == 0:
                    continue
                code_mat[:, d] = self._quantize_dim(Y[:, d], d)
            return ffd_encode(code_mat, self.bits, self._ffd_byte_idx,
                              self._ffd_bit_off, self._ffd_n_bytes)

        # Build a per-vector bit matrix (N, total_bits), MSB-first per dim,
        # then packbits into uint8.
        total_bits = int(self.bits.sum())
        bit_mat = np.zeros((N, total_bits), dtype=np.uint8)
        for d in range(D):
            b = int(self.bits[d])
            if b == 0:
                continue
            idx = self._quantize_dim(Y[:, d], d)  # (N,) in [0, 2^b-1]
            off = int(self._offsets[d])
            positions = np.arange(b - 1, -1, -1)  # MSB..LSB
            bit_mat[:, off:off + b] = ((idx[:, None] >> positions) & 1).astype(np.uint8)

        return np.packbits(bit_mat, axis=1)  # (N, code_size)

    # ----------------------------------------------------------- decompress
    def decompress(self, codes: np.ndarray) -> np.ndarray:
        if self.V is None:
            raise RuntimeError("Quantizer must be fit before decompress().")
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        N = codes.shape[0]
        D = self.D
        total_bits = int(self.bits.sum())

        Yhat = np.zeros((N, D), dtype=np.float64)

        if self.packing == "ffd":
            code_mat = ffd_decode(codes, self.bits, self._ffd_byte_idx,
                                  self._ffd_bit_off, D)  # (N, D) indices
            for d in range(D):
                if int(self.bits[d]) == 0:
                    continue  # Yhat stays 0.0
                Yhat[:, d] = self.cb[d][code_mat[:, d]]
            Xhat = Yhat @ self.V.T + self.mu
            return Xhat.astype(np.float32)

        bit_mat = np.unpackbits(codes, axis=1)[:, :total_bits]  # (N, total_bits)

        for d in range(D):
            b = int(self.bits[d])
            if b == 0:
                continue  # Yhat stays 0.0
            off = int(self._offsets[d])
            sub = bit_mat[:, off:off + b].astype(np.int64)  # (N, b), MSB-first
            weights = (1 << np.arange(b - 1, -1, -1)).astype(np.int64)
            idx = sub @ weights  # (N,)
            Yhat[:, d] = self.cb[d][idx]

        Xhat = Yhat @ self.V.T + self.mu
        return Xhat.astype(np.float32)

    # ----------------------------------------------------- compression ratio
    def get_compression_ratio(self, X: np.ndarray) -> float:
        D = int(X.shape[1])
        return float(D * 4 / self.code_size)
