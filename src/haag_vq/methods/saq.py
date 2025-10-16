"""
SAQ (placeholder) quantizer.

This module scaffolds an SAQ quantizer to match the project’s
BaseQuantizer API (fit/compress/decompress). The concrete algorithmic
details depend on the target SAQ variant from the referenced paper.

Once the intended SAQ is confirmed (e.g., subspace/asymmetric/additive
quantization variant and its training/encoding specifics), fill in the
implementation blocks below.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .base_quantizer import BaseQuantizer


class SAQ(BaseQuantizer):
    """
    Segmented CAQ (SAQ): PCA rotation + segment-wise bit allocation with
    uniform scalar quantization per dimension.

    - Rotation: PCA (orthonormal), i.e., X_c @ P maps to rotated space o.
    - Quantizer: For a rotated vector o, let o_max = max(|o|). For a dimension
      with bitwidth B, define delta = 2*o_max / (2**B). Encode index
      epsilon = floor((o + o_max)/delta) clipped to [0, 2**B - 1].
      Decode o_hat = delta * (epsilon + 0.5) - o_max.
    - Segmentation/bit allocation: Dimensions are grouped into segments and a
      single bitwidth is assigned per segment under a global bit budget.

    Parameters
    ----------
    num_bits : int
        Default per-dimension bitwidth (used when `total_bits` is not set).
        Allowed values are in the range [0, 8], where 0 means the dimension is
        dropped (quantized to 0 in rotated space).
    total_bits : Optional[int]
        If provided, a global per-vector bit budget. The model will choose
        segment boundaries and per-segment bitwidths to (greedily) minimize a
        quantization error model under this budget.
    allowed_bits : Optional[Sequence[int]]
        Discrete bitwidths permitted for segments. Defaults to [0..8]. Include 0
        to allow the allocator to drop low-importance dimensions.
    n_segments : Optional[int]
        Desired number of segments. If None, a heuristic is used.
    random_state : Optional[int]
        Seed for deterministic PCA sign resolution.
    """

    def __init__(
        self,
        num_bits: int = 8,
        *,
        total_bits: Optional[int] = None,
        allowed_bits: Optional[Sequence[int]] = None,
        n_segments: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        valid = set(range(0, 9))  # 0..8 bits per dimension
        if num_bits not in valid:
            raise ValueError(f"num_bits must be in [0..8]; got {num_bits}")
        self.num_bits = int(num_bits)
        self.total_bits = int(total_bits) if total_bits is not None else None
        self.allowed_bits: Tuple[int, ...] = tuple(sorted(allowed_bits or list(range(0, 9))))
        if not set(self.allowed_bits).issubset(valid):
            raise ValueError("allowed_bits must be within [0..8]")
        self.n_segments = n_segments
        self.random_state = random_state

        # Learned parameters
        self.dim: Optional[int] = None
        self.mean_: Optional[np.ndarray] = None  # (D,)
        self.P_: Optional[np.ndarray] = None  # (D,D) orthonormal rotation
        self.var_rot_: Optional[np.ndarray] = None  # per-dim variance after rotation
        self.segments_: Optional[Tuple[Tuple[int, int, int], ...]] = None  # (start,end_exclusive,B)
        self.bits_per_dim_: Optional[np.ndarray] = None  # (D,) per-dim bitwidths

        # Buffers for last encoding (needed for reconstruction using per-vector o_max)
        self._last_o_max: Optional[np.ndarray] = None  # (N,)

        self.fitted: bool = False

    # ---- PCA rotation and variance ----------------------------------------
    @staticmethod
    def _pca_rotation(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean, rotation P, and per-dimension variance in rotated space.

        Uses SVD on centered data: Xc = X - mean; Xc = U S V^T; rotation P = V.
        """
        X = np.asarray(X, dtype=np.float32)
        mean = X.mean(axis=0)
        Xc = X - mean
        # Economy SVD; Vt has principal axes as rows
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Vt.T.astype(np.float32, copy=False)  # (D,D), orthonormal
        # Variance along rotated axes equals (S^2)/(N-1)
        N = max(1, X.shape[0] - 1)
        var_rot = (S.astype(np.float32) ** 2) / N
        # If D > rank, pad zeros
        if var_rot.shape[0] < P.shape[1]:
            pad = np.zeros(P.shape[1] - var_rot.shape[0], dtype=np.float32)
            var_rot = np.concatenate([var_rot, pad], axis=0)
        return mean.astype(np.float32), P, var_rot

    # ---- Segment allocation (greedy under budget) -------------------------
    @staticmethod
    def _segment_by_cumvar(var_rot: np.ndarray, n_segments: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return segment boundaries over dimensions by cumulative variance.

        Returns (starts, ends) arrays with `n_segments` segments that partition
        [0, D). Segments follow the PCA order (no reordering), sized so each
        covers an approximately equal share of total variance.
        """
        D = int(var_rot.shape[0])
        total = float(var_rot.sum())
        if total <= 0:
            # fallback to equal sized segments
            base = np.linspace(0, D, n_segments + 1, dtype=int)
            return base[:-1], base[1:]
        csum = np.cumsum(var_rot)
        targets = np.linspace(total / n_segments, total, n_segments)
        starts = [0]
        ends = []
        last = 0
        for t in targets[:-1]:
            # find smallest idx such that csum[idx] - csum[last] >= t_segment
            idx = np.searchsorted(csum, t)
            idx = int(np.clip(idx, last + 1, D - (n_segments - len(starts))))
            ends.append(idx)
            starts.append(idx)
            last = idx
        ends.append(D)
        return np.array(starts, dtype=int), np.array(ends, dtype=int)

    @staticmethod
    def _segment_cost(seg_var_sum: float, bits: int) -> float:
        # Error model: MSE ~ seg_var_sum / 4^bits (constant factors omitted)
        return float(seg_var_sum) / float(4 ** int(bits))

    @classmethod
    def _allocate_bits_greedy(
        cls,
        var_rot: np.ndarray,
        *,
        total_bits: int,
        allowed_bits: Sequence[int],
        n_segments: int,
    ) -> Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray]:
        """Greedy bit allocation across segments under a bit budget.

        - Partition dims into `n_segments` by cumulative variance.
        - Initialize each segment at the minimum allowed bitwidth.
        - Iteratively promote segments to the next higher allowed bitwidth
          maximizing error reduction per consumed bit until the budget is
          exhausted or no more promotions are possible.
        Returns segment tuples and a per-dimension bitwidth array.
        """
        D = int(var_rot.shape[0])
        allowed = sorted(int(b) for b in allowed_bits)
        min_b, max_b = allowed[0], allowed[-1]
        starts, ends = cls._segment_by_cumvar(var_rot, n_segments)
        seg_vars = np.array([var_rot[s:e].sum() for s, e in zip(starts, ends)], dtype=np.float64)
        seg_lens = (ends - starts).astype(int)
        # Initialize bits
        seg_bits = np.full(n_segments, min_b, dtype=int)
        used_bits = int(np.dot(seg_lens, seg_bits))
        if used_bits > total_bits:
            raise ValueError(
                f"Budget {total_bits} is smaller than minimum feasible {used_bits} "
                f"with {n_segments} segments at {min_b} bits"
            )
        # Precompute next-higher mapping for allowed bits
        next_idx = {b: i for i, b in enumerate(allowed)}

        def seg_cost(i: int, b: int) -> float:
            return cls._segment_cost(seg_vars[i], b)

        # Greedily increase where benefit per added bit is largest
        while True:
            best_gain = 0.0
            best_idx = -1
            best_db = 0
            for i in range(n_segments):
                curr_b = int(seg_bits[i])
                idx = next_idx[curr_b]
                if idx == len(allowed) - 1:
                    continue
                next_b = allowed[idx + 1]
                db = (next_b - curr_b) * int(seg_lens[i])
                if used_bits + db > total_bits:
                    continue
                gain = seg_cost(i, curr_b) - seg_cost(i, next_b)
                # Normalize by consumed bits to balance segment sizes
                norm_gain = gain / max(1, db)
                if norm_gain > best_gain:
                    best_gain = norm_gain
                    best_idx = i
                    best_db = db
            if best_idx < 0:
                break
            # Promote
            curr_b = int(seg_bits[best_idx])
            nxt = allowed[next_idx[curr_b] + 1]
            seg_bits[best_idx] = int(nxt)
            used_bits += best_db

        # Build per-dimension bit vector and segment tuples
        bits_per_dim = np.empty(D, dtype=int)
        segments = []
        for s, e, b in zip(starts, ends, seg_bits):
            bits_per_dim[s:e] = int(b)
            segments.append((int(s), int(e), int(b)))
        return tuple(segments), bits_per_dim

    @classmethod
    def _allocate_bits_dp(
        cls,
        var_rot: np.ndarray,
        *,
        total_bits: int,
        allowed_bits: Sequence[int],
        max_iters: int = 30,
    ) -> Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray]:
        """Dynamic programming with Lagrangian relaxation over budget.

        Solves segmentation with per-segment bitwidths to minimize modeled
        error subject to a global bit budget. Uses a scalar penalty `lambda`
        for bits and binary searches to match the budget approximately.

        Returns segment tuples (start, end, bits) and per-dimension bits.
        """
        D = int(var_rot.shape[0])
        allowed = sorted(int(b) for b in allowed_bits)
        pref = np.concatenate([[0.0], np.cumsum(var_rot.astype(np.float64))])

        def seg_var(i: int, j: int) -> float:
            return float(pref[j] - pref[i])

        # Precompute seg cost for (i,j,b) on demand inside DP via small loop over allowed bits
        def solve_for_lambda(lmbd: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # DP over positions j in [0..D]
            dp = np.full(D + 1, np.inf, dtype=np.float64)
            prev = np.full(D + 1, -1, dtype=int)
            chosen_b = np.full(D + 1, -1, dtype=int)
            dp[0] = 0.0
            for j in range(1, D + 1):
                best_cost = np.inf
                best_i = -1
                best_bits = -1
                for i in range(0, j):
                    len_seg = j - i
                    v = seg_var(i, j)
                    # choose bits minimizing penalized cost
                    local_best = np.inf
                    local_bits = -1
                    for b in allowed:
                        c = cls._segment_cost(v, b) + lmbd * (len_seg * b)
                        if c < local_best:
                            local_best = c
                            local_bits = b
                    cost = dp[i] + local_best
                    if cost < best_cost:
                        best_cost = cost
                        best_i = i
                        best_bits = local_bits
                dp[j] = best_cost
                prev[j] = best_i
                chosen_b[j] = best_bits
            return dp, prev, chosen_b

        def reconstruct(prev: np.ndarray, chosen_b: np.ndarray) -> Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray, int]:
            segs = []
            bits_per_dim = np.empty(D, dtype=int)
            j = D
            total = 0
            while j > 0:
                i = int(prev[j])
                b = int(chosen_b[j])
                if i < 0 or b < 0:
                    # fallback to one segment if reconstruction fails
                    return ((0, D, allowed[0]),), np.full(D, allowed[0], dtype=int), D * allowed[0]
                bits_per_dim[i:j] = b
                segs.append((i, j, b))
                total += (j - i) * b
                j = i
            segs.reverse()
            return tuple(segs), bits_per_dim, int(total)

        # Binary search lambda to meet total_bits
        # heuristic bounds
        lo, hi = 0.0, 1.0
        # Increase hi until budget undershoots
        for _ in range(20):
            _, prev, cb = solve_for_lambda(hi)
            segs, _, used = reconstruct(prev, cb)
            if used <= total_bits:
                break
            hi *= 2.0
        best_bits = None
        best_diff = float('inf')
        best_segs = None
        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            _, prev, cb = solve_for_lambda(mid)
            segs, bits_per_dim, used = reconstruct(prev, cb)
            diff = abs(used - total_bits)
            if diff < best_diff:
                best_diff = diff
                best_bits = bits_per_dim
                best_segs = segs
            if used > total_bits:
                # too many bits, increase penalty
                lo = mid
            elif used < total_bits:
                hi = mid
            else:
                break

        if best_bits is None or best_segs is None:
            # safety fallback
            return ((0, D, allowed[0]),), np.full(D, allowed[0], dtype=int)
        return best_segs, best_bits

    # ---- BaseQuantizer API -------------------------------------------------
    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (N, D)")
        if X.size == 0:
            raise ValueError("X is empty")

        self.dim = int(X.shape[1])
        mean, P, var_rot = self._pca_rotation(X)
        self.mean_, self.P_, self.var_rot_ = mean, P, var_rot

        # Allocate bits per dimension
        if self.total_bits is None:
            # CAQ baseline: constant bits per dim
            bits_per_dim = np.full(self.dim, self.num_bits, dtype=int)
            segments = ((0, self.dim, self.num_bits),)
        else:
            # SAQ: segmented allocation under a global budget
            D = int(self.dim)
            use_dp = (D <= 1024)
            if use_dp:
                segments, bits_per_dim = self._allocate_bits_dp(
                    var_rot,
                    total_bits=int(self.total_bits),
                    allowed_bits=self.allowed_bits,
                )
            else:
                # Guardrail fallback to greedy
                if self.n_segments is None:
                    self.n_segments = max(1, min(8, D))
                segments, bits_per_dim = self._allocate_bits_greedy(
                    var_rot,
                    total_bits=int(self.total_bits),
                    allowed_bits=self.allowed_bits,
                    n_segments=int(self.n_segments),
                )

        self.segments_ = segments
        self.bits_per_dim_ = bits_per_dim.astype(int)
        self.fitted = True

    def compress(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SAQ must be fitted before compress()")
        if self.dim is None or self.mean_ is None or self.P_ is None or self.bits_per_dim_ is None:
            raise RuntimeError("Model is not fully initialized; call fit() first")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != self.dim:
            raise ValueError("Input dimensionality does not match fitted model")

        # Rotate
        Xc = X - self.mean_[None, :]
        O = Xc @ self.P_  # (N,D)
        # Per-vector range parameter
        o_max = np.max(np.abs(O), axis=1)  # (N,)
        # Avoid degenerate zero range
        o_max = np.maximum(o_max, 1e-12)

        # Prepare deltas per (N,D) with varying per-dim bits
        bits = self.bits_per_dim_.astype(np.int32)
        levels = np.power(2, bits, dtype=np.int32)  # (D,)
        # Broadcast: delta = 2*o_max / levels
        delta = (2.0 * o_max[:, None]) / levels[None, :].astype(np.float32)
        # Encode epsilon indices
        eps = np.floor((O + o_max[:, None]) / delta).astype(np.int64)
        eps = np.clip(eps, 0, (levels - 1)[None, :])

        # Store buffers for reconstruction
        self._last_o_max = o_max.astype(np.float32)

        # Use uint8 for 0..8-bit codes per dimension
        return eps.astype(np.uint8, copy=False)

    # Convenience API to return side information making codes self-contained
    def compress_with_info(self, X: np.ndarray, *, return_ip_hint: bool = False) -> Tuple[np.ndarray, dict]:
        if not self.fitted:
            raise RuntimeError("SAQ must be fitted before compress()")
        if self.dim is None or self.mean_ is None or self.P_ is None or self.bits_per_dim_ is None:
            raise RuntimeError("Model is not fully initialized; call fit() first")
        X = np.asarray(X, dtype=np.float32)
        Xc = X - self.mean_[None, :]
        O = Xc @ self.P_
        o_max = np.max(np.abs(O), axis=1)
        o_max = np.maximum(o_max, 1e-12)
        bits = self.bits_per_dim_.astype(np.int32)
        levels = np.power(2, bits, dtype=np.int32)
        delta = (2.0 * o_max[:, None]) / levels[None, :].astype(np.float32)
        eps = np.floor((O + o_max[:, None]) / delta).astype(np.int64)
        eps = np.clip(eps, 0, (levels - 1)[None, :])
        self._last_o_max = o_max.astype(np.float32)

        info = {
            "o_max": self._last_o_max.copy(),
            "x_norm": np.linalg.norm(X, axis=1).astype(np.float32),
        }
        if return_ip_hint:
            # Optional inner-product hint: <O, O_hat>
            O_hat = delta * (eps.astype(np.float32) + 0.5) - o_max[:, None]
            info["o_dot_ohat"] = np.sum(O_hat * O, axis=1).astype(np.float32)
        return eps.astype(np.uint8, copy=False), info

    def decompress(self, codes: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SAQ must be fitted before decompress()")
        if self.dim is None or self.mean_ is None or self.P_ is None or self.bits_per_dim_ is None:
            raise RuntimeError("Model is not fully initialized; call fit() first")
        if self._last_o_max is None:
            raise RuntimeError(
                "Decompression requires per-vector o_max from the last compress() call"
            )

        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.dim:
            raise ValueError("codes must have shape (N, D)")
        if codes.shape[0] != self._last_o_max.shape[0]:
            raise ValueError("codes count does not match last encoded batch")

        bits = self.bits_per_dim_.astype(np.int32)
        levels = np.power(2, bits, dtype=np.int32)  # (D,)
        delta = (2.0 * self._last_o_max[:, None]) / levels[None, :].astype(np.float32)
        O_hat = delta * (codes.astype(np.float32) + 0.5) - self._last_o_max[:, None]
        X_hat = O_hat @ self.P_.T + self.mean_[None, :]
        return X_hat.astype(np.float32)

    def decompress_with_info(self, codes: np.ndarray, info: dict) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SAQ must be fitted before decompress()")
        if self.dim is None or self.mean_ is None or self.P_ is None or self.bits_per_dim_ is None:
            raise RuntimeError("Model is not fully initialized; call fit() first")
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.dim:
            raise ValueError("codes must have shape (N, D)")
        if "o_max" not in info:
            raise ValueError("info must contain 'o_max' per-vector values")
        o_max = np.asarray(info["o_max"], dtype=np.float32)
        if o_max.ndim != 1 or o_max.shape[0] != codes.shape[0]:
            raise ValueError("info['o_max'] must have shape (N,)")
        bits = self.bits_per_dim_.astype(np.int32)
        levels = np.power(2, bits, dtype=np.int32)
        delta = (2.0 * o_max[:, None]) / levels[None, :].astype(np.float32)
        O_hat = delta * (codes.astype(np.float32) + 0.5) - o_max[:, None]
        X_hat = O_hat @ self.P_.T + self.mean_[None, :]
        return X_hat.astype(np.float32)

    def get_compression_ratio(self, X: np.ndarray) -> float:
        if self.dim is None:
            raise RuntimeError("Call fit() before estimating compression ratio")
        original_bytes = X.shape[1] * 4  # float32 per dimension
        if self.bits_per_dim_ is None:
            raise RuntimeError("Model bits_per_dim not initialized")
        total_bits = int(self.bits_per_dim_.sum())
        compressed_bytes = max(1, (total_bits + 7) // 8)
        return original_bytes / compressed_bytes
