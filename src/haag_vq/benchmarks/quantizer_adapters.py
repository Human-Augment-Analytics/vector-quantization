# src/haag_vq/benchmarks/quantizer_adapters.py
"""Quantizer protocol + adapters for the benchmark study.

Every method is compared as a pure encoder: fit on the database, reconstruct
vectors by global id, and report total stored bytes (incl. the 4-byte/vector
exact-norm side-channel required by the q.x/||x|| distance).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from haag_vq.methods.base_quantizer import BaseQuantizer

NORM_SIDECHANNEL_BYTES = 4  # float32 exact norm per vector


@runtime_checkable
class Quantizer(Protocol):
    """Minimal interface the exact-search harness needs from each method."""

    def fit(self, X: np.ndarray) -> None: ...
    def reconstruct(self, ids: np.ndarray) -> np.ndarray: ...
    def code_bytes(self) -> int: ...


class FaissQuantizerAdapter:
    """Adapts a haag_vq BaseQuantizer (PQ/OPQ/SQ) to the Quantizer protocol.

    Compresses all database vectors at fit() and reconstructs by id from the
    cached codes.
    """

    def __init__(self, quantizer: BaseQuantizer) -> None:
        self._q = quantizer
        self._codes: Optional[np.ndarray] = None
        self._n = 0

    def fit(self, X: np.ndarray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._q.fit(X)
        codes = self._q.compress(X)   # may raise; leave prior state intact
        self._codes = codes
        self._n = X.shape[0]

    def reconstruct(self, ids: np.ndarray) -> np.ndarray:
        if self._codes is None:
            raise RuntimeError("FaissQuantizerAdapter.reconstruct() before fit()")
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0):
            raise ValueError("reconstruct(): negative ids are not allowed")
        return np.asarray(self._q.decompress(self._codes[ids]), dtype=np.float32)

    def code_bytes(self) -> int:
        if self._codes is None:
            raise RuntimeError("FaissQuantizerAdapter.code_bytes() before fit()")
        return int(self._codes.nbytes) + self._n * NORM_SIDECHANNEL_BYTES


class SaqEngineAdapter:
    """Adapts the SAQ C++ wheel's IVF (single-cluster encoder) to the Quantizer
    protocol. Reconstruction via IVF.decompress(ids).

    Method configs:
      - saq_paper : quant_type='CAQ', greedy=False, derive_codebooks=False  (DP alloc, default codebook)
      - ours      : quant_type='CAQ', greedy=True,  derive_codebooks=True   (greedy alloc, native k-means codebook)

    K=1 keeps a single cluster so decompress() covers all vectors (exact-search use).
    """

    def __init__(
        self,
        *,
        quant_type: str = "CAQ",
        avg_bits: float = 4.0,
        greedy: bool = False,
        derive_codebooks: bool = False,
        exact_codebooks: bool = False,
        apply_pca: bool = True,
        K: int = 1,
        num_threads: int = 8,
        seed: int = 0,
        max_bits: int = 13,
    ) -> None:
        import saq as _saq
        self._saq = _saq
        self._quant_type = quant_type
        self._avg_bits = float(avg_bits)
        self._greedy = greedy
        self._derive_codebooks = derive_codebooks
        self._exact_codebooks = exact_codebooks
        self._apply_pca = apply_pca
        self._K = K
        self._num_threads = num_threads
        self._seed = seed
        self._max_bits = max_bits
        self._index = None
        self._n = 0
        self._D = 0

    def _make_config(self):
        saq = self._saq
        cfg = saq.QuantizeConfig()
        cfg.avg_bits = self._avg_bits
        cfg.enable_segmentation = True
        cfg.single.quant_type = getattr(saq.BaseQuantType, self._quant_type)
        cfg.single.random_rotation = True
        cfg.allocator = saq.AllocatorKind.Greedy if self._greedy else saq.AllocatorKind.DP
        return cfg

    def fit(self, X: np.ndarray) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._n, self._D = X.shape
        cfg = self._make_config()
        index = self._saq.IVF(self._n, self._D, self._K, cfg)
        if self._exact_codebooks:
            index.set_derive_codebooks_exact(max_bits=self._max_bits)
        elif self._derive_codebooks:
            index.set_derive_codebooks(max_bits=self._max_bits)
        index.fit(X, self._apply_pca, self._K, self._seed, self._num_threads)
        self._index = index

    def reconstruct(self, ids: np.ndarray) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("SaqEngineAdapter.reconstruct() before fit()")
        ids = np.ascontiguousarray(ids, dtype=np.uint32)
        return np.asarray(self._index.decompress(ids), dtype=np.float32)

    def code_bytes(self) -> int:
        if self._index is None:
            raise RuntimeError("SaqEngineAdapter.code_bytes() before fit()")
        code = int(self._n * self._D * self._avg_bits / 8.0)
        return code + self._n * NORM_SIDECHANNEL_BYTES
