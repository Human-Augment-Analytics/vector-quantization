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
