# src/haag_vq/methods/search/rabitq_index.py
"""RaBitQIndex — native-estimator RaBitQ via faiss.IndexRaBitQ."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class RaBitQIndex(BaseSearchIndex):
    """RaBitQ quantizer with native distance-estimator search.

    Uses ``faiss.IndexRaBitQ`` directly: queries are compared against binary
    codes via RaBitQ's unbiased distance estimator. This is substantially
    faster and slightly higher-recall than the prior path that decoded codes
    back to lossy float32 vectors and ran brute-force L2 on the
    reconstructions.

    Per the original RaBitQ paper (Gao & Long, 2024), each vector is encoded
    at ~1 bit per dimension (plus a small per-vector correction term).
    ``bpd`` is not tunable for this method.

    Args:
        qb: Number of bits used to quantise queries at search time
            (``faiss.IndexRaBitQ.qb``). ``qb=4`` is the SIMD-optimised
            sweet spot: ~99% of ``qb=8``'s recall at ~70% the search time
            on MSMarco-scale data. ``qb=0`` disables query quantisation
            (exact float distances computed per code) and is substantially
            slower without a recall advantage.
    """

    def __init__(self, qb: int = 4) -> None:
        self._qb = int(qb)
        self._idx = None  # type: Optional[object]  # faiss.IndexRaBitQ
        self._D: int = 0
        self._metric: Literal['l2', 'ip'] = 'l2'

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        import faiss  # deferred: keep import cost off the package load path
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._D = int(X.shape[1])
        self._metric = metric
        mt = faiss.METRIC_INNER_PRODUCT if metric == 'ip' else faiss.METRIC_L2
        self._idx = faiss.IndexRaBitQ(self._D, mt)
        self._idx.qb = self._qb
        self._idx.train(X)
        self._idx.add(X)

    def _require_fit(self):
        if self._idx is None:
            raise RuntimeError("RaBitQIndex must be fit() before use.")
        return self._idx

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._require_fit()
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        dists, ids = idx.search(Q, k)
        # IndexRaBitQ returns int64 ids and float32 distances; normalise to
        # the BaseSearchIndex contract (uint32 ids, float32 distances).
        return ids.astype(np.uint32), dists.astype(np.float32)

    def memory_footprint(self) -> int:
        if self._idx is None:
            return 0
        # code_size is bytes per stored vector including per-vector correction
        # factors. ntotal * code_size dominates RaBitQ's memory by orders of
        # magnitude (training state is O(D), not O(N)).
        return int(self._idx.ntotal) * int(self._idx.code_size)

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._idx is None:
            return None
        if sample_ids is None:
            sample_ids = np.arange(int(self._idx.ntotal), dtype=np.int64)
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
        X_hat = self._idx.reconstruct_batch(sample_ids).astype(np.float32)
        return float(np.mean((X[sample_ids].astype(np.float32) - X_hat) ** 2))

    def save(self, path: str | Path) -> None:
        import faiss
        if self._idx is None:
            raise RuntimeError("RaBitQIndex.save() called before fit()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._idx, str(path))

    def load(self, path: str | Path) -> None:
        import faiss
        self._idx = faiss.read_index(str(path))
        self._D = int(self._idx.d)
        self._metric = (
            'ip' if self._idx.metric_type == faiss.METRIC_INNER_PRODUCT else 'l2'
        )
        self._qb = int(self._idx.qb)
