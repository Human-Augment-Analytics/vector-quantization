# src/haag_vq/methods/search/faiss_ivfpq_index.py
"""FaissIvfPqIndex — external baseline wrapping faiss.IndexIVFPQ."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class FaissIvfPqIndex(BaseSearchIndex):
    """External baseline: faiss.IndexIVFPQ.

    Reference point: if SAQ doesn't beat IVFPQ on recall@k at the same
    bits-per-dim, the algorithm isn't pulling its weight.
    """

    def __init__(
        self,
        K: int = 4096,
        m: int = 16,
        nbits: int = 8,
        nprobe: int = 200,
    ) -> None:
        """
        Args:
            K:      Number of IVF cells (coarse quantizer centroids).
            m:      Number of PQ subspaces (subquantizers).
            nbits:  Bits per subspace code (ksub = 2**nbits).
            nprobe: Number of IVF cells probed at search time.
        """
        import faiss as _faiss
        self._faiss = _faiss
        self._K = K
        self._m = m
        self._nbits = nbits
        self._nprobe = nprobe
        self._index = None
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._N, self._D = X.shape
        self._metric = metric

        if metric == 'ip':
            quantizer = self._faiss.IndexFlatIP(self._D)
            self._index = self._faiss.IndexIVFPQ(
                quantizer, self._D, self._K, self._m, self._nbits,
                self._faiss.METRIC_INNER_PRODUCT,
            )
        else:
            quantizer = self._faiss.IndexFlatL2(self._D)
            self._index = self._faiss.IndexIVFPQ(
                quantizer, self._D, self._K, self._m, self._nbits,
            )
        self._index.train(X)
        self._index.add(X)
        self._index.nprobe = self._nprobe

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        _, ids = self._index.search(Q, k)
        return ids.astype(np.uint32)

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        dists, ids = self._index.search(Q, k)
        return ids.astype(np.uint32), dists.astype(np.float32)

    def memory_footprint(self) -> int:
        if self._index is None:
            return 0
        centroid_bytes = self._K * self._D * 4           # float32 coarse centroids
        code_bytes = self._N * self._m                    # 1 byte per subspace code (nbits<=8)
        codebook_bytes = self._K * self._m * (1 << self._nbits) * 4  # PQ codebook float32
        return centroid_bytes + code_bytes + codebook_bytes

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        # IndexIVFPQ.reconstruct() exists but is not reliable across all Faiss
        # builds. Return None — documented limitation.
        return None

    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        self._index = self._faiss.read_index(str(path))
