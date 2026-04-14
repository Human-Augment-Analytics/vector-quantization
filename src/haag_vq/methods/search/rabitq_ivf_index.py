# src/haag_vq/methods/search/rabitq_ivf_index.py
"""RaBitQIVFIndex — IVF coarse-quantisation + RaBitQ fine codes via faiss."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class RaBitQIVFIndex(BaseSearchIndex):
    """IVF + RaBitQ: coarse Voronoi cells with RaBitQ-encoded residuals.

    Compared to flat ``RaBitQIndex``, the IVF structure restricts the scan to
    ``nprobe`` closest cells and delivers meaningfully higher recall at
    comparable code-size and substantially higher QPS on large corpora.

    Args:
        nlist:  Number of IVF centroids (coarse quantiser size).
        nprobe: Cells to inspect at search time; trades recall for speed.
        qb:     Query-bit quantisation for RaBitQ's distance estimator.
                See :class:`RaBitQIndex` for the rationale on the default.
    """

    def __init__(self, nlist: int = 256, nprobe: int = 64, qb: int = 4) -> None:
        self._nlist = int(nlist)
        self._nprobe = int(nprobe)
        self._qb = int(qb)
        self._idx = None  # type: Optional[object]  # faiss.IndexIVFRaBitQ
        self._D: int = 0
        self._metric: Literal['l2', 'ip'] = 'l2'

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._D = int(X.shape[1])
        self._metric = metric
        mt = faiss.METRIC_INNER_PRODUCT if metric == 'ip' else faiss.METRIC_L2
        self._idx = faiss.index_factory(self._D, f"IVF{self._nlist},RaBitQ", mt)
        # qb lives on the inner RaBitQ; nprobe on the IVF wrapper.
        self._idx.qb = self._qb
        self._idx.nprobe = self._nprobe
        self._idx.train(X)
        self._idx.add(X)

    def _require_fit(self):
        if self._idx is None:
            raise RuntimeError("RaBitQIVFIndex must be fit() before use.")
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
        return ids.astype(np.uint32), dists.astype(np.float32)

    def memory_footprint(self) -> int:
        if self._idx is None:
            return 0
        # IVF inverted lists hold (id, code) pairs; each code is rabitq.code_size
        # bytes. Add ntotal * 8 for int64 ids (faiss storage overhead) and the
        # coarse centroids.
        code_bytes = int(self._idx.ntotal) * int(self._idx.rabitq.code_size)
        ids_bytes = int(self._idx.ntotal) * 8
        centroid_bytes = self._nlist * self._D * 4  # float32 centroids
        return code_bytes + ids_bytes + centroid_bytes

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        # IVF+RaBitQ reconstruction requires decoding within each inverted list
        # and adding the coarse centroid; faiss doesn't expose a cheap batched
        # reconstruct for this layout. Leave as the default (None) — mse is a
        # secondary metric and not critical for this index.
        return None

    def save(self, path: str | Path) -> None:
        import faiss
        if self._idx is None:
            raise RuntimeError("RaBitQIVFIndex.save() called before fit()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._idx, str(path))

    def load(self, path: str | Path) -> None:
        import faiss
        self._idx = faiss.read_index(str(path))
        self._D = int(self._idx.d)
        self._nlist = int(self._idx.nlist)
        self._nprobe = int(self._idx.nprobe)
        self._qb = int(self._idx.qb)
        self._metric = (
            'ip' if self._idx.metric_type == faiss.METRIC_INNER_PRODUCT else 'l2'
        )
