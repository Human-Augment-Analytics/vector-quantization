# src/haag_vq/methods/search/saq_index.py
"""SaqIndex — wraps saq.IVF (or saq.GpuIVF) as a BaseSearchIndex."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


def _faiss_kmeans(X: np.ndarray, K: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Run k-means via faiss-cpu. Returns (centroids, assignments)."""
    import faiss
    D = X.shape[1]
    kmeans = faiss.Kmeans(D, K, niter=20, seed=seed, verbose=False)
    kmeans.train(X)
    _, assignments = kmeans.index.search(X, 1)
    centroids = kmeans.centroids.copy()
    assignments = assignments.ravel().astype(np.uint32)
    return centroids, assignments


class SaqIndex(BaseSearchIndex):
    """Wraps the SAQ C++ IVF index as a BaseSearchIndex.

    Preprocessing (k-means clustering) is done in Python via faiss-cpu.
    Quantization and search use the SAQ C++ engine (saq wheel).

    Raises ImportError at construction time if saq is not installed.
    """

    def __init__(
        self,
        bpd: float = 4.0,
        K: int = 4096,
        nprobe: int = 200,
        use_gpu: bool = False,
        use_codebook: bool = False,
        num_threads: int = 8,
    ) -> None:
        import saq as _saq
        self._saq = _saq
        self._bpd = bpd
        self._K = K
        self._nprobe = nprobe
        self._use_gpu = use_gpu and hasattr(_saq, 'GpuIVF')
        self._use_codebook = use_codebook
        self._num_threads = num_threads
        self._index = None
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def _make_config(self, metric: str):
        cfg = self._saq.QuantizeConfig()
        cfg.avg_bits = self._bpd
        if metric == 'ip':
            cfg.single.quant_type = self._saq.BaseQuantType.CAQ
        cfg.single.random_rotation = True
        cfg.enable_segmentation = True
        return cfg

    def _make_searcher_config(self):
        scfg = self._saq.SearcherConfig()
        scfg.dist_type = (
            self._saq.DistType.IP if self._metric == 'ip' else self._saq.DistType.L2Sqr
        )
        return scfg

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._metric = metric
        self._N, self._D = X.shape
        cfg = self._make_config(metric)

        # K-means clustering via faiss-cpu
        K = min(self._K, self._N // 10)  # avoid too many clusters for small data
        self._K = K  # reflect the actually-used K in memory_footprint / reporting
        centroids, assignments = _faiss_kmeans(X, K, seed=0)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)

        # Compute per-dimension variance and set on index
        variance = X.var(axis=0).astype(np.float32)

        IndexClass = self._saq.GpuIVF if self._use_gpu else self._saq.IVF
        self._index = IndexClass(self._N, self._D, K, cfg)
        self._index.set_variance(variance)

        # Build IVF index from pre-computed clustering
        if self._use_gpu:
            self._index.construct(X, centroids, assignments)
        else:
            self._index.construct(X, centroids, assignments, self._num_threads)

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        scfg = self._make_searcher_config()
        ids = self._index.search_batch(Q, k, self._nprobe, scfg)
        # search_batch returns uint32 IDs only; distances not exposed by this path.
        dists = np.zeros((Q.shape[0], k), dtype=np.float32)
        return ids, dists

    def memory_footprint(self) -> int:
        if self._index is None:
            return 0
        # Each of N vectors is encoded as D dims at bpd bits/dim. The prior
        # version dropped the D factor, underreporting code storage by ~D×
        # (e.g. 51.2 MB read as 50 KB at N=100K, D=1024, bpd=4).
        code_bytes = int(self._N * self._D * self._bpd / 8.0)
        centroid_bytes = self._K * self._D * 4  # float32
        return code_bytes + centroid_bytes

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("SaqIndex.save() called before fit()")
        self._index.save(str(path))

    def load(self, path: str | Path) -> None:
        self._index = self._saq.IVF()
        self._index.load(str(path))
        self._N = self._index.num_data
        self._D = self._index.num_dim

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        # decompress() requires fit() (not construct()), which caches raw codes.
        # Since we use construct() with Python-side preprocessing, decompress
        # is not available.
        return None
