# src/haag_vq/methods/search/saq_index.py
"""SaqIndex — wraps saq.IVF (or saq.GpuIVF) as a BaseSearchIndex."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class SaqIndex(BaseSearchIndex):
    """Wraps the SAQ C++ IVF index as a BaseSearchIndex.

    Capability detection: imports saq at construction time and checks for
    GpuIVF. Variant selection happens at wheel install time — install
    saq-cpu, saq-gpu, saq-codebook, or saq-gpu-codebook.

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
        IndexClass = self._saq.GpuIVF if self._use_gpu else self._saq.IVF
        self._index = IndexClass(self._N, self._D, self._K, cfg)
        self._index.fit(
            X, apply_pca=True, K=self._K, seed=0, num_threads=self._num_threads
        )

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
        code_bytes = int(self._N * self._bpd / 8.0)
        centroid_bytes = self._K * self._D * 4  # float32
        return code_bytes + centroid_bytes

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("SaqIndex.save() called before fit()")
        self._index.save(str(path))

    def load(self, path: str | Path) -> None:
        self._index = self._saq.IVF()
        self._index.load(str(path))

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._index is None:
            return None
        X = np.ascontiguousarray(X, dtype=np.float32)
        if sample_ids is None:
            sample_ids = np.arange(len(X), dtype=np.uint32)
        sample_ids = np.ascontiguousarray(sample_ids, dtype=np.uint32)
        X_hat = self._index.decompress(sample_ids)
        return float(np.mean((X[sample_ids] - X_hat) ** 2))
