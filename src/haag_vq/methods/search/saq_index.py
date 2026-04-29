# src/haag_vq/methods/search/saq_index.py
"""SaqIndex — wraps saq.IVF (or saq.GpuIVF) as a BaseSearchIndex."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex


class SaqIndex(BaseSearchIndex):
    """Wraps the SAQ C++ IVF index as a BaseSearchIndex.

    Preprocessing (PCA + k-means) is done inside SAQ's native ``fit()`` so
    that raw codes are cached, enabling ``decompress()`` and reconstruction
    MSE. The previous Python-side faiss preprocessing path went through
    ``construct()`` and could not report MSE.

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

        # Avoid asking for more clusters than there is data.
        K = min(self._K, self._N // 10)
        self._K = K

        IndexClass = self._saq.GpuIVF if self._use_gpu else self._saq.IVF
        self._index = IndexClass(self._N, self._D, K, cfg)

        # Native fit (k-means + construct, no PCA) so raw codes are cached
        # and decompress() / reconstruction_mse work. apply_pca=True rotates
        # the indexed data without applying the same transform to queries
        # at search time, which collapses recall to ~0; apply_pca=False
        # avoids that and matches the prior construct()-path recall within
        # ~1pt (0.903 vs 0.915 on MSMarco 100K bpd=4) — the gap is SAQ's
        # internal k-means vs faiss-cpu, not a correctness issue.
        self._index.fit(
            X,
            apply_pca=False,
            K=K,
            seed=0,
            num_threads=self._num_threads,
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
        """Mean squared error between original and SAQ-reconstructed vectors.

        Uses ``IVF.decompress(ids)`` which is available because ``fit()`` is
        called with raw-code caching. If ``sample_ids`` is None, evaluates
        on all rows (only feasible for small N).
        """
        if self._index is None:
            return None
        if sample_ids is None:
            sample_ids = np.arange(self._N, dtype=np.uint32)
        else:
            sample_ids = np.ascontiguousarray(sample_ids, dtype=np.uint32)
        recon = self._index.decompress(sample_ids)
        original = X[sample_ids]
        return float(np.mean(np.sum((original - recon) ** 2, axis=1)))
