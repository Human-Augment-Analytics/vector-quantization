# src/haag_vq/methods/search/flat_quantized_index.py
"""FlatQuantizedIndex — wraps any BaseQuantizer with brute-force search."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from haag_vq.methods.base_search_index import BaseSearchIndex
from haag_vq.methods.base_quantizer import BaseQuantizer


class FlatQuantizedIndex(BaseSearchIndex):
    """Wraps any BaseQuantizer with brute-force search.

    Fit: compresses all training vectors into codes.
    Search: decompresses stored codes, computes exact distance, top-k.

    Complexity: O(N * D) per query — fair baseline but slow for large N.
    Use IvfQuantizedIndex for faster search at comparable compression.
    """

    def __init__(self, quantizer: BaseQuantizer) -> None:
        self._quantizer = quantizer
        self._codes: Optional[np.ndarray] = None
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._metric = metric
        self._N, self._D = X.shape
        self._quantizer.fit(X)
        self._codes = self._quantizer.compress(X)

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        nq = Q.shape[0]
        k = min(k, self._N)

        if k <= 0:
            empty_ids = np.empty((nq, 0), dtype=np.uint32)
            empty_dists = np.empty((nq, 0), dtype=np.float32)
            return empty_ids, empty_dists

        X_hat = self._quantizer.decompress(self._codes).astype(np.float32)

        if self._metric == 'l2':
            dists = cdist(Q, X_hat, metric='sqeuclidean').astype(np.float32)
            top_k = np.argpartition(dists, k - 1, axis=1)[:, :k]
            top_k_sorted = np.array([
                top_k[i][np.argsort(dists[i][top_k[i]])]
                for i in range(nq)
            ])
        else:  # 'ip'
            sims = (Q @ X_hat.T).astype(np.float32)
            top_k = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_k_sorted = np.array([
                top_k[i][np.argsort(-sims[i][top_k[i]])]
                for i in range(nq)
            ])
            dists = sims

        gathered = np.take_along_axis(dists, top_k_sorted, axis=1)
        return top_k_sorted.astype(np.uint32), gathered

    def memory_footprint(self) -> int:
        return int(self._codes.nbytes) if self._codes is not None else 0

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._codes is None:
            return None
        if sample_ids is None:
            sample_ids = np.arange(self._N)
        X_hat = self._quantizer.decompress(self._codes[sample_ids]).astype(np.float32)
        return float(np.mean((X[sample_ids].astype(np.float32) - X_hat) ** 2))

    def save(self, path: str | Path) -> None:
        """Persist to disk.

        For BaseQuantizer subclasses that wrap a non-picklable C extension
        (e.g. faiss.ProductQuantizer), we call the quantizer's own save
        mechanism if it is a ProductQuantizer; for all other quantizers we
        fall back to pickle. The index state (codes, metric, N, D) is always
        stored as a pickle sidecar alongside the quantizer artefact.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # faiss.ProductQuantizer wraps a SwigPyObject that cannot be pickled.
        # Serialise it by extracting the flat centroid array (a plain ndarray).
        from haag_vq.methods.product_quantization import ProductQuantizer as PQ
        if isinstance(self._quantizer, PQ) and self._quantizer.pq is not None:
            import faiss
            flat_centroids = faiss.vector_to_array(self._quantizer.pq.centroids).copy()
            quantizer_state = {
                'type': 'ProductQuantizer',
                'M': self._quantizer.M,
                'B': self._quantizer.B,
                'flat_centroids': flat_centroids,
                'D': self._quantizer.pq.d,
            }
        else:
            quantizer_state = {'type': 'other', 'obj': self._quantizer}

        with open(path, 'wb') as f:
            pickle.dump({
                'quantizer_state': quantizer_state,
                'codes': self._codes,
                'metric': self._metric,
                'N': self._N,
                'D': self._D,
            }, f)

    def load(self, path: str | Path) -> None:
        import faiss
        with open(path, 'rb') as f:
            state = pickle.load(f)

        qs = state['quantizer_state']
        if qs['type'] == 'ProductQuantizer':
            from haag_vq.methods.product_quantization import ProductQuantizer as PQ
            q = PQ(M=qs['M'], B=qs['B'])
            D = qs['D']
            q.chunk_dim = D // qs['M']
            q.pq = faiss.ProductQuantizer(D, qs['M'], qs['B'])
            faiss.copy_array_to_vector(qs['flat_centroids'], q.pq.centroids)
            ksub = q.pq.ksub
            dsub = q.pq.dsub
            centroids = qs['flat_centroids'].reshape(qs['M'], ksub, dsub)
            q.codebooks = [np.array(centroids[m], copy=True) for m in range(qs['M'])]
            self._quantizer = q
        else:
            self._quantizer = qs['obj']

        self._codes = state['codes']
        self._metric = state['metric']
        self._N = state['N']
        self._D = state['D']
