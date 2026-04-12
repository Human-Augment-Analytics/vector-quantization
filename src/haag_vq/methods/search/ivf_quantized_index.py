# src/haag_vq/methods/search/ivf_quantized_index.py
"""IvfQuantizedIndex — wraps any BaseQuantizer inside a k-means IVF shell."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex
from haag_vq.methods.base_quantizer import BaseQuantizer


class IvfQuantizedIndex(BaseSearchIndex):
    """Wraps any BaseQuantizer inside a k-means IVF shell.

    Fit: k-means (K clusters) + per-cluster residual quantization.
    Search: find nprobe nearest centroids, search within each cluster,
            decompress candidates, compute exact distance, top-k.

    This is a fair comparison point for SAQ: both methods use IVF +
    per-cluster quantization. The difference is in the quantizer itself.
    """

    def __init__(
        self,
        quantizer_factory: Callable[[], BaseQuantizer],
        K: int = 4096,
        nprobe: int = 200,
    ) -> None:
        self._quantizer_factory = quantizer_factory
        self._K = K
        self._nprobe = nprobe
        self._centroids: Optional[np.ndarray] = None
        self._cluster_quantizers: list[Optional[BaseQuantizer]] = []
        self._cluster_codes: list[np.ndarray] = []
        self._cluster_ids: list[np.ndarray] = []
        self._metric: Literal['l2', 'ip'] = 'l2'
        self._N: int = 0
        self._D: int = 0

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._metric = metric
        self._N, self._D = X.shape

        km = faiss.Kmeans(self._D, self._K, seed=0, verbose=False)
        km.train(X)
        self._centroids = km.centroids.copy()

        _, raw_assign = km.index.search(X, 1)
        assignments = raw_assign.ravel().astype(np.int32)

        self._cluster_quantizers = []
        self._cluster_codes = []
        self._cluster_ids = []

        for c in range(self._K):
            mask = assignments == c
            vid = np.where(mask)[0].astype(np.uint32)
            if not mask.any():
                self._cluster_quantizers.append(None)
                self._cluster_codes.append(np.array([], dtype=np.uint8))
                self._cluster_ids.append(vid)
                continue
            residuals = X[mask] - self._centroids[c]
            q = self._quantizer_factory()
            q.fit(residuals)
            self._cluster_quantizers.append(q)
            self._cluster_codes.append(q.compress(residuals))
            self._cluster_ids.append(vid)

    def _nearest_centroids(self, Q: np.ndarray) -> np.ndarray:
        """Return (nq, nprobe) centroid indices sorted by distance."""
        import faiss
        index = faiss.IndexFlatL2(self._D)
        index.add(self._centroids)
        _, ids = index.search(Q, min(self._nprobe, self._K))
        return ids  # (nq, nprobe)

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        ids, _ = self.search_with_scores(Q, k)
        return ids

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        nq = Q.shape[0]
        probe_ids = self._nearest_centroids(Q)  # (nq, nprobe)

        all_ids = np.full((nq, k), -1, dtype=np.int64)
        all_dists = np.full((nq, k), np.inf, dtype=np.float32)

        for qi in range(nq):
            candidates_id: list[np.ndarray] = []
            candidates_vec: list[np.ndarray] = []
            for cid in probe_ids[qi]:
                q = self._cluster_quantizers[cid]
                if q is None:
                    continue
                codes = self._cluster_codes[cid]
                recon = q.decompress(codes).astype(np.float32) + self._centroids[cid]
                vids = self._cluster_ids[cid]
                candidates_id.append(vids)
                candidates_vec.append(recon)
            if not candidates_id:
                continue
            vids = np.concatenate(candidates_id)
            vecs = np.concatenate(candidates_vec, axis=0)

            if self._metric == 'l2':
                dists = np.sum((Q[qi] - vecs) ** 2, axis=1)
            else:
                dists = -(Q[qi] @ vecs.T)

            topk_local = min(k, len(dists))
            idx = np.argpartition(dists, topk_local - 1)[:topk_local]
            idx_sorted = idx[np.argsort(dists[idx])]

            all_ids[qi, :topk_local] = vids[idx_sorted]
            all_dists[qi, :topk_local] = dists[idx_sorted]

        return all_ids.astype(np.uint32), all_dists

    def memory_footprint(self) -> int:
        centroid_bytes = self._centroids.nbytes if self._centroids is not None else 0
        code_bytes = sum(c.nbytes for c in self._cluster_codes)
        return centroid_bytes + code_bytes

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._centroids is None:
            return None
        if sample_ids is None:
            sample_ids = np.arange(self._N)

        # Build reverse lookup: global id -> (cluster_id, local_idx)
        id_to_loc: dict[int, tuple[int, int]] = {}
        for c in range(self._K):
            for local, gid in enumerate(self._cluster_ids[c]):
                id_to_loc[int(gid)] = (c, local)

        mse_sum = 0.0
        count = 0
        for gid in sample_ids:
            if int(gid) not in id_to_loc:
                continue
            c, local = id_to_loc[int(gid)]
            q = self._cluster_quantizers[c]
            if q is None:
                continue
            code = self._cluster_codes[c][local:local + 1]
            x_hat = q.decompress(code)[0].astype(np.float32) + self._centroids[c]
            mse_sum += float(np.mean((X[gid].astype(np.float32) - x_hat) ** 2))
            count += 1
        return mse_sum / count if count > 0 else None

    def save(self, path: str | Path) -> None:
        """Persist fitted index state to disk.

        The quantizer_factory callable is intentionally excluded — it is only
        needed during fit() and cannot always be pickled (e.g. lambdas).
        Cluster quantizers are serialised individually; those wrapping a
        faiss.ProductQuantizer store the centroid array in place of the raw
        SwigPyObject.
        """
        import faiss

        def _serialise_quantizer(q: Optional[BaseQuantizer]) -> dict:
            if q is None:
                return {'type': 'none'}
            from haag_vq.methods.product_quantization import ProductQuantizer as PQ
            if isinstance(q, PQ) and q.pq is not None:
                flat = faiss.vector_to_array(q.pq.centroids).copy()
                return {'type': 'ProductQuantizer', 'M': q.M, 'B': q.B,
                        'flat_centroids': flat, 'D': q.pq.d}
            return {'type': 'other', 'obj': q}

        serialised_quantizers = [_serialise_quantizer(q) for q in self._cluster_quantizers]

        with open(path, 'wb') as f:
            pickle.dump({
                'K': self._K,
                'nprobe': self._nprobe,
                'centroids': self._centroids,
                'serialised_quantizers': serialised_quantizers,
                'cluster_codes': self._cluster_codes,
                'cluster_ids': self._cluster_ids,
                'metric': self._metric,
                'N': self._N,
                'D': self._D,
            }, f)

    def load(self, path: str | Path) -> None:
        import faiss

        def _deserialise_quantizer(d: dict) -> Optional[BaseQuantizer]:
            if d['type'] == 'none':
                return None
            if d['type'] == 'ProductQuantizer':
                from haag_vq.methods.product_quantization import ProductQuantizer as PQ
                q = PQ(M=d['M'], B=d['B'])
                D = d['D']
                q.chunk_dim = D // d['M']
                q.pq = faiss.ProductQuantizer(D, d['M'], d['B'])
                faiss.copy_array_to_vector(d['flat_centroids'], q.pq.centroids)
                ksub, dsub = q.pq.ksub, q.pq.dsub
                centroids = d['flat_centroids'].reshape(d['M'], ksub, dsub)
                q.codebooks = [np.array(centroids[m], copy=True) for m in range(d['M'])]
                return q
            return d['obj']

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self._K = state['K']
        self._nprobe = state['nprobe']
        self._centroids = state['centroids']
        self._cluster_quantizers = [
            _deserialise_quantizer(d) for d in state['serialised_quantizers']
        ]
        self._cluster_codes = state['cluster_codes']
        self._cluster_ids = state['cluster_ids']
        self._metric = state['metric']
        self._N = state['N']
        self._D = state['D']
