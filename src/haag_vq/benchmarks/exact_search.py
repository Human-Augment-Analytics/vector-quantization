# src/haag_vq/benchmarks/exact_search.py
"""Chunked exact-search core for the quantizer benchmark study.

Distance: score(q, x_i) = q . x_hat_i / ||x_i||_exact. We exploit
q . x_hat_i / ||x_i|| = q . (x_hat_i / ||x_i||), so we pre-scale each
reconstruction by 1/||x_i|| and rank by plain inner product via
faiss.IndexFlatIP (exact brute force). Ground truth uses the same metric
over the exact (un-quantized) vectors.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, Tuple

import numpy as np

ReconstructFn = Callable[[np.ndarray], np.ndarray]  # ids (1D uint32) -> (m, D) float32


def compute_exact_norms(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-vector exact L2 norm, floored at eps to avoid division by zero."""
    norms = np.linalg.norm(np.asarray(X, dtype=np.float32), axis=1)
    norms = np.maximum(norms, eps)
    return norms.astype(np.float32)


def _chunks(n: int, chunk: int) -> Iterator[np.ndarray]:
    for start in range(0, n, chunk):
        yield np.arange(start, min(start + chunk, n), dtype=np.uint32)


def build_scaled_ip_index(
    reconstruct_fn: ReconstructFn,
    n: int,
    d: int,
    norms: np.ndarray,
    chunk: int = 50_000,
) -> "faiss.IndexFlatIP":
    """Build a faiss.IndexFlatIP over reconstructions scaled by 1/||x||.

    Reconstructs in chunks so the full (n, d) matrix need not be a single
    allocation peak beyond one chunk + the index storage.
    """
    import faiss

    index = faiss.IndexFlatIP(d)
    for ids in _chunks(n, chunk):
        x_hat = np.ascontiguousarray(reconstruct_fn(ids), dtype=np.float32)
        x_hat *= (1.0 / norms[ids])[:, None]
        index.add(x_hat)
    return index


def normalized_ground_truth(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    norms: np.ndarray | None = None,
    chunk: int = 50_000,
) -> np.ndarray:
    """Exact top-k under q . (x/||x||). Returns (nq, k) uint32 IDs."""
    X = np.asarray(X, dtype=np.float32)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    n, d = X.shape
    if norms is None:
        norms = compute_exact_norms(X)
    index = build_scaled_ip_index(lambda ids: X[ids], n, d, norms, chunk=chunk)
    _, ids = search_index(index, Q, k=k)
    return ids


def search_index(index, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Search a faiss index. Returns (scores (nq,k) float32, ids (nq,k) uint32)."""
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    scores, ids = index.search(Q, k)
    assert (ids >= 0).all(), "faiss returned -1 sentinel; k likely > index.ntotal"
    return scores.astype(np.float32), ids.astype(np.uint32)


def recall_at_ks(
    retrieved_ids: np.ndarray,
    gt_ids: np.ndarray,
    ks: Tuple[int, ...] = (1, 10, 100),
) -> Dict[int, float]:
    """Recall@k for several k: mean over queries of |ret[:k] ∩ gt[:k]| / min(k, |ret|, |gt|)."""
    nq = retrieved_ids.shape[0]
    out: Dict[int, float] = {}
    for k in ks:
        kk_ret = min(k, retrieved_ids.shape[1])
        kk_gt = min(k, gt_ids.shape[1])
        denom = min(kk_ret, kk_gt)
        if denom == 0:
            out[k] = 0.0
            continue
        total = 0.0
        for i in range(nq):
            gt_set = set(gt_ids[i, :kk_gt].tolist())
            ret_set = set(retrieved_ids[i, :kk_ret].tolist())
            total += len(gt_set & ret_set) / denom
        out[k] = total / nq if nq else 0.0
    return out


def reconstruction_mse(
    X: np.ndarray,
    reconstruct_fn: ReconstructFn,
    sample_ids: np.ndarray,
    chunk: int = 50_000,
) -> float:
    """Mean per-element squared error between X[sample] and its reconstruction."""
    X = np.asarray(X, dtype=np.float32)
    sample_ids = np.asarray(sample_ids, dtype=np.uint32)
    d = X.shape[1]
    sq_err = 0.0
    count = 0
    for start in range(0, sample_ids.size, chunk):
        block = sample_ids[start:start + chunk]
        x_hat = np.asarray(reconstruct_fn(block), dtype=np.float32)
        diff = X[block] - x_hat
        sq_err += float(np.sum(diff * diff))
        count += block.size * d
    return sq_err / count if count else 0.0
