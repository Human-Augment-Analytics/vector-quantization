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


def compute_exact_norms(X: np.ndarray, eps: float = 1e-12,
                        chunk: int = 1_000_000) -> np.ndarray:
    """Per-vector exact L2 norm, floored at eps to avoid division by zero.
    Chunked so a memmapped X never materializes a full-corpus temporary;
    per-row norms are bit-identical to the unchunked computation."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        blk = np.asarray(X[s:s + chunk], dtype=np.float32)
        out[s:s + blk.shape[0]] = np.linalg.norm(blk, axis=1)
    return np.maximum(out, eps)


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


def stream_search_scaled_ip(
    reconstruct_fn: ReconstructFn,
    n: int,
    d: int,
    norms: np.ndarray,
    Q: np.ndarray,
    k: int,
    chunk: int = 50_000,
    q_block: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Streaming exact top-k under q . (x_hat/||x||) — no full-corpus index.

    Replaces build_scaled_ip_index + search_index for large n (a faiss FlatIP
    over 53M x 1024 reconstructions is ~217 GB). Iterates DB chunks, scores
    against query blocks, and maintains a running top-k per query. Peak memory
    is O(chunk*d + q_block*chunk) instead of O(n*d).

    Final ordering is deterministic: score desc, id asc on exact score ties
    (faiss tie order is heap-dependent; normalize both sides by (-score, id)
    when comparing outputs). Returns (scores (nq,k) f32, ids (nq,k) uint32).
    """
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    nq = Q.shape[0]
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, k), dtype=np.int64)
    inv = (1.0 / np.asarray(norms, dtype=np.float32))
    for ids in _chunks(n, chunk):
        xc = np.ascontiguousarray(reconstruct_fn(ids), dtype=np.float32)
        xc *= inv[ids][:, None]
        c = xc.shape[0]
        kk = min(k, c)
        cid = ids.astype(np.int64)
        for qs in range(0, nq, q_block):
            qe = min(qs + q_block, nq)
            S = Q[qs:qe] @ xc.T                                   # (b, c)
            rows = np.arange(qe - qs)[:, None]
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk] if kk < c \
                else np.broadcast_to(np.arange(c), (qe - qs, c))
            ms = np.concatenate([best_s[qs:qe], S[rows, part]], axis=1)
            mi = np.concatenate([best_i[qs:qe], cid[part]], axis=1)
            keep = np.argpartition(-ms, k - 1, axis=1)[:, :k]
            best_s[qs:qe] = ms[rows, keep]
            best_i[qs:qe] = mi[rows, keep]
    order = np.lexsort((best_i, -best_s), axis=1)
    rows = np.arange(nq)[:, None]
    return best_s[rows, order], best_i[rows, order].astype(np.uint32)


def normalized_ground_truth(
    X: np.ndarray,
    Q: np.ndarray,
    k: int,
    norms: np.ndarray | None = None,
    chunk: int = 50_000,
) -> np.ndarray:
    """Exact top-k under q . (x/||x||). Returns (nq, k) uint32 IDs.
    stream=True (or env VQ_STREAM_EVAL=1) avoids the O(n*d) faiss index."""
    import os
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    n, d = X.shape
    if norms is None:
        norms = compute_exact_norms(X)
    if os.environ.get("VQ_STREAM_EVAL") == "1":
        _, ids = stream_search_scaled_ip(lambda i: np.asarray(X[i], dtype=np.float32),
                                         n, d, norms, Q, k, chunk=chunk)
        return ids
    X = np.asarray(X, dtype=np.float32)
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
