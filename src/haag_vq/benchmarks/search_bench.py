# src/haag_vq/benchmarks/search_bench.py
"""Unified benchmark harness for BaseSearchIndex implementations.

Primary metrics : recall@k, QPS, memory_footprint
Secondary metric: reconstruction MSE
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from haag_vq.methods.base_search_index import BaseSearchIndex


def compute_ground_truth(
    X_train: np.ndarray,
    X_query: np.ndarray,
    k: int = 10,
    metric: str = 'l2',
) -> np.ndarray:
    """Brute-force ground truth using faiss.IndexFlatL2 or IndexFlatIP.

    Args:
        X_train: (N, D) float32 database vectors.
        X_query: (nq, D) float32 query vectors.
        k:       Number of true neighbors to return.
        metric:  'l2' or 'ip'.

    Returns:
        (nq, k) int64 array of true neighbor IDs (sorted by distance).
    """
    import faiss

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_query = np.ascontiguousarray(X_query, dtype=np.float32)
    N, D = X_train.shape
    k = min(k, N)

    if metric == 'ip':
        index = faiss.IndexFlatIP(D)
    else:
        index = faiss.IndexFlatL2(D)

    index.add(X_train)
    _, ids = index.search(X_query, k)
    return ids.astype(np.int64)


def _compute_recall(ids: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Recall@k: fraction of queries that contain at least one true top-k neighbor.

    Args:
        ids:          (nq, k_returned) — returned neighbor IDs.
        ground_truth: (nq, k_gt) — true neighbor IDs (sorted by distance).
        k:            recall cutoff.
    """
    nq = ids.shape[0]
    k_gt = min(k, ground_truth.shape[1])
    k_ret = min(k, ids.shape[1])
    hits = 0
    for i in range(nq):
        gt_set = set(ground_truth[i, :k_gt].tolist())
        ret_set = set(ids[i, :k_ret].tolist())
        if gt_set & ret_set:
            hits += 1
    return hits / nq


def benchmark_index(
    index: BaseSearchIndex,
    X_train: np.ndarray,
    X_query: np.ndarray,
    gt_ids: np.ndarray,
    k: int = 10,
    repeats: int = 3,
    mse_sample: int = 1000,
) -> dict:
    """Fit the index on X_train, search X_query, compute benchmark metrics.

    Args:
        index:    An unfitted BaseSearchIndex instance.
        X_train:  (N, D) float32 training/database vectors.
        X_query:  (nq, D) float32 query vectors.
        gt_ids:   (nq, k_gt) ground-truth neighbor IDs (from compute_ground_truth).
        k:        Number of neighbors for recall and search.
        repeats:  Number of search timing repetitions; best time is used.
        mse_sample: Max number of vectors sampled for reconstruction MSE.

    Returns:
        dict with keys:
            method, recall_at_k, qps, memory_bytes, compression_ratio,
            mse, k, N, D
    """
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_query = np.ascontiguousarray(X_query, dtype=np.float32)
    N, D = X_train.shape
    nq = X_query.shape[0]

    # Fit
    index.fit(X_train)

    # Search timing — use best-of-repeats to reduce noise
    ids = None
    best_time = float('inf')
    for _ in range(repeats):
        t0 = time.perf_counter()
        ids = index.search(X_query, k)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if elapsed < best_time:
            best_time = elapsed

    qps = nq / best_time if best_time > 0 else float('inf')

    # Recall@k
    recall = _compute_recall(ids, gt_ids, k)

    # Memory and compression ratio
    mem_bytes = index.memory_footprint()
    float32_baseline = N * D * 4  # bytes if stored as raw float32
    compression_ratio = float32_baseline / mem_bytes if mem_bytes > 0 else float('inf')

    # Reconstruction MSE (optional)
    n_total = N
    if n_total > mse_sample:
        rng = np.random.default_rng(0)
        sample_ids = rng.choice(n_total, mse_sample, replace=False).astype(np.uint32)
    else:
        sample_ids = np.arange(n_total, dtype=np.uint32)

    mse = index.reconstruction_mse(X_train, sample_ids=sample_ids)

    return {
        'method': type(index).__name__,
        'recall_at_k': recall,
        'qps': qps,
        'memory_bytes': mem_bytes,
        'compression_ratio': compression_ratio,
        'mse': mse,
        'k': k,
        'N': N,
        'D': D,
    }


def sweep_bpd(
    index_factory: Callable[[float], BaseSearchIndex],
    bpd_values: list[float],
    X_train: np.ndarray,
    X_query: np.ndarray,
    gt_ids: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """Sweep bits-per-dimension values and benchmark each resulting index.

    Args:
        index_factory: Callable that takes a bpd float and returns a
                       (unfitted) BaseSearchIndex instance.
        bpd_values:    List of bpd values to sweep over.
        X_train:       (N, D) training/database vectors.
        X_query:       (nq, D) query vectors.
        gt_ids:        (nq, k_gt) ground-truth neighbor IDs.
        k:             Number of neighbors.

    Returns:
        DataFrame with one row per bpd value.
    """
    rows = []
    for bpd in bpd_values:
        index = index_factory(bpd)
        result = benchmark_index(index, X_train, X_query, gt_ids, k=k)
        result['bpd'] = bpd
        rows.append(result)
    return pd.DataFrame(rows)


def compare_methods(
    method_configs: dict[str, BaseSearchIndex],
    X_train: np.ndarray,
    X_query: np.ndarray,
    gt_ids: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """Benchmark each method and return a combined DataFrame.

    Args:
        method_configs: Dict mapping method_name -> (unfitted) BaseSearchIndex.
        X_train:        (N, D) training/database vectors.
        X_query:        (nq, D) query vectors.
        gt_ids:         (nq, k_gt) ground-truth neighbor IDs.
        k:              Number of neighbors.

    Returns:
        DataFrame with one row per method.
    """
    rows = []
    for name, index in method_configs.items():
        result = benchmark_index(index, X_train, X_query, gt_ids, k=k)
        result['method'] = name
        rows.append(result)
    return pd.DataFrame(rows)


def pareto_plot(
    df: pd.DataFrame,
    x: str = 'compression_ratio',
    y: str = 'recall_at_k',
    hue: str = 'method',
    save_path: Optional[str] = None,
) -> None:
    """Scatter/line plot — one series per method on the recall vs compression Pareto front.

    Args:
        df:        DataFrame as returned by compare_methods or sweep_bpd.
        x:         Column name for the x-axis (default: compression_ratio).
        y:         Column name for the y-axis (default: recall_at_k).
        hue:       Column used to colour/label series (default: method).
        save_path: If given, save figure to this path instead of calling plt.show().
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, group in df.groupby(hue):
        ax.scatter(group[x], group[y], label=str(method_name), s=80)
        ax.plot(group[x], group[y], linewidth=1, alpha=0.6)
        for _, row in group.iterrows():
            ax.annotate(
                str(method_name),
                (row[x], row[y]),
                textcoords='offset points',
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'Pareto: {y} vs {x}')
    ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
