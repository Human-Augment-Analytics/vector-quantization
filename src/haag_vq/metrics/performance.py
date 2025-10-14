"""Performance metrics around compression latency and query throughput."""

from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from haag_vq.methods.base_quantizer import BaseQuantizer
from haag_vq.utils.faiss_export import query_codebook


def time_compress(model: BaseQuantizer, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compress ``X`` and return the codes with elapsed time in seconds."""
    start = perf_counter()
    codes = model.compress(X)
    duration = perf_counter() - start
    return codes, float(duration)


def time_decompress(model: BaseQuantizer, codes: np.ndarray) -> Tuple[np.ndarray, float]:
    """Decompress ``codes`` and return reconstructions with elapsed seconds."""
    start = perf_counter()
    reconstructed = model.decompress(codes)
    duration = perf_counter() - start
    return reconstructed, float(duration)


def measure_qps(
    queries: np.ndarray,
    *,
    model: Optional[BaseQuantizer] = None,
    codebook_vectors: Optional[np.ndarray] = None,
    codebook_path: Optional[str] = None,
    repeats: int = 3,
    topk: int = 1,
) -> Dict[str, float]:
    """Measure query throughput (QPS) by calling :func:`query_codebook` repeatedly."""
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.size == 0:
        raise ValueError("No queries provided for QPS measurement")

    timed_runs = max(1, repeats)
    durations: List[float] = []
    for _ in range(timed_runs):
        start = perf_counter()
        query_codebook(
            queries,
            model=model,
            codebook_vectors=codebook_vectors,
            codebook_path=codebook_path,
            topk=topk,
        )
        durations.append(perf_counter() - start)

    durations = [max(d, 1e-12) for d in durations]
    total_queries = float(len(queries))
    qps_values = [total_queries / d for d in durations]
    latencies = [d / total_queries * 1000.0 for d in durations]

    return {
        "qps": float(np.mean(qps_values)),
        "qps_std": float(np.std(qps_values, ddof=0)),
        "avg_query_latency_ms": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies, ddof=0)),
    }
