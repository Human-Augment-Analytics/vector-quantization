"""Performance metrics around compression latency and query throughput."""

from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from haag_vq.methods.base_quantizer import BaseQuantizer
from haag_vq.methods.rabit_quantization import RaBitQuantizer
from haag_vq.methods.saq import SAQ
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
    """Measure query throughput (QPS).

    For PQ/SQ/OPQ-like models, this runs :func:`query_codebook` (nearest-centroid
    lookup against the exported codebook). For :class:`RaBitQuantizer`, which does
    not expose a static codebook, this measures the throughput of ``model.compress``
    (code assignment) as a proxy for query-time performance.
    """
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.size == 0:
        raise ValueError("No queries provided for QPS measurement")

    timed_runs = max(1, repeats)
    durations: List[float] = []
    # Choose the runner: codebook search for most models, code assignment for RaBitQ
    if isinstance(model, (RaBitQuantizer, SAQ)):
        def _run_once():
            # Use compress as the closest analog to query-time work for RaBitQ
            model.compress(queries)
    else:
        def _run_once():
            query_codebook(
                queries,
                model=model,
                codebook_vectors=codebook_vectors,
                codebook_path=codebook_path,
                topk=topk,
            )

    for _ in range(timed_runs):
        start = perf_counter()
        _run_once()
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
