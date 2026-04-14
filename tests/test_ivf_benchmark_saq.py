# tests/test_ivf_benchmark_saq.py
"""Integration: ivf_benchmark's SAQ runner hits the C++ SaqIndex path."""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic(N: int, D: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def test_ivf_benchmark_saq_runs_via_cpp_engine():
    """METHOD_RUNNERS['saq'] must produce non-trivial metrics using SaqIndex.

    Guards against a regression where ``_run_saq`` silently falls back to a
    Python scaffold with bogus memory / recall numbers. In that historical
    bug the compression ratio was hard-coded at 4× and recall at bpd=2 was
    ~0.04 on normalised embeddings; here we assert a lower bound on compression
    reflecting the real N*D*bpd/8 code budget so that any regression back to
    a per-dim byte layout would fail.
    """
    pytest.importorskip("faiss")
    pytest.importorskip("saq")

    from haag_vq.benchmarks.ivf_benchmark import METHOD_RUNNERS, _compute_ground_truth

    N, D, bpd = 2000, 64, 4
    X = _synthetic(N, D)
    Q = _synthetic(20, D, seed=1)
    gt = _compute_ground_truth(X, Q, k=10)

    runner = METHOD_RUNNERS["saq"]
    metrics = runner(X, Q, gt, k=5, bpd=bpd, K=64, nprobe=16)

    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["qps"] > 0

    # Exact memory-footprint check: detects both the old D-factor bug
    # (would give ~17 KB here) and the placeholder's 1-byte-per-dim layout
    # (would give ~128 KB here). The correct value is codes + centroids.
    K_actual = min(64, N // 10)  # SaqIndex clamps K in fit()
    expected_code_bytes = N * D * bpd // 8
    expected_centroid_bytes = K_actual * D * 4
    expected_total = expected_code_bytes + expected_centroid_bytes
    assert metrics["memory_bytes"] == expected_total, (
        f"memory_bytes={metrics['memory_bytes']} expected {expected_total} "
        f"(code={expected_code_bytes} + centroid={expected_centroid_bytes})"
    )

    # SAQ via ivf_benchmark now matches run_benchmarks: mse is left blank.
    assert metrics["mse"] == ""
