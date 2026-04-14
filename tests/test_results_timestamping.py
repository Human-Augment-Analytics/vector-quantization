# tests/test_results_timestamping.py
"""Verify timestamp column in DataFrame results + timestamped output filename."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


ISO_8601_UTC = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def _synthetic(N: int = 64, D: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def test_compare_methods_adds_timestamp_column():
    pytest.importorskip("faiss")
    from haag_vq.benchmarks.search_bench import compare_methods, compute_ground_truth
    from haag_vq.methods.search import FlatQuantizedIndex
    from haag_vq.methods.scalar_quantization import ScalarQuantizer

    X = _synthetic(N=64, D=8)
    Q = _synthetic(N=4, D=8, seed=1)
    gt = compute_ground_truth(X, Q, k=3)

    configs = {'sq_flat': FlatQuantizedIndex(ScalarQuantizer(num_bits=8))}
    df = compare_methods(configs, X, Q, gt, k=3)

    assert 'timestamp' in df.columns
    assert df['timestamp'].notna().all()
    for ts in df['timestamp']:
        assert ISO_8601_UTC.match(ts), f"Not an ISO-8601 UTC stamp: {ts!r}"


def test_sweep_bpd_adds_timestamp_column():
    pytest.importorskip("faiss")
    from haag_vq.benchmarks.search_bench import sweep_bpd, compute_ground_truth
    from haag_vq.methods.search import FlatQuantizedIndex
    from haag_vq.methods.scalar_quantization import ScalarQuantizer

    X = _synthetic(N=64, D=8)
    Q = _synthetic(N=4, D=8, seed=1)
    gt = compute_ground_truth(X, Q, k=3)

    def factory(_bpd):
        return FlatQuantizedIndex(ScalarQuantizer(num_bits=8))

    df = sweep_bpd(factory, [4.0, 8.0], X, Q, gt, k=3)

    assert 'timestamp' in df.columns
    assert df['timestamp'].notna().all()
    # A single sweep call stamps all rows with the same instant.
    assert df['timestamp'].nunique() == 1


def test_timestamped_output_path_appends_utc_stamp():
    from haag_vq.benchmarks.run_benchmarks import timestamped_output_path

    fixed = datetime(2026, 4, 13, 14, 22, 1, tzinfo=timezone.utc)
    p = timestamped_output_path(Path('results/msmarco_100k.csv'), now=fixed)

    assert p == Path('results/msmarco_100k_20260413_142201.csv')


def test_timestamped_output_path_preserves_multisuffix_stem():
    from haag_vq.benchmarks.run_benchmarks import timestamped_output_path

    fixed = datetime(2026, 4, 13, 14, 22, 1, tzinfo=timezone.utc)
    # Only the final suffix is stripped by Path.stem; a "name.bpd4.csv" becomes
    # "name.bpd4" + "_YYYYMMDD_HHMMSS" + ".csv" — documented behaviour.
    p = timestamped_output_path(Path('out/name.bpd4.csv'), now=fixed)

    assert p == Path('out/name.bpd4_20260413_142201.csv')


def test_timestamped_output_path_default_now_is_recent():
    from haag_vq.benchmarks.run_benchmarks import timestamped_output_path

    p = timestamped_output_path(Path('x.csv'))
    m = re.match(r'x_(\d{8})_(\d{6})\.csv$', p.name)
    assert m is not None, f"Unexpected filename: {p.name}"
    # Stamp is UTC; parse it and confirm it's within a 60-second window of now.
    stamp = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').replace(
        tzinfo=timezone.utc
    )
    delta = abs((datetime.now(timezone.utc) - stamp).total_seconds())
    assert delta < 60, f"Stamp {stamp} is {delta}s from now"
