# tests/test_rabitq_index.py
"""Contract tests for RaBitQIndex — mirrors test_flat_quantized.py."""

from __future__ import annotations

import numpy as np
import pytest


def _make_data(N: int = 256, D: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture
def rabitq_index():
    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIndex
    idx = RaBitQIndex()
    idx.fit(_make_data())
    return idx


def test_search_shape(rabitq_index):
    Q = _make_data(N=5, seed=42)
    ids = rabitq_index.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32


def test_search_with_scores_shape(rabitq_index):
    Q = _make_data(N=3, seed=7)
    ids, dists = rabitq_index.search_with_scores(Q, k=4)
    assert ids.shape == (3, 4)
    assert dists.shape == (3, 4)


def test_memory_footprint(rabitq_index):
    assert rabitq_index.memory_footprint() > 0


def test_reconstruction_mse(rabitq_index):
    X = _make_data()
    mse = rabitq_index.reconstruction_mse(X)
    assert mse is not None
    assert np.isfinite(mse) and mse >= 0.0


def test_reconstruction_mse_with_sample_ids(rabitq_index):
    X = _make_data()
    sample_ids = np.arange(10, dtype=np.uint32)
    mse = rabitq_index.reconstruction_mse(X, sample_ids=sample_ids)
    assert mse is not None and np.isfinite(mse) and mse >= 0.0


def test_unfit_search_raises():
    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIndex
    idx = RaBitQIndex()
    with pytest.raises(RuntimeError):
        idx.search(_make_data(N=1), k=1)


def test_memory_footprint_before_fit():
    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIndex
    idx = RaBitQIndex()
    assert idx.memory_footprint() == 0


def test_save_load_roundtrip(tmp_path, rabitq_index):
    # RaBitQIndex now uses faiss.write_index / read_index internally, so
    # save/load should round-trip without loss.
    p = tmp_path / "rabitq.faiss"
    rabitq_index.save(p)

    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIndex
    fresh = RaBitQIndex()
    fresh.load(p)

    Q = _make_data(N=4, seed=1)
    original_ids = rabitq_index.search(Q, k=3)
    loaded_ids = fresh.search(Q, k=3)
    assert np.array_equal(original_ids, loaded_ids)


def test_qb_parameter_preserved_through_save_load(tmp_path):
    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIndex
    idx = RaBitQIndex(qb=8)
    idx.fit(_make_data())
    p = tmp_path / "rabitq_qb8.faiss"
    idx.save(p)

    loaded = RaBitQIndex()
    loaded.load(p)
    assert loaded._qb == 8
