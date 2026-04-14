# tests/test_rabitq_ivf_index.py
"""Contract tests for RaBitQIVFIndex (faiss.IndexIVFRaBitQ wrapper)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_data(N: int = 800, D: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


@pytest.fixture
def rabitq_ivf_index():
    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIVFIndex
    idx = RaBitQIVFIndex(nlist=16, nprobe=8, qb=4)
    idx.fit(_make_data())
    return idx


def test_search_shape(rabitq_ivf_index):
    Q = _make_data(N=5, seed=42)
    ids = rabitq_ivf_index.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32


def test_memory_footprint_nonzero(rabitq_ivf_index):
    assert rabitq_ivf_index.memory_footprint() > 0


def test_reconstruction_mse_is_none(rabitq_ivf_index):
    # IVF+RaBitQ intentionally returns None (see class docstring).
    X = _make_data()
    assert rabitq_ivf_index.reconstruction_mse(X) is None


def test_unfit_search_raises():
    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIVFIndex
    idx = RaBitQIVFIndex()
    with pytest.raises(RuntimeError):
        idx.search(_make_data(N=1), k=1)


def test_save_load_roundtrip(tmp_path, rabitq_ivf_index):
    p = tmp_path / "rabitq_ivf.faiss"
    rabitq_ivf_index.save(p)

    pytest.importorskip("faiss")
    from haag_vq.methods.search import RaBitQIVFIndex
    fresh = RaBitQIVFIndex()
    fresh.load(p)

    Q = _make_data(N=4, seed=1)
    assert np.array_equal(
        rabitq_ivf_index.search(Q, k=3),
        fresh.search(Q, k=3),
    )
