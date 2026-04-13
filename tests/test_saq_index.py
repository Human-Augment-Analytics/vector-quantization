# tests/test_saq_index.py
import pytest
import numpy as np


def make_data(N: int = 512, D: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture
def saq_index():
    pytest.importorskip("saq")  # skip if SAQ wheel not installed
    from haag_vq.methods.search.saq_index import SaqIndex
    return SaqIndex(bpd=4.0, K=16, nprobe=8, num_threads=1)


def test_fit_does_not_raise(saq_index):
    X = make_data()
    saq_index.fit(X, metric='l2')


def test_search_shape(saq_index):
    X = make_data()
    saq_index.fit(X)
    Q = make_data(N=10, seed=99)
    ids = saq_index.search(Q, k=5)
    assert ids.shape == (10, 5)
    assert ids.dtype == np.uint32


def test_search_with_scores_shape(saq_index):
    X = make_data()
    saq_index.fit(X)
    Q = make_data(N=5, seed=42)
    ids, dists = saq_index.search_with_scores(Q, k=3)
    assert ids.shape == (5, 3)
    assert dists.shape == (5, 3)


def test_memory_footprint_positive(saq_index):
    X = make_data()
    saq_index.fit(X)
    assert saq_index.memory_footprint() > 0


def test_reconstruction_mse_returns_none(saq_index):
    """SaqIndex uses construct() path which doesn't support decompress()."""
    X = make_data()
    saq_index.fit(X)
    mse = saq_index.reconstruction_mse(X)
    assert mse is None


def test_save_load_roundtrip(saq_index, tmp_path):
    X = make_data()
    saq_index.fit(X)
    p = tmp_path / "saq_test.idx"
    saq_index.save(p)

    from haag_vq.methods.search.saq_index import SaqIndex
    loaded = SaqIndex(bpd=4.0, K=16, nprobe=8, num_threads=1)
    loaded.load(p)

    Q = make_data(N=5, seed=7)
    ids_orig = saq_index.search(Q, k=3)
    ids_loaded = loaded.search(Q, k=3)
    assert np.array_equal(ids_orig, ids_loaded)
