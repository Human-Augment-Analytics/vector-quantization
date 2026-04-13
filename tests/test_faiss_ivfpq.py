# tests/test_faiss_ivfpq.py
import pytest
import numpy as np


def make_data(N: int = 256, D: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture
def faiss_idx():
    pytest.importorskip("faiss")
    from haag_vq.methods.search.faiss_ivfpq_index import FaissIvfPqIndex
    # K=8 cells, m=4 subspaces, nbits=4 (ksub=16), D=16 -> dsub=4 (divisible)
    return FaissIvfPqIndex(K=8, m=4, nbits=4, nprobe=4)


def test_fit_search_shape(faiss_idx):
    X = make_data()
    faiss_idx.fit(X)
    Q = make_data(N=5, seed=42)
    ids = faiss_idx.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32


def test_search_with_scores_shape(faiss_idx):
    X = make_data()
    faiss_idx.fit(X)
    Q = make_data(N=3, seed=7)
    ids, dists = faiss_idx.search_with_scores(Q, k=4)
    assert ids.shape == (3, 4)
    assert dists.shape == (3, 4)


def test_memory_footprint(faiss_idx):
    X = make_data()
    faiss_idx.fit(X)
    assert faiss_idx.memory_footprint() > 0


def test_reconstruction_mse_none(faiss_idx):
    X = make_data()
    faiss_idx.fit(X)
    assert faiss_idx.reconstruction_mse(X) is None


def test_save_load(faiss_idx, tmp_path):
    X = make_data()
    faiss_idx.fit(X)
    p = tmp_path / "faiss.idx"
    faiss_idx.save(str(p))

    from haag_vq.methods.search.faiss_ivfpq_index import FaissIvfPqIndex
    loaded = FaissIvfPqIndex(K=8, m=4, nbits=4, nprobe=4)
    loaded.load(str(p))

    Q = make_data(N=5, seed=1)
    assert np.array_equal(faiss_idx.search(Q, k=3), loaded.search(Q, k=3))


def test_ip_metric():
    pytest.importorskip("faiss")
    from haag_vq.methods.search.faiss_ivfpq_index import FaissIvfPqIndex
    idx = FaissIvfPqIndex(K=8, m=4, nbits=4, nprobe=4)
    X = make_data()
    idx.fit(X, metric='ip')
    Q = make_data(N=3, seed=55)
    ids, scores = idx.search_with_scores(Q, k=4)
    assert ids.shape == (3, 4)
    assert ids.dtype == np.uint32
