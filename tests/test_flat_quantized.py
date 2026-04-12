# tests/test_flat_quantized.py
import pytest
import numpy as np


def make_data(N: int = 256, D: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture
def flat_pq():
    pytest.importorskip("faiss")
    from haag_vq.methods.product_quantization import ProductQuantizer
    from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
    # M=4 subquantizers, B=4 bits (ksub=16), D=16 -> dsub=4 (divisible)
    return FlatQuantizedIndex(ProductQuantizer(M=4, B=4))


@pytest.fixture
def flat_sq():
    from haag_vq.methods.scalar_quantization import ScalarQuantizer
    from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
    return FlatQuantizedIndex(ScalarQuantizer(num_bits=8))


def test_fit_search_shape(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    Q = make_data(N=5, seed=42)
    ids = flat_pq.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32


def test_search_with_scores_shape(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    Q = make_data(N=3, seed=7)
    ids, dists = flat_pq.search_with_scores(Q, k=4)
    assert ids.shape == (3, 4)
    assert dists.shape == (3, 4)


def test_memory_footprint(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    assert flat_pq.memory_footprint() > 0


def test_reconstruction_mse(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    mse = flat_pq.reconstruction_mse(X)
    assert mse is not None
    assert np.isfinite(mse) and mse >= 0.0


def test_reconstruction_mse_with_sample_ids(flat_pq):
    X = make_data()
    flat_pq.fit(X)
    sample_ids = np.arange(10, dtype=np.uint32)
    mse = flat_pq.reconstruction_mse(X, sample_ids=sample_ids)
    assert mse is not None and np.isfinite(mse) and mse >= 0.0


def test_save_load(flat_pq, tmp_path):
    X = make_data()
    flat_pq.fit(X)
    p = tmp_path / "flat.pkl"
    flat_pq.save(p)

    from haag_vq.methods.product_quantization import ProductQuantizer
    from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex
    loaded = FlatQuantizedIndex(ProductQuantizer(M=4, B=4))
    loaded.load(p)

    Q = make_data(N=5, seed=1)
    assert np.array_equal(flat_pq.search(Q, k=3), loaded.search(Q, k=3))


def test_ip_metric(flat_sq):
    X = make_data()
    flat_sq.fit(X, metric='ip')
    Q = make_data(N=4, seed=99)
    ids, scores = flat_sq.search_with_scores(Q, k=5)
    assert ids.shape == (4, 5)
    assert ids.dtype == np.uint32
    # Inner-product scores: each row should be non-increasing
    for row in scores:
        assert np.all(row[:-1] >= row[1:] - 1e-5), "IP scores not descending"
