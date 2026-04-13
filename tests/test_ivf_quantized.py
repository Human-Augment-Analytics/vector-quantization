# tests/test_ivf_quantized.py
import pytest
import numpy as np


def make_data(N: int = 256, D: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture
def ivf_pq():
    pytest.importorskip("faiss")
    from haag_vq.methods.product_quantization import ProductQuantizer
    from haag_vq.methods.search.ivf_quantized_index import IvfQuantizedIndex
    # K=8 clusters, nprobe=4, inner PQ: M=4 subquantizers, B=3 bits (ksub=8)
    # D=16, M=4 -> dsub=4 (divisible)
    return IvfQuantizedIndex(
        quantizer_factory=lambda: ProductQuantizer(M=4, B=3),
        K=8,
        nprobe=4,
    )


@pytest.fixture
def ivf_sq():
    pytest.importorskip("faiss")
    from haag_vq.methods.scalar_quantization import ScalarQuantizer
    from haag_vq.methods.search.ivf_quantized_index import IvfQuantizedIndex
    return IvfQuantizedIndex(
        quantizer_factory=lambda: ScalarQuantizer(num_bits=8),
        K=8,
        nprobe=4,
    )


def test_fit_search_shape(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    Q = make_data(N=5, seed=42)
    ids = ivf_pq.search(Q, k=4)
    assert ids.shape == (5, 4)
    assert ids.dtype == np.uint32


def test_search_with_scores_shape(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    Q = make_data(N=3)
    ids, dists = ivf_pq.search_with_scores(Q, k=3)
    assert ids.shape == (3, 3)
    assert dists.shape == (3, 3)


def test_memory_footprint(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    assert ivf_pq.memory_footprint() > 0


def test_reconstruction_mse(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    mse = ivf_pq.reconstruction_mse(X, sample_ids=np.arange(10, dtype=np.uint32))
    assert mse is not None and np.isfinite(mse) and mse >= 0.0


def test_reconstruction_mse_all(ivf_pq):
    X = make_data()
    ivf_pq.fit(X)
    mse = ivf_pq.reconstruction_mse(X)
    assert mse is not None and np.isfinite(mse) and mse >= 0.0


def test_save_load(ivf_pq, tmp_path):
    X = make_data()
    ivf_pq.fit(X)
    p = tmp_path / "ivf.pkl"
    ivf_pq.save(p)

    from haag_vq.methods.product_quantization import ProductQuantizer
    from haag_vq.methods.search.ivf_quantized_index import IvfQuantizedIndex
    loaded = IvfQuantizedIndex(
        quantizer_factory=lambda: ProductQuantizer(M=4, B=3),
        K=8,
        nprobe=4,
    )
    loaded.load(p)

    Q = make_data(N=5, seed=1)
    assert np.array_equal(ivf_pq.search(Q, k=3), loaded.search(Q, k=3))


def test_ip_metric(ivf_sq):
    X = make_data()
    ivf_sq.fit(X, metric='ip')
    Q = make_data(N=3, seed=77)
    ids, dists = ivf_sq.search_with_scores(Q, k=4)
    assert ids.shape == (3, 4)
    assert ids.dtype == np.uint32
