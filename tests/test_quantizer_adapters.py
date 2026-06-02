import numpy as np
import pytest

from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer


def _data(seed=0, n=512, d=32):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_faiss_adapter_fit_reconstruct_shape_and_ids():
    X = _data()
    adapter = FaissQuantizerAdapter(ScalarQuantizer(num_bits=8))
    adapter.fit(X)
    ids = np.array([0, 5, 10, 511], dtype=np.uint32)
    x_hat = adapter.reconstruct(ids)
    assert x_hat.shape == (4, X.shape[1])
    assert x_hat.dtype == np.float32
    full = adapter.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
    np.testing.assert_allclose(x_hat, full[ids], rtol=1e-5)


def test_faiss_adapter_code_bytes_includes_norm_sidechannel():
    X = _data(d=32)
    pq = ProductQuantizer(M=8, B=8)  # 8 bytes/vector codes
    adapter = FaissQuantizerAdapter(pq)
    adapter.fit(X)
    n = X.shape[0]
    assert adapter.code_bytes() == n * 8 + n * 4


def test_faiss_adapter_sq8_reconstruction_is_close():
    X = _data()
    adapter = FaissQuantizerAdapter(ScalarQuantizer(num_bits=8))
    adapter.fit(X)
    x_hat = adapter.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
    mse = float(np.mean((X - x_hat) ** 2))
    assert mse < 0.01
