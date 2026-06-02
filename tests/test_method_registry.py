import numpy as np
import pytest

from haag_vq.benchmarks.method_registry import (
    largest_divisor_leq,
    build_faiss_quantizer,
    FAISS_METHODS,
)


def test_largest_divisor_leq():
    assert largest_divisor_leq(1536, 1536) == 1536
    assert largest_divisor_leq(1536, 600) == 512   # 1536 = 512*3
    assert largest_divisor_leq(1536, 1) == 1
    assert largest_divisor_leq(30, 7) == 6


@pytest.mark.parametrize("method", ["pq", "opq", "sq"])
@pytest.mark.parametrize("bpd", [1, 2, 4, 8])
def test_build_faiss_quantizer_fits_and_reconstructs(method, bpd):
    D = 48  # divisible by many M
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, D)).astype(np.float32)
    q = build_faiss_quantizer(method, bpd=bpd, D=D)
    q.fit(X)
    x_hat = q.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
    assert x_hat.shape == X.shape
    assert q.code_bytes() > 0


def test_faiss_methods_constant():
    assert set(FAISS_METHODS) == {"pq", "opq", "sq"}
