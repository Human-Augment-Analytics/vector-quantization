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


def test_all_methods_and_unified_dispatch():
    from haag_vq.benchmarks.method_registry import ALL_METHODS, build_quantizer
    assert set(ALL_METHODS) == {"pq", "opq", "sq", "saq_paper", "ours", "rabitq", "lvq"}
    # faiss dispatch works without the saq wheel
    q = build_quantizer("pq", bpd=4, D=48)
    assert q is not None
    # numpy saq-study methods dispatch and build
    import numpy as np
    q2 = build_quantizer("rabitq", bpd=4, D=48)
    q2.fit(np.random.default_rng(0).standard_normal((200, 48)).astype("float32"))
    assert q2.code_bytes() > 0


def test_build_quantizer_unknown_raises():
    import pytest
    from haag_vq.benchmarks.method_registry import build_quantizer
    with pytest.raises(ValueError):
        build_quantizer("bogus", bpd=4, D=48)
