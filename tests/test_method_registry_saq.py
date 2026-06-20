import numpy as np
import pytest

from haag_vq.benchmarks.method_registry_saq import build_saq_quantizer, SAQ_METHODS


def _data(seed=0, n=2000, d=64):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_saq_methods_set():
    assert set(SAQ_METHODS) == {"saq_paper", "ours", "rabitq", "lvq", "rankaware", "perdim_mse"}


@pytest.mark.parametrize("method", ["rabitq", "lvq"])
def test_numpy_methods_build_fit_reconstruct(method):
    X = _data()
    q = build_saq_quantizer(method, bpd=4, D=X.shape[1])
    q.fit(X)
    xh = q.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
    assert xh.shape == X.shape
    assert q.code_bytes() > 0


@pytest.mark.parametrize("method", ["saq_paper", "ours"])
def test_engine_methods_build_fit_reconstruct(method):
    pytest.importorskip("saq")
    X = _data(n=4000, d=128)
    q = build_saq_quantizer(method, bpd=4, D=X.shape[1])
    q.fit(X)
    xh = q.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
    assert xh.shape == X.shape
    assert q.code_bytes() > 0


def test_unknown_raises():
    with pytest.raises(ValueError):
        build_saq_quantizer("nope", bpd=4, D=64)


@pytest.mark.parametrize("method", ["rankaware", "perdim_mse"])
def test_perdim_numpy_methods_build_fit_reconstruct(method):
    X = _data(n=2000, d=64)
    q = build_saq_quantizer(method, bpd=4, D=X.shape[1])
    q.fit(X)
    xh = q.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
    assert xh.shape == X.shape
    assert q.code_bytes() > 0
