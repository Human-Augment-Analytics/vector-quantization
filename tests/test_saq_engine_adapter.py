import numpy as np
import pytest

saq = pytest.importorskip("saq")

from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter


def _data(seed=0, n=4000, d=128):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


@pytest.mark.parametrize("greedy,derive", [(False, False), (True, True)])
def test_saq_engine_fit_reconstruct(greedy, derive):
    X = _data()
    q = SaqEngineAdapter(quant_type="CAQ", avg_bits=4.0, greedy=greedy, derive_codebooks=derive)
    q.fit(X)
    ids = np.array([0, 1, 100, 3999], dtype=np.uint32)
    x_hat = q.reconstruct(ids)
    assert x_hat.shape == (4, X.shape[1])
    assert x_hat.dtype == np.float32
    assert q.code_bytes() > 0


@pytest.mark.parametrize("greedy,derive", [(False, False), (True, True)])
def test_saq_engine_more_bits_lower_mse(greedy, derive):
    X = _data()
    out = {}
    for bpd in (2.0, 8.0):
        q = SaqEngineAdapter(quant_type="CAQ", avg_bits=bpd, greedy=greedy, derive_codebooks=derive)
        q.fit(X)
        xh = q.reconstruct(np.arange(X.shape[0], dtype=np.uint32))
        out[bpd] = float(np.mean((X - xh) ** 2))
    assert out[8.0] < out[2.0], f"not multi-bit: {out}"
