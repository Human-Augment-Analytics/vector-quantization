import numpy as np
import pytest

from haag_vq.methods.extended_rabitq import ExtendedRaBitQuantizer


def _data(seed=0, n=2000, d=64):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_fit_compress_decompress_roundtrip_shape():
    X = _data()
    q = ExtendedRaBitQuantizer(num_bits=4)
    q.fit(X)
    codes = q.compress(X)
    assert codes.shape[0] == X.shape[0]
    xhat = q.decompress(codes)
    assert xhat.shape == X.shape
    assert np.isfinite(xhat).all()


def test_decompress_is_row_independent():
    # CRITICAL: the benchmark slices codes[ids]; decompress must be self-contained.
    X = _data()
    q = ExtendedRaBitQuantizer(num_bits=4)
    q.fit(X)
    codes = q.compress(X)
    ids = np.array([0, 7, 100, 1999])
    full = q.decompress(codes)
    sliced = q.decompress(codes[ids])
    np.testing.assert_allclose(sliced, full[ids], rtol=1e-5, atol=1e-5)


def test_mse_decreases_with_bits():
    X = _data()
    out = {}
    for b in (2, 4, 8):
        q = ExtendedRaBitQuantizer(num_bits=b)
        q.fit(X)
        xhat = q.decompress(q.compress(X))
        out[b] = float(np.mean((X - xhat) ** 2))
    assert out[8] < out[4] < out[2], out


def test_reconstruction_beats_trivial():
    X = _data()
    q = ExtendedRaBitQuantizer(num_bits=4)
    q.fit(X)
    xhat = q.decompress(q.compress(X))
    mse = float(np.mean((X - xhat) ** 2))
    assert mse < float(np.var(X))  # does meaningfully better than predicting the mean


def test_code_size_grows_with_bits():
    X = _data()
    s = {}
    for b in (2, 8):
        q = ExtendedRaBitQuantizer(num_bits=b)
        q.fit(X)
        s[b] = q.compress(X).shape[1]
    assert s[8] > s[2]
