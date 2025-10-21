import numpy as np

from haag_vq.methods.saq import SAQ


def _rng_data(n=64, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d), dtype=np.float32)


def test_saq_roundtrip_no_budget():
    X = _rng_data(n=64, d=16, seed=42)
    q = SAQ(num_bits=4)
    q.fit(X)

    # Model learned attributes
    assert q.fitted is True
    assert q.dim == X.shape[1]
    assert q.bits_per_dim_ is not None
    assert q.segments_ is not None
    assert q.bits_per_dim_.shape == (X.shape[1],)
    assert np.all(q.bits_per_dim_ == 4)

    # Compress/Decompress on the same batch
    codes = q.compress(X)
    assert codes.shape == X.shape
    assert codes.dtype == np.uint8

    X_hat = q.decompress(codes)
    assert X_hat.shape == X.shape
    assert np.isfinite(X_hat).all()

    # Basic reconstruction sanity: not identical but finite error
    mse = np.mean((X_hat - X) ** 2)
    assert mse > 0.0


def test_saq_error_decreases_with_more_bits():
    X = _rng_data(n=128, d=32, seed=1)

    q2 = SAQ(num_bits=2)
    q2.fit(X)
    mse2 = np.mean((q2.decompress(q2.compress(X)) - X) ** 2)

    q8 = SAQ(num_bits=8)
    q8.fit(X)
    mse8 = np.mean((q8.decompress(q8.compress(X)) - X) ** 2)

    # More bits should not increase reconstruction error
    assert mse8 <= mse2


def test_saq_with_bit_budget_allocates_reasonably():
    X = _rng_data(n=100, d=24, seed=7)
    D = X.shape[1]
    budget_bits = D * 2  # aim ~2 bits/dim on average
    allowed = [0, 2, 4, 6, 8]

    q = SAQ(total_bits=budget_bits, allowed_bits=allowed)
    q.fit(X)

    # Bits per dim should only use allowed bitwidths
    bits = q.bits_per_dim_
    assert bits is not None
    assert set(np.unique(bits)).issubset(set(allowed))

    # Sum of assigned bits should be less than unconstrained 8 bits/dim
    assert bits.sum() < D * 8

    # Codes/decompress round trip works
    codes = q.compress(X)
    X_hat = q.decompress(codes)
    assert codes.shape == X.shape
    assert X_hat.shape == X.shape
    assert np.isfinite(X_hat).all()


def test_compress_with_info_matches_internal_decompress():
    X = _rng_data(n=32, d=12, seed=11)
    q = SAQ(num_bits=6)
    q.fit(X)

    codes1 = q.compress(X)
    recon1 = q.decompress(codes1)

    codes2, info = q.compress_with_info(X, return_ip_hint=True)
    recon2 = q.decompress_with_info(codes2, info)

    # Outputs should be consistent and the hint present
    assert np.allclose(recon1, recon2, atol=1e-5)
    assert "o_max" in info and "x_norm" in info and "o_dot_ohat" in info
    assert info["o_max"].shape == (X.shape[0],)
    assert info["o_dot_ohat"].shape == (X.shape[0],)


def test_api_errors_before_compress_and_shape_mismatch():
    X = _rng_data(n=8, d=6, seed=5)
    q = SAQ(num_bits=3)
    q.fit(X)

    # Decompress without a preceding compress should fail (missing o_max)
    import pytest

    with pytest.raises(RuntimeError):
        q.decompress(np.zeros_like(X, dtype=np.uint8))

    # After a compress, shape mismatches should raise
    _ = q.compress(X)
    with pytest.raises(ValueError):
        q.compress(X[:, :4])

