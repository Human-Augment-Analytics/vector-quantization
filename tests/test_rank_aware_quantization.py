import numpy as np

from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer


def _data(seed=0, n=3000, d=64):
    # anisotropic + correlated so PCA variances span a real range
    rng = np.random.default_rng(seed)
    A = np.diag(np.linspace(4.0, 0.2, d)) @ rng.standard_normal((d, d))
    return (rng.standard_normal((n, d)) @ A).astype(np.float32)


def test_roundtrip_shape_finite():
    X = _data()
    q = RankAwareQuantizer(avg_bits=4, alpha=1.0)
    q.fit(X)
    codes = q.compress(X)
    xhat = q.decompress(codes)
    assert xhat.shape == X.shape and np.isfinite(xhat).all()


def test_row_independent():
    X = _data()
    q = RankAwareQuantizer(avg_bits=4, alpha=1.0); q.fit(X)
    codes = q.compress(X)
    ids = np.array([0, 3, 50, 2999])
    np.testing.assert_allclose(q.decompress(codes[ids]), q.decompress(codes)[ids], rtol=1e-4, atol=1e-4)


def test_budget_respected():
    X = _data(d=64)
    q = RankAwareQuantizer(avg_bits=3, alpha=1.0); q.fit(X)
    assert q.bits.sum() <= round(3 * X.shape[1])
    assert q.bits.sum() >= round(3 * X.shape[1]) - 1  # essentially fully spent


def test_mse_decreases_with_avg_bits():
    X = _data()
    m = {}
    for ab in (2, 4, 6):
        q = RankAwareQuantizer(avg_bits=ab, alpha=1.0); q.fit(X)
        m[ab] = float(np.mean((X - q.decompress(q.compress(X)))**2))
    assert m[6] < m[4] < m[2], m


def test_higher_alpha_concentrates_on_head():
    # KEY: higher alpha must put MORE bits on the high-variance head (low-index dims).
    X = _data()
    D = X.shape[1]
    q0 = RankAwareQuantizer(avg_bits=3, alpha=0.0); q0.fit(X)
    q2 = RankAwareQuantizer(avg_bits=3, alpha=2.0); q2.fit(X)
    head = D // 2
    assert q2.bits[:head].sum() > q0.bits[:head].sum(), (q0.bits[:head].sum(), q2.bits[:head].sum())


def test_alpha0_is_mse_optimal():
    # alpha=0 (pure MSE objective) should give the LOWEST reconstruction MSE.
    X = _data()
    mse = {}
    for a in (0.0, 1.0, 2.0):
        q = RankAwareQuantizer(avg_bits=3, alpha=a); q.fit(X)
        mse[a] = float(np.mean((X - q.decompress(q.compress(X)))**2))
    assert mse[0.0] <= mse[1.0] <= mse[2.0], mse


def test_ffd_roundtrip():
    # FFD packing is lossless: same dequantized reconstruction as dense.
    X = _data()
    qd = RankAwareQuantizer(avg_bits=4, alpha=1.0, packing="dense"); qd.fit(X)
    qf = RankAwareQuantizer(avg_bits=4, alpha=1.0, packing="ffd"); qf.fit(X)
    # same per-dim allocation/codebooks -> reconstruction must be identical.
    np.testing.assert_array_equal(qf.bits, qd.bits)
    xhat_dense = qd.decompress(qd.compress(X))
    xhat_ffd = qf.decompress(qf.compress(X))
    assert xhat_ffd.shape == X.shape and np.isfinite(xhat_ffd).all()
    np.testing.assert_array_equal(xhat_ffd, xhat_dense)


def test_ffd_compression_between_dense_and_naive():
    X = _data()
    D = X.shape[1]
    qd = RankAwareQuantizer(avg_bits=4, alpha=1.0, packing="dense"); qd.fit(X)
    qf = RankAwareQuantizer(avg_bits=4, alpha=1.0, packing="ffd"); qf.fit(X)
    dense_bytes = qd.code_size
    ffd_bytes = qf.code_size
    assert dense_bytes <= ffd_bytes <= D
