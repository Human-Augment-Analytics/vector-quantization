import numpy as np
import pytest

from haag_vq.benchmarks.exact_search import (
    compute_exact_norms,
    normalized_ground_truth,
    build_scaled_ip_index,
    search_index,
    recall_at_ks,
    reconstruction_mse,
)


def _toy_data(seed=0, n=200, d=16, nq=30):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Q = rng.standard_normal((nq, d)).astype(np.float32)
    return X, Q


def test_compute_exact_norms_matches_linalg():
    X, _ = _toy_data()
    norms = compute_exact_norms(X)
    assert norms.shape == (X.shape[0],)
    assert norms.dtype == np.float32
    np.testing.assert_allclose(norms, np.linalg.norm(X, axis=1), rtol=1e-5)


def test_compute_exact_norms_guards_zero():
    X = np.zeros((3, 4), dtype=np.float32)
    norms = compute_exact_norms(X)
    assert np.all(norms > 0)


def test_normalized_gt_equals_bruteforce_cosine():
    X, Q = _toy_data()
    norms = compute_exact_norms(X)
    ref_scores = Q @ (X / norms[:, None]).T
    ref_top = np.argsort(-ref_scores, axis=1)[:, :10]
    gt = normalized_ground_truth(X, Q, k=10)
    assert np.array_equal(gt[:, 0], ref_top[:, 0])
    for i in range(Q.shape[0]):
        assert set(gt[i].tolist()) == set(ref_top[i].tolist())


def test_scaled_index_search_matches_gt_for_perfect_reconstruction():
    X, Q = _toy_data()
    norms = compute_exact_norms(X)
    gt = normalized_ground_truth(X, Q, k=10)

    def perfect_reconstruct(ids):
        return X[ids]

    index = build_scaled_ip_index(perfect_reconstruct, n=X.shape[0],
                                   d=X.shape[1], norms=norms, chunk=64)
    _, ids = search_index(index, Q, k=10)
    rec = recall_at_ks(ids, gt, ks=(1, 10))
    assert rec[1] == pytest.approx(1.0)
    assert rec[10] == pytest.approx(1.0)


def test_recall_at_ks_partial():
    gt = np.array([[0, 1, 2]], dtype=np.uint32)
    ret = np.array([[0, 9, 2]], dtype=np.uint32)
    rec = recall_at_ks(ret, gt, ks=(1, 3))
    assert rec[1] == pytest.approx(1.0)
    assert rec[3] == pytest.approx(2.0 / 3.0)


def test_reconstruction_mse_zero_for_perfect():
    X, _ = _toy_data()
    sample = np.arange(X.shape[0], dtype=np.uint32)
    mse = reconstruction_mse(X, lambda ids: X[ids], sample_ids=sample, chunk=64)
    assert mse == pytest.approx(0.0)


def test_reconstruction_mse_known_value():
    X = np.zeros((2, 3), dtype=np.float32)
    Xhat_const = np.ones((2, 3), dtype=np.float32)
    sample = np.arange(2, dtype=np.uint32)
    mse = reconstruction_mse(X, lambda ids: Xhat_const[ids], sample_ids=sample, chunk=1)
    assert mse == pytest.approx(1.0)
