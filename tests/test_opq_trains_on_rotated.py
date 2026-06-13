import numpy as np

from haag_vq.methods.optimized_product_quantization import OptimizedProductQuantizer
from haag_vq.methods.product_quantization import ProductQuantizer


def _data(seed=0, n=4000, d=64):
    rng = np.random.default_rng(seed)
    # anisotropic + correlated so a rotation genuinely helps OPQ
    A = rng.standard_normal((d, d))
    return (rng.standard_normal((n, d)) @ A).astype(np.float32)


def test_opq_mse_not_worse_than_pq():
    # OPQ generalizes PQ (rotation + PQ); with codebooks trained on the rotated
    # data, OPQ reconstruction MSE must be <= PQ MSE at matched M, B.
    X = _data()
    M, B = 8, 8
    pq = ProductQuantizer(M=M, B=B); pq.fit(X)
    pq_mse = float(np.mean((X - pq.decompress(pq.compress(X))) ** 2))

    opq = OptimizedProductQuantizer(M=M, B=B); opq.fit(X)
    opq_mse = float(np.mean((X - opq.decompress(opq.compress(X))) ** 2))

    # allow a tiny tolerance for optimizer noise
    assert opq_mse <= pq_mse * 1.02, f"OPQ {opq_mse:.6g} should be <= PQ {pq_mse:.6g}"
