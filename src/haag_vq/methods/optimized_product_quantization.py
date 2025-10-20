import numpy as np

from .base_quantizer import BaseQuantizer

import faiss

class OptimizedProductQuantizer(BaseQuantizer):
    def __init__(self, M: int, B: int = 8):
        """ OPQ
        Ge, T., He, K., Ke, Q., & Sun, J. (2013). Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence, 36(4), 744-755.
        https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf
        Args:
            M (int): Number of low-dimensional subquantizers.
            B (int, optional): Number of bits per subvector index. Defaults to 8.
        """
        self.M = M
        self.B = B
        self.opq: faiss.OPQMatrix = None
        self.pq: faiss.ProductQuantizer = None

    def fit(self, X: np.ndarray):
        N, D = X.shape
        assert D % self.M == 0, "D must be divisible by M"
        self.opq = faiss.OPQMatrix(D, self.M, D)
        self.opq.train(X)
        self.pq = faiss.ProductQuantizer(D, self.M, self.B)
        self.pq.train(X)

    def compress(self, X: np.ndarray) -> np.ndarray:
        return self.pq.compute_codes(self.opq.apply(X))  # shape: (N, M)

    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        return self.opq.reverse_transform(self.pq.decode(compressed))  # shape: (N, D)

    def get_compression_ratio(self, X: np.ndarray) -> float:
        """Return compression ratio (original bytes / compressed bytes).

        OPQ applies a rotation (no per-vector storage) and then PQ codes of size M * B/8 bytes.
        Assumes `float32` inputs.
        """
        D = int(X.shape[1])
        original_size_bytes = D * 4
        # Same storage as PQ: ceil(M * B / 8) bytes per vector
        compressed_size_bytes = int((self.M * self.B + 7) // 8)
        return float(original_size_bytes / compressed_size_bytes)
