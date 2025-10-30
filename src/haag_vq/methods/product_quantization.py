from typing import Optional, List
import numpy as np

from .base_quantizer import BaseQuantizer

import faiss


class ProductQuantizer(BaseQuantizer):
    def __init__(
        self,
        M: Optional[int] = None,
        B: int = 8,
        *,
        # Backward-compat kwargs (deprecated):
        num_chunks: Optional[int] = None,
        num_clusters: Optional[int] = None,
    ):
        """
        Product Quantization using FAISS' ProductQuantizer.

        Args:
            M: Number of subquantizers (chunks).
            B: Number of bits per subvector index (ksub = 2**B).
            num_chunks: Deprecated alias for `M`.
            num_clusters: Deprecated alias for `2**B` (must be a power of two).
        """
        # Resolve parameter aliases for backward compatibility
        if M is None and num_chunks is not None:
            M = int(num_chunks)
        if num_chunks is not None and M is not None and int(num_chunks) != int(M):
            raise ValueError("Conflicting values for M and num_chunks")

        if num_clusters is not None:
            # Derive B from num_clusters; must be an exact power of two
            if num_clusters <= 0:
                raise ValueError("num_clusters must be positive")
            log2_clusters = int(round(np.log2(num_clusters)))
            if 2 ** log2_clusters != int(num_clusters):
                raise ValueError("num_clusters must be a power of two")
            B = log2_clusters

        if M is None:
            M = 8  # sensible default

        self.M: int = int(M)
        self.B: int = int(B)

        # Compatibility attributes used elsewhere in the codebase/tests
        self.num_chunks: int = self.M
        self.num_clusters: int = 2 ** self.B

        # Learned state
        self.codebooks: List[np.ndarray] = []  # list of (ksub, dsub)
        self.chunk_dim: Optional[int] = None
        self.pq: Optional[faiss.ProductQuantizer] = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        if D % self.M != 0:
            raise AssertionError("D must be divisible by M (number of subquantizers)")

        self.chunk_dim = D // self.M

        # Train FAISS ProductQuantizer
        self.pq = faiss.ProductQuantizer(D, self.M, self.B)
        self.pq.train(X)

        # Extract trained centroids into compatibility structure
        # centroids is a flat array of size (M * ksub * dsub)
        flat = faiss.vector_to_array(self.pq.centroids)
        centroids = flat.reshape(self.M, self.pq.ksub, self.pq.dsub)
        self.codebooks = [np.array(centroids[m], copy=True) for m in range(self.M)]

    def compress(self, X: np.ndarray) -> np.ndarray:
        if self.pq is None:
            raise RuntimeError("ProductQuantizer must be fitted before compress(). Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        return self.pq.compute_codes(X)

    def decompress(self, codes: np.ndarray) -> np.ndarray:
        if self.pq is None:
            raise RuntimeError("ProductQuantizer must be fitted before decompress(). Call fit() first.")
        codes = np.asarray(codes)
        return self.pq.decode(codes)

    def get_compression_ratio(self, X: np.ndarray) -> float:
        """Return compression ratio (original bytes / compressed bytes).

        Assumes `float32` inputs and `B` bits per subvector code across `M` subvectors.
        """
        D = int(X.shape[1])
        original_size_bytes = D * 4  # float32 = 4 bytes
        if self.pq is not None and hasattr(self.pq, "code_size"):
            compressed_size_bytes = int(self.pq.code_size)
        else:
            compressed_size_bytes = int((self.M * self.B + 7) // 8)
        return float(original_size_bytes / compressed_size_bytes)
