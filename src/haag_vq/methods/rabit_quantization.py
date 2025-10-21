import numpy as np

from .base_quantizer import BaseQuantizer
from ..utils.faiss_utils import MetricType

import faiss


class RaBitQuantizer(BaseQuantizer):
    def __init__(self, metric_type: MetricType = MetricType.L2):
        """ RaBitQ
        Gao, J., & Long, C. (2024). Rabitq: Quantizing high-dimensional vectors with a theoretical error bound for approximate nearest neighbor search. Proceedings of the ACM on Management of Data, 2(3), 1-27.
        https://dl.acm.org/doi/pdf/10.1145/3654970
        Args:
            metric_type (MetricType): Distance metric type enum for FAISS.
        """
        self.metric_type = metric_type
        self.rabitq: faiss.RaBitQuantizer = None

    def fit(self, X: np.ndarray):
        N, D = X.shape
        self.rabitq = faiss.RaBitQuantizer(D, self.metric_type)
        self.rabitq.train(X)

    def compress(self, X: np.ndarray) -> np.ndarray:
        return self.rabitq.compute_codes(X)

    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        return self.rabitq.decode(compressed)  # shape: (N, D)

    def get_compression_ratio(self, X: np.ndarray) -> float:
        """Return compression ratio (original bytes / compressed bytes).

        RaBitQ stores `code_size` bytes per vector as reported by FAISS.
        Assumes original inputs are float32 (4 bytes per dimension).
        """
        D = int(X.shape[1])
        original_size_bytes = D * 4
        compressed_size_bytes = int(self.rabitq.code_size)
        return float(original_size_bytes / compressed_size_bytes)
