# src/haag_vq/methods/base_search_index.py
"""Primary benchmark interface for all VQ search methods.

Every method (SAQ, PQ, OPQ, SQ, RaBitQ, Faiss baselines) implements this
interface — either natively (SAQ, Faiss) or via wrapper classes that adapt
BaseQuantizer implementations (FlatQuantizedIndex, IvfQuantizedIndex).

Primary metrics : recall@k, QPS, memory_footprint
Secondary metric: reconstruction MSE (optional, via reconstruction_mse())
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np


class BaseSearchIndex(ABC):
    """Abstract base class for all VQ search methods benchmarked in haag_vq."""

    @abstractmethod
    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        """Learn index from training vectors X of shape (N, D).

        All preprocessing (PCA, k-means, codebook fitting) is internal.
        After fit(), search() and memory_footprint() must be valid.
        """

    @abstractmethod
    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        """Return (nq, k) uint32 array of approximate nearest neighbor IDs.

        Q: (nq, D) float32 query matrix.
        k: number of neighbors.
        """

    @abstractmethod
    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) where both are (nq, k) float32 / uint32.

        For 'l2' metric: distances are squared L2.
        For 'ip' metric: distances are inner products (higher = better).
        """

    @abstractmethod
    def memory_footprint(self) -> int:
        """Estimated index memory in bytes.

        Used for compression ratio computation.
        Does not include Python object overhead — only the encoded data
        and auxiliary structures (centroids, codebooks, etc.).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist index to disk at the given path."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore index from disk."""

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Return mean reconstruction MSE over a sample of vectors.

        This is a secondary benchmark metric. The default implementation
        returns None (method does not support reconstruction).

        Concrete classes override this when their underlying quantizer or
        index supports decompress(). Used as a secondary benchmark column
        alongside recall@k, QPS, and memory_footprint.

        Args:
            X:          Original vectors (N, D) used as ground truth.
            sample_ids: If provided, compute MSE over X[sample_ids].
                        If None, use all vectors.

        Returns:
            Mean per-element MSE as float, or None if not supported.
        """
        return None
