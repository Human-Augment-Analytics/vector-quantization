"""
Pairwise distance distortion metrics.

This module evaluates how well vector quantization preserves pairwise distances
between vectors. This is critical for similarity search applications where the
goal is to preserve relative distances, not perfect reconstruction.

Key Concept:
-----------
In many vector database applications, we care more about whether two vectors are
"close" or "far" than about exact reconstruction. Pairwise distance distortion
measures how much the compression distorts the distance between pairs of vectors.

Metric Definition:
-----------------
For a pair of vectors (v1, v2):
    distortion = |distance(compressed(v1), compressed(v2)) / distance(v1, v2) - 1|

This measures the relative error in distance computation:
- distortion = 0: perfect distance preservation
- distortion < 0.1: distances preserved within 10% error
- distortion > 1: severely distorted distances

Why This Matters:
----------------
1. Asymmetric Distance Computation: Many vector DBs (e.g., FAISS with PQ) compute
   distances directly in compressed space without decompression.
2. Ranking Quality: Preserving relative distances is crucial for top-k retrieval.
3. Trade-off Analysis: Low reconstruction error doesn't guarantee good distance
   preservation, and vice versa.
"""

import numpy as np
from typing import Dict


def compute_pairwise_distortion(
    X_original: np.ndarray,
    X_compressed_codes: np.ndarray,
    model,
    num_pairs: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute pairwise distance distortion for random pairs of vectors.

    Args:
        X_original: Original vectors, shape (N, D)
        X_compressed_codes: Compressed codes, shape (N, ...)
        model: Quantizer with a decompress() method
        num_pairs: Number of random pairs to sample
        seed: Random seed for reproducibility

    Returns:
        Dictionary with statistics:
        - mean: Average distortion across all pairs
        - median: Median distortion
        - max: Maximum distortion observed
        - std: Standard deviation of distortion

    Example:
        >>> pq = ProductQuantizer(M=8, B=8)
        >>> pq.fit(X)
        >>> codes = pq.compress(X)
        >>> distortion = compute_pairwise_distortion(X, codes, pq, num_pairs=1000)
        >>> print(f"Mean distance distortion: {distortion['mean']:.4f}")
    """
    np.random.seed(seed)
    N = len(X_original)

    # Sample random pairs
    idx1 = np.random.randint(0, N, num_pairs)
    idx2 = np.random.randint(0, N, num_pairs)
    # Ensure pairs are distinct
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    if len(idx1) == 0:
        # Fallback: create distinct pairs manually
        idx1 = np.arange(min(num_pairs, N // 2))
        idx2 = np.arange(min(num_pairs, N // 2)) + 1

    # Compute original distances
    pairs_original_1 = X_original[idx1]
    pairs_original_2 = X_original[idx2]
    original_dists = np.linalg.norm(pairs_original_1 - pairs_original_2, axis=1)

    # Decompress and compute compressed distances
    X_decompressed = model.decompress(X_compressed_codes)
    pairs_compressed_1 = X_decompressed[idx1]
    pairs_compressed_2 = X_decompressed[idx2]
    compressed_dists = np.linalg.norm(pairs_compressed_1 - pairs_compressed_2, axis=1)

    # Compute relative distortion
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    relative_distortion = np.abs(compressed_dists / (original_dists + epsilon) - 1)

    return {
        "mean": float(np.mean(relative_distortion)),
        "median": float(np.median(relative_distortion)),
        "max": float(np.max(relative_distortion)),
        "std": float(np.std(relative_distortion)),
        "num_pairs": len(idx1)
    }


def compute_asymmetric_pairwise_distortion(
    X_original: np.ndarray,
    X_compressed_codes: np.ndarray,
    model,
    num_pairs: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute pairwise distance distortion using asymmetric distance computation.

    This is more realistic for many vector DB implementations where queries are
    uncompressed but the database is compressed.

    Args:
        X_original: Original vectors, shape (N, D)
        X_compressed_codes: Compressed codes for database vectors
        model: Quantizer with asymmetric distance support
        num_pairs: Number of random pairs to sample
        seed: Random seed

    Returns:
        Dictionary with distortion statistics

    Note:
        This function assumes the model supports asymmetric distance computation.
        For basic quantizers, this may not be implemented yet.
    """
    # TODO: Implement asymmetric distance computation when models support it
    # For now, fall back to symmetric (decompress-based) computation
    return compute_pairwise_distortion(
        X_original, X_compressed_codes, model, num_pairs, seed
    )
