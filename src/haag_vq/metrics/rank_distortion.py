"""
Rank distortion metrics for evaluating top-k neighbor preservation.

This module measures how well vector quantization preserves the ranking of
nearest neighbors. Unlike recall (which measures overlap), rank distortion
measures positional disagreement using Hamming distance.

Key Concept:
-----------
For similarity search, we care about whether the "right" neighbors appear in
the top-k results. Rank distortion quantifies how much the compressed
representation changes the ranking order.

Metric Definition:
-----------------
For a query vector q:
    rank_distortion@k = hamming_distance(
        top_k_indices(compressed_space, q),
        top_k_indices(original_space, q)
    ) / k

This measures the fraction of disagreement in the top-k rankings:
- rank_distortion = 0: perfect ranking preservation (all top-k are the same)
- rank_distortion = 0.5: 50% of top-k neighbors differ
- rank_distortion = 1: completely different top-k sets

Difference from Recall:
----------------------
- Recall@k: What fraction of true top-k neighbors were retrieved?
- Rank Distortion@k: What fraction of retrieved top-k are WRONG?

They're complementary:
    rank_distortion@k = 1 - recall@k

Why This Matters:
----------------
1. More interpretable than recall for some use cases
2. Directly measures ranking error
3. Useful for evaluating approximate nearest neighbor (ANN) quality
4. Can be aggregated across multiple queries for overall system quality
"""

import numpy as np
from haag_vq.data.datasets import Dataset


def compute_rank_distortion(
    data: Dataset,
    model,
    k: int = 10,
    num_queries: int = 100
) -> float:
    """
    Compute rank distortion for top-k nearest neighbor retrieval.

    Args:
        data: Dataset with vectors, queries, and ground truth
        model: Quantizer with compress() and decompress() methods
        k: Number of neighbors to consider (top-k)
        num_queries: Number of query vectors to evaluate

    Returns:
        Average rank distortion across all queries (float between 0 and 1)

    Example:
        >>> pq = ProductQuantizer(M=8, B=8)
        >>> pq.fit(data.vectors)
        >>> rank_dist = compute_rank_distortion(data, pq, k=10)
        >>> print(f"Rank distortion@10: {rank_dist:.4f}")
        >>> # Interpretation: rank_dist=0.3 means 30% of top-10 results are wrong
    """
    # Step 1: Get ground truth top-k neighbors
    queries = data.queries[:num_queries]
    # data.ground_truth contains sorted indices (closest first)
    true_top_k_indices = data.ground_truth[:num_queries, :k]

    # Step 2: Get approximate top-k neighbors from compressed space
    X_compressed = model.compress(data.vectors)
    X_decompressed = model.decompress(X_compressed)

    # Compute distances from queries to decompressed vectors
    dists = data.distance_metric(queries, X_decompressed)
    retrieved_top_k_indices = dists.argsort(axis=1)[:, :k]

    # Step 3: Compute fraction of missing true neighbors
    total_missing = 0
    for i in range(num_queries):
        # Convert to sets
        true_set = set(true_top_k_indices[i])
        retrieved_set = set(retrieved_top_k_indices[i])

        # Count how many true top-k neighbors are missing from retrieved set
        missing = len(true_set - retrieved_set)
        total_missing += missing

    # Average fraction of missing neighbors (equivalent to 1 - recall)
    avg_rank_distortion = total_missing / (num_queries * k)

    return float(avg_rank_distortion)


def compute_rank_distortion_per_query(
    data: Dataset,
    model,
    k: int = 10,
    num_queries: int = 100
) -> np.ndarray:
    """
    Compute rank distortion for each query individually.

    Useful for analyzing variance in retrieval quality across different queries.

    Args:
        data: Dataset with vectors, queries, and ground truth
        model: Quantizer with compress() and decompress() methods
        k: Number of neighbors to consider
        num_queries: Number of query vectors to evaluate

    Returns:
        Array of shape (num_queries,) with rank distortion for each query

    Example:
        >>> rank_dists = compute_rank_distortion_per_query(data, pq, k=10)
        >>> print(f"Best query: {rank_dists.min():.4f}")
        >>> print(f"Worst query: {rank_dists.max():.4f}")
        >>> print(f"Std dev: {rank_dists.std():.4f}")
    """
    queries = data.queries[:num_queries]
    true_top_k_indices = data.ground_truth[:num_queries, :k]

    X_compressed = model.compress(data.vectors)
    X_decompressed = model.decompress(X_compressed)
    dists = data.distance_metric(queries, X_decompressed)
    retrieved_top_k_indices = dists.argsort(axis=1)[:, :k]

    rank_distortions = np.zeros(num_queries)
    for i in range(num_queries):
        true_set = set(true_top_k_indices[i])
        retrieved_set = set(retrieved_top_k_indices[i])
        # Count missing true neighbors
        missing = len(true_set - retrieved_set)
        rank_distortions[i] = missing / k

    return rank_distortions
