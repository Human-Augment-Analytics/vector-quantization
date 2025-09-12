import numpy as np

from haag_vq.data.datasets import Dataset


def evaluate_recall(data: Dataset, model, num_queries=100):
    # Step 1: True neighbors
    queries = data.queries[:num_queries]
    true_nn_indices = data.ground_truth[:num_queries]

    # Step 2: Approximate neighbors
    X_compressed = model.compress(data.vectors)
    X_decompressed = model.decompress(X_compressed)
    dists = data.distance_metric(queries, X_decompressed)
    retrieved_indices = dists.argsort(axis=1)

    # Step 3: Evaluate recall
    recall_10 = recall_at_k(true_nn_indices, retrieved_indices, k=10)
    recall_100 = recall_at_k(true_nn_indices, retrieved_indices, k=100)

    return {
        "recall@10": recall_10,
        "recall@100": recall_100
    }

def recall_at_k(true_nn: np.ndarray, retrieved: np.ndarray, k: int) -> float:
    """
    Compute Recall@k using full GT and retrieved index lists

    Args:
        true_nn: shape (num_queries, N), sorted GT neighbors (closest first)
        retrieved: shape (num_queries, N), sorted retrieved neighbors
        k: top-k to check

    Returns:
        Recall@k as float between 0 and 1
    """
    hits = 0
    for i in range(len(true_nn)):
        gt_top_k = set(true_nn[i, :k])
        retrieved_top_k = set(retrieved[i, :k])
        hits += len(gt_top_k & retrieved_top_k) / k
    return hits / len(true_nn)