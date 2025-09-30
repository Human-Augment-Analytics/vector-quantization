from pathlib import Path
from typing import Callable, Optional
import numpy as np
import os

from sklearn.metrics import pairwise_distances


class Dataset:
    def __init__(
        self,
        vectors: np.ndarray,
        queries: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        num_queries: int = 100,
        distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_distances,
    ):
        """ Vector DB dataset

        Args:
            vectors (np.ndarray):
                A `num_samples` X `dim` matrix of database vectors.
            queries (Optional[np.ndarray], optional):
                A `num_queries` X `dim` matrix of query vectors.
                If None, use the first `num_queries` database vectors as query vectors.
            ground_truth (Optional[np.ndarray], optional):
                A `num_queries` X `num_samples` matrix of ground truth vectors.
                Each ground truth vector is the indices of all database vectors,
                sorted by their distance from the query vectors.
                If None, compute the ground truth.
            num_queries (int, optional):
                The number of query vectors.
            distance_metric (Callable[[np.ndarray, np.ndarray], np.ndarray], optional):
                The function for measuring the distance between two vectors.
                Default to Euclidean `pairwise_distances`.
        """
        self.vectors = vectors
        if queries is None:
            self.queries = vectors[:num_queries]
        else:
            assert num_queries <= len(queries)
            self.queries = queries[:num_queries]
        if ground_truth is None:
            self.ground_truth = distance_metric(self.queries, self.vectors).argsort(axis=1)
        else:
            assert num_queries <= len(ground_truth)
            self.ground_truth = ground_truth[:num_queries]
        self.distance_metric = distance_metric

    @staticmethod
    def load(
        directory: Path,
        num_queries: int = 100,
        distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_distances,
    ) -> "Dataset":
        """ Load a vector DB dataset from files.
        If the queries and/or ground truth files do not exist, compute and dump the files.

        Args:
            directory (Path):
                The directory containing numpy array files
            num_queries (int, optional):
                The number of query vectors.
            distance_metric (Callable[[np.ndarray, np.ndarray], np.ndarray], optional):
                The function for measuring the distance between two vectors.
                Default to Euclidean `pairwise_distances`.
                Note that if we use a public benchmark with a custom distance metric,
                we should not reuse the ground truth from that benchmark.

        Raises:
            FileNotFoundError: The vectors file is not found.

        Returns:
            Dataset: The dataset loaded from files.
        """
        if os.path.exists(vectors_path := directory / "vectors.npy"):
            vectors = np.fromfile(vectors_path)
        else:
            raise FileNotFoundError(f"{vectors_path} not found")
        if os.path.exists(queries_path := directory / "queries.npy"):
            queries = np.fromfile(queries_path)
        else:
            queries = None
        if os.path.exists(ground_truth_path := directory / "ground_truth.npy"):
            ground_truth = np.fromfile(ground_truth_path)
        else:
            ground_truth = None
        dataset = Dataset(vectors, queries, ground_truth, num_queries, distance_metric)
        if not queries:
            np.save(queries_path, dataset.queries)
        if not ground_truth:
            np.save(ground_truth_path, dataset.ground_truth)


def load_dummy_dataset(num_samples=10000, dim=1024, seed=42) -> Dataset:
    np.random.seed(seed)
    return Dataset(np.random.randn(num_samples, dim))
