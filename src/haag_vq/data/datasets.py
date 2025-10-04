from typing import Callable, Optional
import numpy as np

from sklearn.metrics import pairwise_distances
from datasets import load_dataset # Importing the necessary Hugging Face library NK 10/3/2025
from sentence_transformers import SentenceTransformer # Need the SBERT model NK 10/3/2025

class Dataset:
    def __init__(
        self,
        vectors: np.ndarray,
        queries: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        num_queries: int = 100,
        distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_distances,
    ):

        self.vectors = vectors
        if queries is None:
            self.queries = vectors[:num_queries]
        else:
            assert num_queries <= len(queries)
            self.queries = queries[:num_queries]
        if ground_truth is None:
            # NOTE: This calculates the full distance matrix between queries and vectors,
            # which can be memory-intensive. For large datasets, a more efficient
            # approach (like k-nearest neighbor search) might be needed.
            dist_matrix = distance_metric(self.queries, self.vectors)
            # Sort distances to get ground truth nearest neighbor indices
            self.ground_truth = dist_matrix.argsort(axis=1)
        else:
            assert num_queries <= len(ground_truth)
            self.ground_truth = ground_truth[:num_queries]
        self.distance_metric = distance_metric


def load_dummy_dataset(num_samples=10000, dim=1024, seed=42) -> Dataset:
    np.random.seed(seed)
    return Dataset(np.random.randn(num_samples, dim))

# Adding a new function here NK 10/3/2025
def load_huggingface_dataset(
    dataset_name: str = "stsb_multi_mt",
    config_name: str = "en",
    model_name: str = "all-MiniLM-L6-v2",
    split: str = "train"
) -> Dataset:
    """ Load a text dataset from Hugging Face, compute Sentence-BERT embeddings,
    and return a local Dataset object.
    """
    print(f"1. Loading text data: {dataset_name}/{config_name} split={split}...")
    
    hf_dataset = load_dataset(dataset_name, config_name, split=split)
    
    sentences = hf_dataset['sentence1']
    
    print(f"2. Computing {len(sentences)} embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    vectors = model.encode(sentences, show_progress_bar=True)
    
    print("3. Initializing local Dataset object.")
    return Dataset(vectors=vectors)

