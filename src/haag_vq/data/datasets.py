from typing import Callable, Optional
import numpy as np

from sklearn.metrics import pairwise_distances
from datasets import load_dataset # Importing the necessary Hugging Face library NK 10/3/2025
from sentence_transformers import SentenceTransformer # Need the SBERT model NK 10/3/2025

def compute_ground_truth_faiss(queries: np.ndarray, vectors: np.ndarray, k: int = 100) -> np.ndarray:
    """
    Compute ground truth k-nearest neighbors using FAISS (memory-efficient).

    Args:
        queries: Query vectors (num_queries x dimension)
        vectors: Database vectors (num_vectors x dimension)
        k: Number of nearest neighbors to compute

    Returns:
        Ground truth indices (num_queries x k)
    """
    try:
        import faiss
    except ImportError:
        print("WARNING: FAISS not available, falling back to sklearn pairwise_distances")
        print("Install FAISS for better performance: pip install faiss-cpu")
        dist_matrix = pairwise_distances(queries, vectors)
        return dist_matrix.argsort(axis=1)[:, :k]

    # Use FAISS for efficient k-NN search
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors.astype(np.float32))

    _, indices = index.search(queries.astype(np.float32), k)
    return indices

class Dataset:
    def __init__(
        self,
        vectors: np.ndarray,
        queries: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        num_queries: int = 100,
        distance_metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = pairwise_distances,
        skip_ground_truth: bool = False,
    ):
        """Initialize a Dataset for vector quantization benchmarking.

        Args:
            vectors: The main dataset vectors (N x D)
            queries: Optional query vectors (default: use first num_queries from vectors)
            ground_truth: Optional precomputed ground truth nearest neighbors (num_queries x k)
            num_queries: Number of queries to use if queries is None
            distance_metric: Distance metric function (for computing ground truth on-the-fly)
            skip_ground_truth: If True, don't compute ground truth (for large datasets)
        """
        self.vectors = vectors
        if queries is None:
            self.queries = vectors[:num_queries]
        else:
            assert num_queries <= len(queries)
            self.queries = queries[:num_queries]

        if ground_truth is not None:
            # Use precomputed ground truth
            assert num_queries <= len(ground_truth)
            self.ground_truth = ground_truth[:num_queries]
        elif skip_ground_truth:
            # Skip ground truth computation (recall metrics will not be available)
            self.ground_truth = None
            print("WARNING: Ground truth computation skipped. Recall metrics will not be available.")
        else:
            # Compute ground truth using FAISS (memory-efficient)
            print(f"Computing ground truth for {len(self.queries)} queries x {len(self.vectors)} vectors...")
            self.ground_truth = compute_ground_truth_faiss(self.queries, self.vectors, k=100)

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

