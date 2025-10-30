# src/haag_vq/data/cohere_msmarco_loader.py
"""
Loader for Cohere MS MARCO v2.1 pre-embedded dataset.
Dataset: https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3

This dataset contains ~53.2M passages and ~1.6K queries, pre-embedded with Cohere Embed English v3.
Embeddings are ~1024 dimensions.
"""
from typing import Optional
import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from .datasets import Dataset


def load_cohere_msmarco_passages(
    limit: Optional[int] = 100_000,
    num_queries: int = 100,
    cache_dir: Optional[str] = None,
    streaming: bool = True,
) -> Dataset:
    """Load pre-embedded MS MARCO v2.1 passages from Cohere.

    Args:
        limit: Maximum number of passages to load (default: 100k). Use None for all ~53M passages.
        num_queries: Number of query vectors to use (default: 100)
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')
        streaming: If True, stream the dataset to avoid loading all into memory at once

    Returns:
        Dataset object with pre-computed embeddings

    Note:
        - The passages subset is very large (~53.2M rows)
        - Embeddings are already computed (dimension ~1024)
        - Set streaming=True for large limits to avoid memory issues
    """
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face 'datasets' not installed. pip install datasets")

    print(f"Loading Cohere MS MARCO v2.1 passages (limit={limit}, streaming={streaming})...")

    # Load the passages subset
    ds = load_dataset(
        "Cohere/msmarco-v2.1-embed-english-v3",
        "passages",
        split="train",
        cache_dir=cache_dir,
        streaming=streaming,
    )

    # Extract embeddings
    embeddings = []
    count = 0
    max_count = limit if limit is not None else float('inf')

    print(f"Extracting embeddings...")
    for item in tqdm(ds, total=limit, desc="Loading vectors"):
        if count >= max_count:
            break
        embeddings.append(item['emb'])
        count += 1

    vectors = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")

    # Use first num_queries vectors as queries
    queries = vectors[:num_queries] if len(vectors) >= num_queries else vectors[:len(vectors)]

    # Skip ground truth computation for large datasets
    return Dataset(
        vectors=vectors,
        queries=queries,
        skip_ground_truth=True,
    )


def load_cohere_msmarco_queries(
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Load pre-embedded MS MARCO v2.1 queries from Cohere.

    Args:
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')

    Returns:
        Array of query embeddings (~1.6K queries, ~1024 dimensions)
    """
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face 'datasets' not installed. pip install datasets")

    print(f"Loading Cohere MS MARCO v2.1 queries...")

    # Load the queries subset (non-streaming since it's small)
    ds = load_dataset(
        "Cohere/msmarco-v2.1-embed-english-v3",
        "queries",
        split="train",
        cache_dir=cache_dir,
        streaming=False,
    )

    # Extract embeddings
    embeddings = [item['emb'] for item in ds]
    queries = np.array(embeddings, dtype=np.float32)

    print(f"Loaded {len(queries)} query vectors with dimension {queries.shape[1]}")
    return queries


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Cohere MS MARCO v2.1 loader")
    p.add_argument("--limit", type=int, default=100_000, help="Max passages to load")
    p.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    p.add_argument("--cache-dir", type=str, default="../datasets", help="HF cache directory")
    p.add_argument("--out", type=str, default="data/cohere-msmarco.npz", help="Output file")
    p.add_argument("--no-streaming", action="store_true", help="Disable streaming")

    args = p.parse_args()

    ds = load_cohere_msmarco_passages(
        limit=args.limit,
        num_queries=args.num_queries,
        cache_dir=args.cache_dir,
        streaming=not args.no_streaming,
    )

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        vectors=ds.vectors,
        queries=ds.queries,
    )
    print(f"Saved: {args.out}")
