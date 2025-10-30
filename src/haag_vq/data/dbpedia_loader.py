# src/haag_vq/data/dbpedia_loader.py
"""
Loaders for DBpedia pre-embedded datasets from Qdrant.
Datasets:
  - https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M
  - https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M

These datasets contain 1M DBpedia entity records pre-embedded with OpenAI models.
"""
from typing import Optional, Literal
import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from .datasets import Dataset


def load_dbpedia_openai_1536(
    limit: Optional[int] = 100_000,
    num_queries: int = 100,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> Dataset:
    """Load DBpedia entities with OpenAI text-embedding-3-large (1536 dimensions).

    Args:
        limit: Maximum number of entities to load (default: 100k). Use None for all 1M entities.
        num_queries: Number of query vectors to use (default: 100)
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')
        streaming: If True, stream the dataset to avoid loading all into memory at once

    Returns:
        Dataset object with pre-computed 1536-dim embeddings

    Note:
        - Contains 1M DBpedia entities
        - Embeddings: OpenAI text-embedding-3-large (1536 dimensions)
        - Generated from concatenated title + text fields
    """
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face 'datasets' not installed. pip install datasets")

    print(f"Loading DBpedia OpenAI 1536-dim dataset (limit={limit}, streaming={streaming})...")

    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        split="train",
        cache_dir=cache_dir,
        streaming=streaming,
    )

    # Extract embeddings
    embeddings = []
    count = 0
    max_count = limit if limit is not None else 1_000_000

    print(f"Extracting embeddings...")
    iterator = tqdm(ds, total=min(limit or 1_000_000, 1_000_000), desc="Loading vectors")

    for item in iterator:
        if count >= max_count:
            break
        embeddings.append(item['text-embedding-3-large-1536-embedding'])
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


def load_dbpedia_openai_3072(
    limit: Optional[int] = 100_000,
    num_queries: int = 100,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> Dataset:
    """Load DBpedia entities with OpenAI text-embedding-3-large (3072 dimensions).

    Args:
        limit: Maximum number of entities to load (default: 100k). Use None for all 1M entities.
        num_queries: Number of query vectors to use (default: 100)
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')
        streaming: If True, stream the dataset to avoid loading all into memory at once

    Returns:
        Dataset object with pre-computed 3072-dim embeddings

    Note:
        - Contains 1M DBpedia entities
        - Embeddings: OpenAI text-embedding-3-large (3072 dimensions)
        - Generated from concatenated title + text fields
        - This dataset also includes legacy ada-002 embeddings (1536-dim) but we use the 3072-dim ones
    """
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face 'datasets' not installed. pip install datasets")

    print(f"Loading DBpedia OpenAI 3072-dim dataset (limit={limit}, streaming={streaming})...")

    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        split="train",
        cache_dir=cache_dir,
        streaming=streaming,
    )

    # Extract embeddings (use the 3072-dim version, not the ada-002 1536-dim)
    embeddings = []
    count = 0
    max_count = limit if limit is not None else 1_000_000

    print(f"Extracting embeddings...")
    iterator = tqdm(ds, total=min(limit or 1_000_000, 1_000_000), desc="Loading vectors")

    for item in iterator:
        if count >= max_count:
            break
        embeddings.append(item['text-embedding-3-large-3072-embedding'])
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


def load_dbpedia_openai(
    embedding_dim: Literal[1536, 3072] = 1536,
    limit: Optional[int] = 100_000,
    num_queries: int = 100,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> Dataset:
    """Load DBpedia entities with OpenAI embeddings (convenience function).

    Args:
        embedding_dim: Embedding dimension to use (1536 or 3072)
        limit: Maximum number of entities to load (default: 100k)
        num_queries: Number of query vectors to use (default: 100)
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')
        streaming: If True, stream the dataset

    Returns:
        Dataset object with pre-computed embeddings
    """
    if embedding_dim == 1536:
        return load_dbpedia_openai_1536(limit, num_queries, cache_dir, streaming)
    elif embedding_dim == 3072:
        return load_dbpedia_openai_3072(limit, num_queries, cache_dir, streaming)
    else:
        raise ValueError(f"Unsupported embedding_dim: {embedding_dim}. Must be 1536 or 3072.")


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="DBpedia OpenAI embeddings loader")
    p.add_argument("--dim", type=int, choices=[1536, 3072], default=1536,
                   help="Embedding dimension (1536 or 3072)")
    p.add_argument("--limit", type=int, default=100_000, help="Max entities to load")
    p.add_argument("--num-queries", type=int, default=100, help="Number of queries")
    p.add_argument("--cache-dir", type=str, default="../datasets", help="HF cache directory")
    p.add_argument("--out", type=str, help="Output file (default: data/dbpedia-{dim}.npz)")
    p.add_argument("--streaming", action="store_true", help="Enable streaming")

    args = p.parse_args()

    ds = load_dbpedia_openai(
        embedding_dim=args.dim,
        limit=args.limit,
        num_queries=args.num_queries,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    out_file = args.out or f"data/dbpedia-{args.dim}.npz"
    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)
    np.savez_compressed(
        out_file,
        vectors=ds.vectors,
        queries=ds.queries,
    )
    print(f"Saved: {out_file}")
