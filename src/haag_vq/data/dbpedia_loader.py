# src/haag_vq/data/dbpedia_loader.py
"""
Loaders for DBpedia pre-embedded datasets from Qdrant.
Datasets:
  - https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K
  - https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M
  - https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M

These datasets contain DBpedia entity records (100K or 1M) pre-embedded with OpenAI models.
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

    # Extract embeddings - MEMORY OPTIMIZED: pre-allocate numpy array
    max_count = limit if limit is not None else 1_000_000
    dimension = 1536

    # Pre-allocate numpy array directly (saves ~2x memory vs list approach)
    vectors = np.zeros((max_count, dimension), dtype=np.float32)
    count = 0

    print(f"Extracting embeddings (pre-allocated for {max_count:,} vectors)...")
    iterator = tqdm(ds, total=max_count, desc="Loading vectors")

    for item in iterator:
        if count >= max_count:
            break
        vectors[count] = item['text-embedding-3-large-1536-embedding']
        count += 1

    # Trim array if we loaded fewer vectors than expected
    if count < max_count:
        vectors = vectors[:count]

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")

    # Use first num_queries vectors as queries
    queries = vectors[:num_queries] if len(vectors) >= num_queries else vectors[:len(vectors)]

    # Compute ground truth with FAISS (fast even for 1M vectors with 100 queries)
    return Dataset(
        vectors=vectors,
        queries=queries,
        skip_ground_truth=False,  # FAISS makes this fast
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

    # Extract embeddings - MEMORY OPTIMIZED: pre-allocate numpy array
    # (use the 3072-dim version, not the ada-002 1536-dim)
    max_count = limit if limit is not None else 1_000_000
    dimension = 3072

    # Pre-allocate numpy array directly (saves ~2x memory vs list approach)
    vectors = np.zeros((max_count, dimension), dtype=np.float32)
    count = 0

    print(f"Extracting embeddings (pre-allocated for {max_count:,} vectors)...")
    iterator = tqdm(ds, total=max_count, desc="Loading vectors")

    for item in iterator:
        if count >= max_count:
            break
        vectors[count] = item['text-embedding-3-large-3072-embedding']
        count += 1

    # Trim array if we loaded fewer vectors than expected
    if count < max_count:
        vectors = vectors[:count]

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")

    # Use first num_queries vectors as queries
    queries = vectors[:num_queries] if len(vectors) >= num_queries else vectors[:len(vectors)]

    # Compute ground truth with FAISS (fast even for 1M vectors with 100 queries)
    return Dataset(
        vectors=vectors,
        queries=queries,
        skip_ground_truth=False,  # FAISS makes this fast
    )


def load_dbpedia_openai_1536_100k(
    limit: Optional[int] = None,
    num_queries: int = 100,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> Dataset:
    """Load DBpedia entities with OpenAI text-embedding-3-large (1536 dimensions, 100K subset).

    Args:
        limit: Maximum number of entities to load (default: None = all 100K).
        num_queries: Number of query vectors to use (default: 100)
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')
        streaming: If True, stream the dataset to avoid loading all into memory at once

    Returns:
        Dataset object with pre-computed 1536-dim embeddings

    Note:
        - Contains 100K DBpedia entities (smaller subset for faster testing)
        - Embeddings: OpenAI text-embedding-3-large (1536 dimensions)
        - Generated from concatenated title + text fields
        - Ideal for quick experiments and testing before scaling to 1M dataset
    """
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face 'datasets' not installed. pip install datasets")

    print(f"Loading DBpedia OpenAI 1536-dim 100K dataset (limit={limit}, streaming={streaming})...")

    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K",
        split="train",
        cache_dir=cache_dir,
        streaming=streaming,
    )

    # Extract embeddings - MEMORY OPTIMIZED: pre-allocate numpy array
    # This avoids Python list overhead and temporary allocations during conversion
    max_count = limit if limit is not None else 100_000
    dimension = 1536

    # Pre-allocate numpy array directly (saves ~2x memory vs list approach)
    vectors = np.zeros((max_count, dimension), dtype=np.float32)
    count = 0

    print(f"Extracting embeddings (pre-allocated for {max_count:,} vectors)...")
    iterator = tqdm(ds, total=max_count, desc="Loading vectors")

    for item in iterator:
        if count >= max_count:
            break
        vectors[count] = item['text-embedding-3-large-1536-embedding']
        count += 1

    # Trim array if we loaded fewer vectors than expected
    if count < max_count:
        vectors = vectors[:count]

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")

    # Use first num_queries vectors as queries
    queries = vectors[:num_queries] if len(vectors) >= num_queries else vectors[:len(vectors)]

    # For 100K dataset, compute ground truth (fast with FAISS)
    # This enables recall and rank distortion metrics
    return Dataset(
        vectors=vectors,
        queries=queries,
        skip_ground_truth=False,  # Always compute for 100K - it's manageable
    )


def load_dbpedia_openai(
    embedding_dim: Literal[1536, 3072] = 1536,
    limit: Optional[int] = 100_000,
    num_queries: int = 100,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    use_100k_subset: bool = False,
) -> Dataset:
    """Load DBpedia entities with OpenAI embeddings (convenience function).

    Args:
        embedding_dim: Embedding dimension to use (1536 or 3072)
        limit: Maximum number of entities to load (default: 100k)
        num_queries: Number of query vectors to use (default: 100)
        cache_dir: Optional cache directory for Hugging Face datasets (e.g., '../datasets/')
        streaming: If True, stream the dataset
        use_100k_subset: If True and embedding_dim=1536, use the 100K dataset variant

    Returns:
        Dataset object with pre-computed embeddings
    """
    if embedding_dim == 1536:
        if use_100k_subset:
            return load_dbpedia_openai_1536_100k(limit, num_queries, cache_dir, streaming)
        else:
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
    p.add_argument("--use-100k", action="store_true", help="Use 100K subset (only for dim=1536)")

    args = p.parse_args()

    ds = load_dbpedia_openai(
        embedding_dim=args.dim,
        limit=args.limit,
        num_queries=args.num_queries,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        use_100k_subset=args.use_100k,
    )

    suffix = "-100k" if args.use_100k else ""
    out_file = args.out or f"data/dbpedia-{args.dim}{suffix}.npz"
    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)
    np.savez_compressed(
        out_file,
        vectors=ds.vectors,
        queries=ds.queries,
    )
    print(f"Saved: {out_file}")
