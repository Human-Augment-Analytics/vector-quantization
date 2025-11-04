#!/usr/bin/env python3
"""
Streaming demo for large-scale MS MARCO dataset (53M vectors).

Strategy:
1. Train quantizers on a representative subset (100K-1M vectors)
2. Compress full dataset in batches using streaming
3. Compute metrics on compressed batches

MEMORY REQUIREMENTS:
- Training subset: ~500 MB (100K vectors) to ~5 GB (1M vectors)
- Streaming batch size: ~50-100 MB per batch
- Total peak: Training subset + batch overhead = ~1-6 GB

This approach allows processing 53M vectors without loading them all into memory.
"""

import os
import numpy as np
from tqdm import tqdm

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set HuggingFace cache directories
# Priority: 1) Shared cache (if exists), 2) $TMPDIR, 3) Local .cache
SHARED_CACHE = "/storage/ice-shared/cs8903onl/.cache/huggingface"

if os.path.exists(SHARED_CACHE):
    hf_cache_base = SHARED_CACHE
elif "TMPDIR" in os.environ:
    hf_cache_base = os.path.join(os.environ["TMPDIR"], "hf_cache")
else:
    hf_cache_base = os.path.abspath("../.cache/huggingface")

os.environ["HF_HOME"] = hf_cache_base
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_base, "datasets")
from haag_vq.data import load_cohere_msmarco_passages
from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.utils.run_logger import log_run


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_training_subset(limit=100_000):
    """Load a subset for training the quantizer."""
    print_section("Loading Training Subset")

    cache_dir = os.path.join(hf_cache_base, "datasets")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nLoading {limit:,} vectors from MS MARCO for training...")
    print("(Using streaming to avoid loading full 53M dataset)")

    data = load_cohere_msmarco_passages(
        limit=limit,
        num_queries=100,
        cache_dir=cache_dir,
        streaming=True,  # Critical for large dataset
    )

    print(f"✅ Training subset loaded: {data.vectors.shape}")
    print(f"   Vectors: {data.vectors.shape[0]:,}")
    print(f"   Dimensions: {data.vectors.shape[1]}")
    print(f"   Memory: ~{data.vectors.nbytes / 1024**2:.0f} MB")

    return data


def train_quantizer(model, training_data):
    """Train quantizer on subset."""
    print_section(f"Training {model.__class__.__name__}")

    print(f"\nTraining on {training_data.vectors.shape[0]:,} vectors...")
    model.fit(training_data.vectors)

    print("✅ Training complete")

    # Compute metrics on training set
    compressed = model.compress(training_data.vectors)
    compression_ratio = model.get_compression_ratio(training_data.vectors)
    distortion = compute_distortion(training_data.vectors, compressed, model)

    print(f"\nTraining Set Metrics:")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    print(f"  Reconstruction distortion: {distortion:.6f}")

    return model


def compress_streaming_batches(model, batch_size=10_000, max_batches=None, output_path=None):
    """Compress full dataset in streaming batches."""
    print_section("Streaming Compression")

    cache_dir = os.path.join(hf_cache_base, "datasets")

    print(f"\nCompressing MS MARCO in batches of {batch_size:,} vectors...")
    print("(Streaming mode - will process full 53M dataset)")

    from datasets import load_dataset

    ds = load_dataset(
        "Cohere/msmarco-v2.1-embed-english-v3",
        "passages",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )

    compressed_batches = []
    total_vectors = 0
    batch_vectors = []

    iterator = tqdm(ds, desc="Processing vectors", unit=" vectors")

    for i, item in enumerate(iterator):
        if max_batches and len(compressed_batches) >= max_batches:
            break

        batch_vectors.append(item['emb'])

        # Process batch when full
        if len(batch_vectors) >= batch_size:
            batch_array = np.array(batch_vectors, dtype=np.float32)
            compressed_batch = model.compress(batch_array)
            compressed_batches.append(compressed_batch)

            total_vectors += len(batch_vectors)
            batch_vectors = []

            # Optionally save batch to disk
            if output_path:
                batch_file = f"{output_path}_batch_{len(compressed_batches):04d}.npy"
                np.save(batch_file, compressed_batch)

    # Process remaining vectors
    if batch_vectors:
        batch_array = np.array(batch_vectors, dtype=np.float32)
        compressed_batch = model.compress(batch_array)
        compressed_batches.append(compressed_batch)
        total_vectors += len(batch_vectors)

        if output_path:
            batch_file = f"{output_path}_batch_{len(compressed_batches):04d}.npy"
            np.save(batch_file, compressed_batch)

    print(f"\n✅ Compressed {total_vectors:,} vectors in {len(compressed_batches)} batches")

    if output_path:
        print(f"   Saved to: {output_path}_batch_*.npy")

    return compressed_batches, total_vectors


def demo_msmarco_streaming(
    training_size=100_000,
    compression_batch_size=10_000,
    max_compression_batches=10,  # Process first 100K vectors as demo
):
    """Demo streaming approach for MS MARCO."""
    from datetime import datetime
    import uuid

    print("\n" + "=" * 70)
    print("  MS MARCO Streaming Compression Demo")
    print("=" * 70)
    print("\nAPPROACH:")
    print("  1. Train on representative subset (100K-1M vectors)")
    print("  2. Compress full dataset in streaming batches")
    print("  3. Save compressed batches to disk")
    print("\nDATASET:")
    print("  • MS MARCO v2.1 (Cohere embeddings)")
    print("  • 53.2M passages, ~1024 dimensions")
    print(f"\nCONFIG:")
    print(f"  • Training subset: {training_size:,} vectors")
    print(f"  • Compression batch size: {compression_batch_size:,}")
    print(f"  • Demo batches: {max_compression_batches} (for testing)")
    print("=" * 70)

    sweep_id = f"msmarco_streaming_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Load training subset
    training_data = load_training_subset(limit=training_size)

    # Create output directory for compressed batches
    output_dir = "compressed_msmarco"
    os.makedirs(output_dir, exist_ok=True)

    # Test with two methods
    configs = [
        {"method": "pq", "model": ProductQuantizer(M=16, B=8), "name": "PQ(M=16, B=8)"},
        {"method": "sq", "model": ScalarQuantizer(), "name": "SQ(8-bit)"},
    ]

    for config in configs:
        print_section(f"Method: {config['name']}")

        # Train
        model = train_quantizer(config["model"], training_data)

        # Compress in batches
        output_path = os.path.join(output_dir, f"{config['method']}_compressed")
        compressed_batches, total_vectors = compress_streaming_batches(
            model,
            batch_size=compression_batch_size,
            max_batches=max_compression_batches,
            output_path=output_path,
        )

        # Compute metrics on first batch
        print("\nComputing metrics on first batch...")
        first_batch = compressed_batches[0]
        # Note: We'd need the original vectors to compute distortion
        # For now, just report compression ratio
        compression_ratio = model.get_compression_ratio(training_data.vectors)

        # Log run
        metrics = {
            "compression_ratio": compression_ratio,
            "training_size": training_size,
            "compressed_vectors": total_vectors,
            "num_batches": len(compressed_batches),
        }

        log_run(
            method=config["method"],
            dataset="msmarco-streaming",
            metrics=metrics,
            config={"batch_size": compression_batch_size},
            sweep_id=sweep_id,
        )

        print(f"✅ {config['name']} complete")
        print(f"   Compression: {compression_ratio:.1f}x")
        print(f"   Compressed: {total_vectors:,} vectors")

    print_section("Demo Complete")
    print(f"\n🔖 Sweep ID: {sweep_id}")
    print(f"📁 Compressed batches: {output_dir}/")
    print(f"📊 Database: logs/benchmark_runs.db")
    print("\nTo process full 53M dataset:")
    print("  1. Remove max_compression_batches limit")
    print("  2. Request large compute node (--mem=8G --tmp=10G)")
    print("  3. Submit as Slurm job (see slurm_msmarco.sh)")
    print()


if __name__ == "__main__":
    # Demo with small numbers for testing
    # Increase these for production runs
    demo_msmarco_streaming(
        training_size=100_000,      # Train on 100K vectors (~500 MB)
        compression_batch_size=10_000,  # Process 10K at a time (~50 MB)
        max_compression_batches=10,     # Only process 100K total (for demo)
    )

    # For full 53M dataset, use:
    # demo_msmarco_streaming(
    #     training_size=1_000_000,     # Train on 1M vectors (~5 GB)
    #     compression_batch_size=10_000,
    #     max_compression_batches=None,  # Process all 53M
    # )
