"""
Streaming batch compression for large datasets (e.g., MS MARCO 53M).

This module implements true streaming compression where:
1. Train quantizer on a subset (e.g., 1M vectors)
2. Stream full dataset in batches
3. Compress each batch
4. Save compressed batches to disk
5. Evaluate on compressed data

This allows benchmarking on datasets that don't fit in memory.
"""

import os
from pathlib import Path
from typing import Optional
import uuid
from datetime import datetime

import numpy as np
import typer
from tqdm import tqdm

from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.optimized_product_quantization import OptimizedProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.methods.saq import SAQ
from haag_vq.methods.rabit_quantization import RaBitQuantizer
from haag_vq.utils.faiss_utils import MetricType
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.utils.run_logger import log_run

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def streaming_sweep(
    method: str = typer.Option("pq", help="Compression method: pq, opq, sq, saq, rabitq"),
    dataset: str = typer.Option("cohere-msmarco", help="Dataset (currently only cohere-msmarco supported)"),
    training_size: int = typer.Option(1_000_000, help="Number of vectors to use for training quantizer"),
    batch_size: int = typer.Option(10_000, help="Batch size for streaming compression"),
    max_batches: Optional[int] = typer.Option(None, help="Max batches to compress (None = all ~53M)"),
    cache_dir: str = typer.Option("../datasets", help="HuggingFace cache directory"),
    output_dir: str = typer.Option("compressed_msmarco", help="Directory to save compressed batches"),
    # Method-specific parameters
    pq_subquantizers: str = typer.Option("16", help="[PQ] M value"),
    pq_bits: str = typer.Option("8", help="[PQ] B value"),
    opq_quantizers: str = typer.Option("16", help="[OPQ] M value"),
    opq_bits: str = typer.Option("8", help="[OPQ] B value"),
    saq_num_bits: str = typer.Option("4", help="[SAQ] num_bits"),
    db_path: str = typer.Option(None, help="SQLite database path"),
):
    """
    Run streaming batch compression on full MS MARCO 53M dataset.

    This trains a quantizer on a subset, then streams and compresses the full dataset in batches.

    Example:
        vq-benchmark streaming-sweep --method pq --training-size 1000000 --batch-size 10000
    """
    if not HF_AVAILABLE:
        raise RuntimeError("datasets library required: pip install datasets")

    if dataset != "cohere-msmarco":
        raise ValueError("Currently only cohere-msmarco is supported for streaming")

    sweep_id = f"streaming_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print("=" * 70)
    print("  Streaming Batch Compression")
    print("=" * 70)
    print(f"Sweep ID: {sweep_id}")
    print(f"Method: {method}")
    print(f"Training size: {training_size:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Max batches: {max_batches or 'ALL (~5300 batches for 53M)'}")
    print("=" * 70)

    # Create output directory
    output_path = Path(output_dir) / sweep_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load training subset
    print(f"\n[1/4] Loading {training_size:,} vectors for training...")
    ds = load_dataset(
        "Cohere/msmarco-v2.1-embed-english-v3",
        "passages",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )

    # Load training vectors
    training_vectors = []
    for i, item in enumerate(tqdm(ds, total=training_size, desc="Loading training set")):
        if i >= training_size:
            break
        training_vectors.append(item['emb'])

    training_vectors = np.array(training_vectors, dtype=np.float32)
    dimension = training_vectors.shape[1]
    print(f"✓ Loaded {len(training_vectors):,} vectors, {dimension} dims")

    # Step 2: Train quantizer
    print(f"\n[2/4] Training {method} quantizer...")

    if method == "pq":
        M = int(pq_subquantizers)
        B = int(pq_bits)
        model = ProductQuantizer(M=M, B=B)
        config = {"M": M, "B": B}
    elif method == "opq":
        M = int(opq_quantizers)
        B = int(opq_bits)
        model = OptimizedProductQuantizer(M=M, B=B)
        config = {"M": M, "B": B}
    elif method == "sq":
        model = ScalarQuantizer()
        config = {}
    elif method == "saq":
        model = SAQ(num_bits=int(saq_num_bits))
        config = {"num_bits": int(saq_num_bits)}
    elif method == "rabitq":
        model = RaBitQuantizer(metric_type=MetricType.L2)
        config = {}
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(training_vectors)
    print(f"✓ Quantizer trained")

    # Step 3: Stream and compress full dataset
    print(f"\n[3/4] Streaming and compressing full dataset in batches...")

    # Reload dataset for streaming compression
    ds = load_dataset(
        "Cohere/msmarco-v2.1-embed-english-v3",
        "passages",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )

    batch_count = 0
    total_compressed = 0
    batch_vectors = []

    iterator = iter(ds)

    while True:
        # Collect batch
        batch_vectors = []
        try:
            for _ in range(batch_size):
                item = next(iterator)
                batch_vectors.append(item['emb'])
        except StopIteration:
            if not batch_vectors:
                break  # Done

        if not batch_vectors:
            break

        # Compress batch
        batch_array = np.array(batch_vectors, dtype=np.float32)
        compressed_batch = model.compress(batch_array)

        # Save compressed batch
        batch_file = output_path / f"batch_{batch_count:05d}.npz"
        np.savez_compressed(batch_file, compressed=compressed_batch)

        total_compressed += len(batch_vectors)
        batch_count += 1

        if batch_count % 100 == 0:
            print(f"  Compressed {batch_count} batches ({total_compressed:,} vectors)")

        # Check if we've reached max batches
        if max_batches and batch_count >= max_batches:
            print(f"  Reached max batches limit ({max_batches})")
            break

    print(f"✓ Compressed {total_compressed:,} vectors in {batch_count} batches")
    print(f"✓ Saved to: {output_path}")

    # Step 4: Compute metrics on training set
    print(f"\n[4/4] Computing metrics on training set...")
    compressed_training = model.compress(training_vectors)

    metrics = {
        "compression_ratio": model.get_compression_ratio(training_vectors),
        "mse": compute_distortion(training_vectors, compressed_training, model),
        "total_vectors_compressed": total_compressed,
        "num_batches": batch_count,
    }

    # Log to database
    log_run(
        method=method,
        dataset=f"{dataset}-streaming",
        metrics=metrics,
        config=config,
        sweep_id=sweep_id,
    )

    print("\n" + "=" * 70)
    print("✓ Streaming compression complete!")
    print(f"  Sweep ID: {sweep_id}")
    print(f"  Compression ratio: {metrics['compression_ratio']:.1f}x")
    print(f"  MSE (on training set): {metrics['mse']:.6f}")
    print(f"  Total vectors compressed: {total_compressed:,}")
    print(f"  Compressed batches saved to: {output_path}")
    print("=" * 70)

    return sweep_id


if __name__ == "__main__":
    typer.run(streaming_sweep)
