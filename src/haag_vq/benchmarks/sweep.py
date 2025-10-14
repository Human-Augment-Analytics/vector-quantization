"""
Parameter sweep functionality for vector quantization benchmarks.

This module enables systematic exploration of the compression-distortion trade-off
by running benchmarks across grids of hyperparameters (e.g., number of chunks,
number of clusters for Product Quantization).

The results are automatically logged to the database for later analysis and visualization.
"""

import itertools
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
import typer
import uuid
from datetime import datetime

from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.metrics.pairwise_distortion import compute_pairwise_distortion
from haag_vq.metrics.rank_distortion import compute_rank_distortion
from haag_vq.metrics.recall import evaluate_recall
from haag_vq.metrics.performance import measure_qps, time_compress, time_decompress
from haag_vq.data.datasets import Dataset, load_dummy_dataset, load_huggingface_dataset
from haag_vq.utils.run_logger import log_run


def sweep(
    method: str = typer.Option("pq", help="Compression method: pq, sq, etc."),
    dataset: str = typer.Option("dummy", help="Dataset name: dummy or huggingface"),
    num_samples: int = typer.Option(10000, help="Number of samples for dummy dataset"),
    dim: int = typer.Option(1024, help="Dimensionality for dummy dataset"),
    # PQ-specific sweep parameters
    pq_chunks: str = typer.Option("8,16,32", help="[PQ only] Comma-separated chunk values"),
    pq_clusters: str = typer.Option("128,256", help="[PQ only] Comma-separated cluster values"),
    # SQ-specific sweep parameters (example for future methods)
    sq_bits: str = typer.Option("8", help="[SQ only] Comma-separated bit values (e.g., '4,8,16')"),
    # Evaluation options
    with_recall: bool = typer.Option(True, help="Compute recall metrics"),
    with_pairwise: bool = typer.Option(True, help="Compute pairwise distance distortion"),
    with_rank: bool = typer.Option(True, help="Compute rank distortion"),
    num_pairs: int = typer.Option(1000, help="Number of random pairs for pairwise distortion"),
    rank_k: int = typer.Option(10, help="k for rank distortion (top-k neighbors)"),
):
    """
    Run parameter sweep to generate compression-distortion trade-off curves.

    This command systematically varies hyperparameters and logs all results to the database.
    Use 'vq-benchmark plot' to visualize the trade-offs.

    Examples:
        # Sweep PQ with different chunks and clusters
        vq-benchmark sweep --method pq --chunks "4,8,16" --clusters "128,256,512"

        # Sweep on real embeddings
        vq-benchmark sweep --method pq --dataset huggingface
    """
    # Generate unique sweep ID
    sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print("=" * 70)
    print("  HAAG Vector Quantization - Parameter Sweep")
    print("=" * 70)
    print(f"\n🔖 Sweep ID: {sweep_id}")
    print(f"   Use this ID to filter plots: vq-benchmark plot --sweep-id {sweep_id}")

    # Load dataset
    print(f"\nLoading dataset: {dataset}...")
    if dataset == "dummy":
        data = load_dummy_dataset(num_samples=num_samples, dim=dim)
    elif dataset == "huggingface":
        data = load_huggingface_dataset()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    print(f"Dataset shape: {data.vectors.shape}")

    # Generate parameter grid based on method
    if method == "pq":
        configs = _generate_pq_configs(pq_chunks, pq_clusters)
    elif method == "sq":
        configs = _generate_sq_configs(sq_bits)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: pq, sq")

    print(f"\nRunning {len(configs)} configurations...")
    print("-" * 70)

    # Run each configuration
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")
        _run_single_config(
            method=method,
            dataset=dataset,
            data=data,
            config=config,
            with_recall=with_recall,
            with_pairwise=with_pairwise,
            with_rank=with_rank,
            num_pairs=num_pairs,
            rank_k=rank_k,
            sweep_id=sweep_id,
        )

    print("\n" + "=" * 70)
    print("  Sweep Complete!")
    print("=" * 70)
    print(f"\nRan {len(configs)} configurations.")
    print(f"🔖 Sweep ID: {sweep_id}")
    print("Results logged to: logs/benchmark_runs.db")
    print("\nNext steps:")
    print(f"  • Visualize this sweep only: vq-benchmark plot --sweep-id {sweep_id}")
    print("  • Visualize all results: vq-benchmark plot")
    print("  • Query database: sqlite3 logs/benchmark_runs.db")


def _generate_pq_configs(chunks: str, clusters: str) -> List[Dict[str, Any]]:
    """Generate Product Quantization parameter grid."""
    configs = []
    chunk_values = [int(x.strip()) for x in chunks.split(",")]
    cluster_values = [int(x.strip()) for x in clusters.split(",")]

    for num_chunks, num_clusters in itertools.product(chunk_values, cluster_values):
        configs.append({
            "name": f"PQ(chunks={num_chunks}, clusters={num_clusters})",
            "num_chunks": num_chunks,
            "num_clusters": num_clusters,
        })

    return configs


def _generate_sq_configs(bits: str) -> List[Dict[str, Any]]:
    """
    Generate Scalar Quantization parameter grid.

    Note: Current SQ implementation is fixed at 8-bit.
    This is a placeholder for future multi-bit SQ implementations.
    """
    configs = []
    bit_values = [int(x.strip()) for x in bits.split(",")]

    for num_bits in bit_values:
        if num_bits != 8:
            print(f"  Warning: SQ currently only supports 8-bit. Skipping {num_bits}-bit.")
            continue
        configs.append({
            "name": f"SQ({num_bits}-bit)",
            "num_bits": num_bits,
        })

    # If no valid configs, add default 8-bit
    if not configs:
        configs.append({
            "name": "SQ(8-bit)",
            "num_bits": 8,
        })

    return configs


def _get_codebook_vectors(model: Any) -> Optional[np.ndarray]:
    """Return codebook vectors for QPS measurement without touching disk."""
    if isinstance(model, ProductQuantizer):
        if not getattr(model, "codebooks", None):
            return None
        chunks = [np.asarray(cb, dtype=np.float32) for cb in model.codebooks]
        return np.concatenate(chunks, axis=0)
    if isinstance(model, ScalarQuantizer):
        if model.min is None or model.max is None:
            return None
        return np.stack([model.min, model.max]).astype(np.float32)
    return None


def _run_single_config(
    method: str,
    dataset: str,
    data: Dataset,
    config: Dict[str, Any],
    with_recall: bool,
    with_pairwise: bool,
    with_rank: bool,
    num_pairs: int,
    rank_k: int,
    sweep_id: str = None,
) -> None:
    """Run benchmark for a single configuration."""
    # Create model based on method and config
    if method == "pq":
        model = ProductQuantizer(
            num_chunks=config["num_chunks"],
            num_clusters=config["num_clusters"],
        )
    elif method == "sq":
        # SQ currently has no hyperparameters, but config is logged for consistency
        model = ScalarQuantizer()
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Fit and compress
    X = data.vectors

    fit_start = perf_counter()
    model.fit(X)
    fit_time = perf_counter() - fit_start

    X_compressed, compression_time = time_compress(model, X)
    X_reconstructed, decompression_time = time_decompress(model, X_compressed)
    # 1. Reconstruction distortion (MSE)
    reconstruction_distortion = compute_distortion(X, X_compressed, model)
    compression_ratio = model.get_compression_ratio(X)
    quantization_time = fit_time + compression_time

    metrics: Dict[str, Any] = {
        "reconstruction_distortion": reconstruction_distortion,
        "compression_ratio": compression_ratio,
        "fit_latency_ms": fit_time * 1000.0,
        "compression_latency_ms": compression_time * 1000.0,
        "decompression_latency_ms": decompression_time * 1000.0,
        "quantization_latency_ms": quantization_time * 1000.0,
    }

    # 2. Pairwise distance distortion
    if with_pairwise:
        pairwise_dist = compute_pairwise_distortion(
            X, X_compressed, model, num_pairs=num_pairs,
        )
        metrics["pairwise_distortion_mean"] = pairwise_dist["mean"]
        metrics["pairwise_distortion_median"] = pairwise_dist["median"]
        metrics["pairwise_distortion_max"] = pairwise_dist["max"]
    else:
        pairwise_dist = None
    # 3. Rank distortion
    if with_rank:
        rank_dist = compute_rank_distortion(data, model, k=rank_k)
        metrics[f"rank_distortion@{rank_k}"] = rank_dist
    else:
        rank_dist = None

    #4. Speed metrics
    codebook_vectors = _get_codebook_vectors(model)
    qps_metrics = None
    qps_metrics = measure_qps(
                data.queries,
                model=model,
                codebook_vectors=codebook_vectors,
            )
    metrics.update(qps_metrics)

    # 5. Recall
    if with_recall:
        recall_metrics = evaluate_recall(data, model, num_queries=100)
        metrics.update(recall_metrics)

    # Log to database
    log_run(method=method, dataset=dataset, metrics=metrics, config=config, sweep_id=sweep_id)

    # Print summary
    print(f"  Compression ratio:           {compression_ratio:.2f}x")
    print(f"  Reconstruction MSE:          {reconstruction_distortion:.4f}")
    print(f"  Fit latency (ms):            {metrics['fit_latency_ms']:.2f}")
    print(f"  Compression latency (ms):    {metrics['compression_latency_ms']:.2f}")
    print(f"  Quantization latency (ms):   {metrics['quantization_latency_ms']:.2f}")
    print(f"  Decompression latency (ms):  {metrics['decompression_latency_ms']:.2f}")

    if with_pairwise:
        print(f"  Pairwise distortion (mean):  {pairwise_dist['mean']:.4f}")
    if with_rank:
        print(f"  Rank distortion@{rank_k}:       {rank_dist:.4f}")
    if "qps" in metrics:
        print(f"  QPS:                         {metrics['qps']:.2f}")
        if "avg_query_latency_ms" in metrics:
            print(f"  Avg query latency (ms):     {metrics['avg_query_latency_ms']:.4f}")
    if with_recall:
        recall_key = "recall@10"
        if recall_key in metrics:
            print(f"  Recall@10:                  {metrics[recall_key]:.4f}")

