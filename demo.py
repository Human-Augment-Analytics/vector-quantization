#!/usr/bin/env python3
"""
Comprehensive demo of HAAG vector quantization benchmarking framework.

Demonstrates:
1. Individual method runs with all metrics
2. Parameter sweeps to explore compression-distortion trade-offs
3. Visualization of results

NEW FEATURES:
- Pairwise distance distortion (how well distances are preserved)
- Rank distortion (fraction of wrong top-k neighbors)
- Parameter sweeps for systematic exploration
- Automated plotting of trade-off curves
- All metrics explained in METRICS_GUIDE.md
"""

import os
# Suppress tokenizers parallelism warning when using git subprocess calls
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from haag_vq.data.datasets import load_dummy_dataset, load_huggingface_dataset
from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.metrics.pairwise_distortion import compute_pairwise_distortion
from haag_vq.metrics.rank_distortion import compute_rank_distortion
from haag_vq.metrics.recall import evaluate_recall
from haag_vq.utils.run_logger import log_run


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_demo_quantizer(name, quantizer, data, with_all_metrics=True):
    print(f"\n--- {name} ---")

    # Fit and compress
    print(f"Fitting {name}...")
    quantizer.fit(data.vectors)
    compressed = quantizer.compress(data.vectors)

    # 1. Reconstruction metrics
    distortion = compute_distortion(data.vectors, compressed, quantizer)
    compression_ratio = quantizer.get_compression_ratio(data.vectors)

    print(f"  Compression ratio:    {compression_ratio:.2f}x")
    print(f"  Reconstruction MSE:   {distortion:.4f}")

    if with_all_metrics:
        # 2. Pairwise distance preservation
        pairwise = compute_pairwise_distortion(data.vectors, compressed, quantizer, num_pairs=500)
        print(f"  Pairwise distortion:  {pairwise['mean']:.4f} (mean), {pairwise['max']:.4f} (max)")

        # 3. Rank distortion
        rank_dist = compute_rank_distortion(data, quantizer, k=10, num_queries=50)
        print(f"  Rank distortion@10:   {rank_dist:.4f} ({rank_dist*100:.1f}% wrong neighbors)")

        # 4. Recall
        recall_metrics = evaluate_recall(data, quantizer, num_queries=50)
        print(f"  Recall@10:            {recall_metrics['recall@10']:.4f}")


def demo_synthetic():
    print_section("DEMO 1: Synthetic Gaussian Data")

    print("\nGenerating 10,000 random 128-dimensional vectors...")
    data = load_dummy_dataset(num_samples=10000, dim=128, seed=42)
    print(f"Dataset shape: {data.vectors.shape}")

    # Product Quantization
    pq = ProductQuantizer(num_chunks=8, num_clusters=256)
    run_demo_quantizer("Product Quantization (8 chunks, 256 clusters)", pq, data)

    # Scalar Quantization
    sq = ScalarQuantizer()
    run_demo_quantizer("Scalar Quantization (8-bit)", sq, data)


def demo_huggingface():
    print_section("DEMO 2: Real Text Embeddings (Hugging Face)")

    print("\nLoading STS-B dataset with Sentence-BERT embeddings...")
    print("(This may take a moment on first run...)")

    try:
        data = load_huggingface_dataset(
            dataset_name="stsb_multi_mt",
            config_name="en",
            model_name="all-MiniLM-L6-v2",
            split="train"
        )
        print(f"Dataset shape: {data.vectors.shape}")

        # Product Quantization
        pq = ProductQuantizer(num_chunks=8, num_clusters=256)
        run_demo_quantizer("Product Quantization (8 chunks, 256 clusters)", pq, data)

        # Scalar Quantization
        sq = ScalarQuantizer()
        run_demo_quantizer("Scalar Quantization (8-bit)", sq, data)

    except Exception as e:
        print(f"\n⚠️  Hugging Face demo failed: {e}")
        print("Make sure you have 'datasets' and 'sentence-transformers' installed:")
        print("  pip install datasets sentence-transformers")


def demo_parameter_sweep():
    """Demonstrate parameter sweep functionality."""
    import uuid
    from datetime import datetime

    print_section("DEMO 3: Parameter Sweep for Trade-off Analysis")

    # Generate unique sweep ID for this demo run
    sweep_id = f"demo_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    print(f"\n🔖 Sweep ID: {sweep_id}")
    print(f"   This ID will be used to filter plots for this specific sweep")

    print("\nRunning PQ and SQ with varying configurations to explore trade-offs...")
    print("\nProduct Quantization (PQ) Parameters:")
    print("  • num_chunks: How many pieces to split each vector into")
    print("  • num_clusters: How many representative vectors per chunk")
    print("\nScalar Quantization (SQ) Parameters:")
    print("  • num_bits: Quantization precision (4, 8, or 16 bits)")
    print("\nTrade-offs:")
    print("  PQ: More chunks = higher compression but more distortion")
    print("  SQ: Fewer bits = higher compression but less precision")
    print()

    # Use smaller dataset for faster demo
    data = load_dummy_dataset(num_samples=5000, dim=128, seed=42)

    # Sweep over different quantization methods and configurations
    # Each config explores a different point in the compression-quality trade-off
    all_configs = [
        # Product Quantization configs
        {"method": "pq", "num_chunks": 4, "num_clusters": 128, "name": "PQ(4 chunks, 128 clusters)"},
        {"method": "pq", "num_chunks": 8, "num_clusters": 128, "name": "PQ(8 chunks, 128 clusters)"},
        {"method": "pq", "num_chunks": 8, "num_clusters": 256, "name": "PQ(8 chunks, 256 clusters)"},
        {"method": "pq", "num_chunks": 16, "num_clusters": 256, "name": "PQ(16 chunks, 256 clusters)"},
        # Scalar Quantization with different bit depths (for comparison)
        {"method": "sq", "num_bits": 4, "name": "SQ(4-bit)"},
        {"method": "sq", "num_bits": 8, "name": "SQ(8-bit)"},
        {"method": "sq", "num_bits": 16, "name": "SQ(16-bit)"},
    ]

    print(f"Testing {len(all_configs)} different configurations (PQ + SQ):\n")

    for i, config in enumerate(all_configs, 1):
        method = config.pop("method")
        name = config.pop("name")
        print(f"[{i}/{len(all_configs)}] {name}")

        # Create and train model
        if method == "pq":
            model = ProductQuantizer(num_chunks=config["num_chunks"], num_clusters=config["num_clusters"])
        elif method == "sq":
            model = ScalarQuantizer(num_bits=config["num_bits"])

        model.fit(data.vectors)
        compressed = model.compress(data.vectors)

        # Compute all metrics
        metrics = {
            "compression_ratio": model.get_compression_ratio(data.vectors),
            "reconstruction_distortion": compute_distortion(data.vectors, compressed, model),
            "pairwise_distortion_mean": compute_pairwise_distortion(data.vectors, compressed, model, num_pairs=200)["mean"],
            "rank_distortion@10": compute_rank_distortion(data, model, k=10, num_queries=30),
            "recall@10": evaluate_recall(data, model, num_queries=30)["recall@10"],
        }

        # Log to database (restore method to config for logging)
        config_for_log = dict(config)
        log_run(method=method, dataset="demo_sweep", metrics=metrics, config=config_for_log, sweep_id=sweep_id)

        # Print summary
        print(f"  Compression: {metrics['compression_ratio']:.1f}x | "
              f"MSE: {metrics['reconstruction_distortion']:.3f} | "
              f"Pairwise: {metrics['pairwise_distortion_mean']:.3f} | "
              f"Recall@10: {metrics['recall@10']:.3f}")

    print(f"\n✅ Sweep complete! Results logged to database.")
    print(f"   🔖 Sweep ID: {sweep_id}")
    print(f"   Run 'vq-benchmark plot --sweep-id {sweep_id}' to visualize this sweep.")

    # Return the sweep_id so demo_visualization can use it
    return sweep_id


def demo_visualization(sweep_id=None):
    """Demonstrate visualization capabilities.

    Args:
        sweep_id: Optional sweep ID to filter plots to
    """
    print_section("DEMO 4: Visualization of Results")

    print("\nGenerating plots from logged benchmark runs...")
    if sweep_id:
        print(f"   Filtering to sweep: {sweep_id}")

    # Check if we have any runs
    import sqlite3
    db_path = "logs/benchmark_runs.db"

    if not os.path.exists(db_path):
        print("⚠️  No benchmark database found. Run the sweep demo first!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if sweep_id:
        cursor.execute("SELECT COUNT(*) FROM runs WHERE sweep_id = ?", (sweep_id,))
    else:
        cursor.execute("SELECT COUNT(*) FROM runs")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        if sweep_id:
            print(f"⚠️  No runs found for sweep {sweep_id}!")
        else:
            print("⚠️  No runs in database. Run the sweep demo first!")
        return

    print(f"Found {count} benchmark runs in database.")

    # Generate plots
    try:
        from haag_vq.visualization.plot import (_load_runs_from_db, _plot_compression_distortion,
                                                 _plot_pairwise_distortion, _plot_rank_distortion, _plot_recall)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from datetime import datetime

        runs = _load_runs_from_db(db_path, sweep_id=sweep_id)

        # Create timestamped output directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir_combined = f"demo_plots/{timestamp}/combined"
        output_dir_separate = f"demo_plots/{timestamp}/separate"
        os.makedirs(output_dir_combined, exist_ok=True)
        os.makedirs(output_dir_separate, exist_ok=True)

        print(f"\nGenerating COMBINED plots in {output_dir_combined}/...")
        _plot_compression_distortion(runs, output_dir_combined, "png", 150, separate_methods=False)
        _plot_pairwise_distortion(runs, output_dir_combined, "png", 150, separate_methods=False)
        _plot_rank_distortion(runs, output_dir_combined, "png", 150, separate_methods=False)
        _plot_recall(runs, output_dir_combined, "png", 150, separate_methods=False)
        print(f"  ✅ Combined plots generated")

        print(f"\nGenerating SEPARATE plots in {output_dir_separate}/...")
        _plot_compression_distortion(runs, output_dir_separate, "png", 150, separate_methods=True)
        _plot_pairwise_distortion(runs, output_dir_separate, "png", 150, separate_methods=True)
        _plot_rank_distortion(runs, output_dir_separate, "png", 150, separate_methods=True)
        _plot_recall(runs, output_dir_separate, "png", 150, separate_methods=True)
        print(f"  ✅ Separate plots generated (one per method)")

        print("\n📊 Plots generated successfully!")
        print(f"   Combined: {output_dir_combined}/")
        print(f"   Separate: {output_dir_separate}/")

    except ImportError:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")


def main():
    print("\n" + "=" * 70)
    print("  HAAG Vector Quantization Benchmarking Framework")
    print("  Georgia Tech CS 8903 - Comprehensive Demo")
    print("=" * 70)
    print("\nFEATURES DEMONSTRATED:")
    print("  • All quantization metrics (reconstruction, pairwise, rank)")
    print("  • Parameter sweeps for systematic exploration")
    print("  • Automated visualization of trade-offs")
    print("  • See METRICS_GUIDE.md for detailed explanations")
    print("=" * 70)

    # Demo 1: Synthetic data with single config
    demo_synthetic()

    # Demo 2: Real embeddings
    demo_huggingface()

    # Demo 3: Parameter sweep
    sweep_id = demo_parameter_sweep()

    # Demo 4: Visualization (using the sweep_id from the parameter sweep)
    demo_visualization(sweep_id=sweep_id)

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nWhat You Learned:")
    print("  1. How to run benchmarks with all metrics")
    print("  2. How to sweep parameters for trade-off analysis")
    print("  3. How to visualize results")
    print("\nNext Steps:")
    print("  • Read METRICS_GUIDE.md to understand the metrics")
    print("  • Check demo_plots/ for generated visualizations")
    print("  • Run custom sweeps: vq-benchmark sweep --help")
    print("  • Query results: sqlite3 logs/benchmark_runs.db")
    print("  • Add new quantization methods by inheriting BaseQuantizer")
    print()


if __name__ == "__main__":
    main()