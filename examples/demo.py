#!/usr/bin/env python3
"""
Comprehensive demo of HAAG vector quantization benchmarking framework.

Demonstrates:
1. Complete parameter sweep across ALL 5 implemented methods (PQ, OPQ, SQ, SAQ, RaBitQ)
2. DBpedia 100K dataset (real pre-embedded data)
3. All quantization metrics (reconstruction, pairwise, rank, recall)
4. Automated visualization of results

NEW FEATURES:
- All 5 quantization methods: PQ, OPQ, SQ, SAQ, RaBitQ
- Real dataset: DBpedia 100K (1536-dim OpenAI embeddings)
- Comprehensive metrics and visualizations
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from haag_vq.data import load_dbpedia_openai_1536_100k
from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.optimized_product_quantization import OptimizedProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.methods.saq import SAQ
from haag_vq.methods.rabit_quantization import RaBitQuantizer
from haag_vq.utils.faiss_utils import MetricType
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.metrics.pairwise_distortion import compute_pairwise_distortion
from haag_vq.metrics.rank_distortion import compute_rank_distortion
from haag_vq.metrics.recall import evaluate_recall
from haag_vq.utils.run_logger import log_run


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_dataset():
    """Load DBpedia 100K dataset."""
    print_section("Loading DBpedia 100K Dataset")

    print("\nLoading DBpedia 100K (1536-dim OpenAI embeddings)...")
    print("(This will auto-download on first run to ../datasets/)")

    data = load_dbpedia_openai_1536_100k(
        cache_dir="../datasets",
        limit=None,  # Use 10000 for faster demo
    )

    print(f"✅ Dataset loaded: {data.vectors.shape}")
    print(f"   Vectors: {data.vectors.shape[0]:,}")
    print(f"   Dimensions: {data.vectors.shape[1]}")
    print(f"   Queries: {data.queries.shape[0]}")

    # Compute ground truth if not already present (needed for recall/rank metrics)
    if data.ground_truth is None:
        print("\n⏳ Computing ground truth (k-nearest neighbors)...")
        print("   This may take a minute for 100K vectors...")
        from sklearn.metrics.pairwise import pairwise_distances
        dist_matrix = pairwise_distances(data.queries, data.vectors, metric='euclidean')
        data.ground_truth = dist_matrix.argsort(axis=1)
        print("✅ Ground truth computed")

    return data


def demo_complete_sweep():
    """Run comprehensive parameter sweep across all 5 methods."""
    import uuid
    from datetime import datetime

    print_section("Complete Parameter Sweep: All 5 Methods")

    # Generate unique sweep ID
    sweep_id = f"dbpedia_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    print(f"\n🔖 Sweep ID: {sweep_id}")

    # Load dataset
    data = load_dataset()

    # Define comprehensive sweep configurations
    # For 1536 dimensions, M must divide evenly: 8, 12, 16, 24, 32, 48
    all_configs = [
        # PRODUCT QUANTIZATION (PQ) - Baseline subspace quantization
        {"method": "pq", "M": 8, "B": 8, "name": "PQ(M=8, B=8)"},
        {"method": "pq", "M": 16, "B": 8, "name": "PQ(M=16, B=8)"},
        {"method": "pq", "M": 32, "B": 8, "name": "PQ(M=32, B=8)"},
        {"method": "pq", "M": 16, "B": 6, "name": "PQ(M=16, B=6)"},

        # OPTIMIZED PRODUCT QUANTIZATION (OPQ) - PQ with learned rotation
        {"method": "opq", "M": 8, "B": 8, "name": "OPQ(M=8, B=8)"},
        {"method": "opq", "M": 16, "B": 8, "name": "OPQ(M=16, B=8)"},
        {"method": "opq", "M": 32, "B": 8, "name": "OPQ(M=32, B=8)"},

        # SCALAR QUANTIZATION (SQ) - Per-dimension quantization
        {"method": "sq", "name": "SQ(4-bit)"},
        {"method": "sq", "name": "SQ(8-bit)"},

        # SEGMENTED CAQ (SAQ) - Adaptive bit allocation
        {"method": "saq", "num_bits": 4, "name": "SAQ(4 bits/dim)"},
        {"method": "saq", "num_bits": 6, "name": "SAQ(6 bits/dim)"},
        {"method": "saq", "total_bits": 3072, "name": "SAQ(3072 total bits)"},
        {"method": "saq", "total_bits": 6144, "name": "SAQ(6144 total bits)"},

        # RaBitQ - Extreme compression with theoretical bounds
        {"method": "rabitq", "metric_type": "L2", "name": "RaBitQ(L2)"},
    ]

    print(f"\n🚀 Running {len(all_configs)} configurations across 5 methods:")
    print("   • PQ (4 configs)")
    print("   • OPQ (3 configs)")
    print("   • SQ (2 configs)")
    print("   • SAQ (4 configs)")
    print("   • RaBitQ (1 config)")
    print(f"\nDataset: DBpedia 100K ({data.vectors.shape[0]:,} vectors, {data.vectors.shape[1]} dims)")
    print()

    for i, config in enumerate(all_configs, 1):
        method = config.pop("method")
        name = config.pop("name")
        print(f"[{i}/{len(all_configs)}] {name}")

        try:
            # Create model based on method
            if method == "pq":
                model = ProductQuantizer(M=config["M"], B=config["B"])
            elif method == "opq":
                model = OptimizedProductQuantizer(M=config["M"], B=config["B"])
            elif method == "sq":
                model = ScalarQuantizer()
            elif method == "saq":
                if "total_bits" in config:
                    model = SAQ(total_bits=config["total_bits"], allowed_bits=[0, 2, 4, 6, 8])
                else:
                    model = SAQ(num_bits=config["num_bits"])
            elif method == "rabitq":
                model = RaBitQuantizer(metric_type=MetricType.L2)

            # Train model
            model.fit(data.vectors)
            compressed = model.compress(data.vectors)

            # Compute all metrics
            metrics = {
                "compression_ratio": model.get_compression_ratio(data.vectors),
                "reconstruction_distortion": compute_distortion(data.vectors, compressed, model),
                "pairwise_distortion_mean": compute_pairwise_distortion(data.vectors, compressed, model, num_pairs=500)["mean"],
                "rank_distortion@10": compute_rank_distortion(data, model, k=10, num_queries=100),
                "recall@10": evaluate_recall(data, model, num_queries=100)["recall@10"],
            }

            # Log to database
            log_run(method=method, dataset="dbpedia-100k", metrics=metrics, config=config, sweep_id=sweep_id)

            # Print summary
            print(f"  ✅ Compression: {metrics['compression_ratio']:.1f}x | "
                  f"MSE: {metrics['reconstruction_distortion']:.4f} | "
                  f"Pairwise: {metrics['pairwise_distortion_mean']:.4f} | "
                  f"Rank@10: {metrics['rank_distortion@10']:.4f} | "
                  f"Recall@10: {metrics['recall@10']:.4f}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Sweep complete! Results logged to database.")
    print(f"   🔖 Sweep ID: {sweep_id}")

    return sweep_id


def demo_visualization(sweep_id):
    """Generate comprehensive visualizations."""
    print_section("Generating Visualizations")

    print("\nCreating plots from benchmark results...")
    print(f"   Filtering to sweep: {sweep_id}")

    # Check database
    import sqlite3
    db_path = "logs/benchmark_runs.db"

    if not os.path.exists(db_path):
        print("⚠️  No benchmark database found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM runs WHERE sweep_id = ?", (sweep_id,))
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        print(f"⚠️  No runs found for sweep {sweep_id}!")
        return

    print(f"Found {count} benchmark runs.")

    # Generate plots
    try:
        from haag_vq.visualization.plot import (_load_runs_from_db, _plot_compression_distortion,
                                                 _plot_pairwise_distortion, _plot_rank_distortion, _plot_recall)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from datetime import datetime

        runs = _load_runs_from_db(db_path, sweep_id=sweep_id)

        # Create output directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir_combined = f"demo_plots/{timestamp}/combined"
        output_dir_separate = f"demo_plots/{timestamp}/separate"
        os.makedirs(output_dir_combined, exist_ok=True)
        os.makedirs(output_dir_separate, exist_ok=True)

        print(f"\nGenerating COMBINED plots (all methods on one plot)...")
        _plot_compression_distortion(runs, output_dir_combined, "png", 300, separate_methods=False)
        _plot_pairwise_distortion(runs, output_dir_combined, "png", 300, separate_methods=False)
        _plot_rank_distortion(runs, output_dir_combined, "png", 300, separate_methods=False)
        _plot_recall(runs, output_dir_combined, "png", 300, separate_methods=False)
        print(f"  ✅ Combined plots: {output_dir_combined}/")

        print(f"\nGenerating SEPARATE plots (one per method)...")
        _plot_compression_distortion(runs, output_dir_separate, "png", 300, separate_methods=True)
        _plot_pairwise_distortion(runs, output_dir_separate, "png", 300, separate_methods=True)
        _plot_rank_distortion(runs, output_dir_separate, "png", 300, separate_methods=True)
        _plot_recall(runs, output_dir_separate, "png", 300, separate_methods=True)
        print(f"  ✅ Separate plots: {output_dir_separate}/")

        print("\n📊 Visualizations complete!")
        print(f"\n   Combined plots: {output_dir_combined}/")
        print(f"      • compression_distortion_tradeoff.png")
        print(f"      • pairwise_distortion.png")
        print(f"      • rank_distortion.png")
        print(f"      • recall_comparison.png")
        print(f"\n   Separate plots: {output_dir_separate}/")
        print(f"      • One set per method (pq, opq, sq, saq, rabitq)")

    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        print("Install matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 70)
    print("  HAAG Vector Quantization - Complete Benchmark Demo")
    print("  DBpedia 100K Dataset | All 5 Methods")
    print("=" * 70)
    print("\nMETHODS BENCHMARKED:")
    print("  1. PQ (Product Quantization) - Baseline subspace quantization")
    print("  2. OPQ (Optimized PQ) - PQ with learned rotation")
    print("  3. SQ (Scalar Quantization) - Per-dimension quantization")
    print("  4. SAQ (Segmented CAQ) - Adaptive bit allocation")
    print("  5. RaBitQ - Extreme compression with theoretical bounds")
    print("\nDATASET:")
    print("  • DBpedia 100K entities (1536-dim OpenAI embeddings)")
    print("  • Pre-embedded, auto-downloaded from HuggingFace")
    print("\nMETRICS:")
    print("  • Compression ratio")
    print("  • Reconstruction distortion (MSE)")
    print("  • Pairwise distance distortion")
    print("  • Rank distortion (neighbor accuracy)")
    print("  • Recall@10 (retrieval quality)")
    print("=" * 70)

    # Run complete sweep
    sweep_id = demo_complete_sweep()

    # Generate visualizations
    demo_visualization(sweep_id)

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nWhat Was Done:")
    print("  ✅ Loaded DBpedia 100K dataset (real OpenAI embeddings)")
    print("  ✅ Benchmarked all 5 quantization methods")
    print("  ✅ Computed comprehensive metrics")
    print("  ✅ Generated visualizations")
    print(f"\nResults:")
    print(f"  📁 Database: logs/benchmark_runs.db")
    print(f"  📊 Plots: demo_plots/")
    print(f"  🔖 Sweep ID: {sweep_id}")
    print("\nNext Steps:")
    print("  • View plots in demo_plots/")
    print("  • Query database: sqlite3 logs/benchmark_runs.db")
    print("  • Run CLI sweeps: vq-benchmark sweep --help")
    print(f"  • Filter plots: vq-benchmark plot --sweep-id {sweep_id}")
    print("  • Read docs: documentation/METHODS.md")
    print()


if __name__ == "__main__":
    main()
