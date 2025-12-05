"""
Visualization tools for benchmark results.

This module provides CLI commands to generate plots from logged benchmark runs,
enabling analysis of compression-distortion trade-offs and method comparisons.
"""

import sqlite3
import json
from typing import List, Dict, Optional
import typer


def plot(
    output: str = typer.Option("plots", help="Output directory for plots"),
    db_path: str = typer.Option("logs/benchmark_runs.db", help="Path to benchmark database"),
    format: str = typer.Option("png", help="Plot format: png, pdf, svg"),
    dpi: int = typer.Option(300, help="DPI for raster formats"),
    separate_methods: bool = typer.Option(False, help="Create separate plots for each method instead of combined"),
    sweep_id: Optional[str] = typer.Option(None, help="Filter results to only this sweep ID"),
):
    """
    Generate visualization plots from logged benchmark runs.

    Creates compression-distortion trade-off curves and method comparison plots.

    Examples:
        # Generate all plots
        vq-benchmark plot

        # Custom output directory
        vq-benchmark plot --output results/figures

        # High-res PDF for papers
        vq-benchmark plot --format pdf --dpi 600
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install with: pip install matplotlib")
        raise typer.Exit(1)

    import os
    from datetime import datetime

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_with_timestamp = f"{output}/{timestamp}"
    os.makedirs(output_with_timestamp, exist_ok=True)

    print("=" * 70)
    print("  HAAG Vector Quantization - Results Visualization")
    print("=" * 70)

    # Load data from database
    print(f"\nLoading results from: {db_path}")
    if sweep_id:
        print(f"Filtering by sweep ID: {sweep_id}")
    runs = _load_runs_from_db(db_path, sweep_id=sweep_id)

    if not runs:
        if sweep_id:
            print(f"No benchmark runs found for sweep ID: {sweep_id}")
        else:
            print("No benchmark runs found in database!")
        print("Run benchmarks first: vq-benchmark run or vq-benchmark sweep")
        raise typer.Exit(1)

    print(f"Loaded {len(runs)} benchmark runs")

    # Generate plots
    print("\nGenerating plots...")

    mode = "separate" if separate_methods else "combined"
    print(f"Plot mode: {mode}")

    # 1. Compression-Distortion Trade-off
    _plot_compression_distortion(runs, output_with_timestamp, format, dpi, separate_methods)

    # 2. Pairwise Distance Distortion
    _plot_pairwise_distortion(runs, output_with_timestamp, format, dpi, separate_methods)

    # 3. Rank Distortion vs Compression
    _plot_rank_distortion(runs, output_with_timestamp, format, dpi, separate_methods)

    # 4. Recall comparison
    _plot_recall(runs, output_with_timestamp, format, dpi, separate_methods)

    # 5. Method comparison table
    _generate_comparison_table(runs, output_with_timestamp)

    # 6. Pareto frontier plot
    _plot_pareto_frontier(runs, output_with_timestamp, format, dpi)

    # 7. Multi-dimensional radar chart
    _plot_radar_chart(runs, output_with_timestamp, format, dpi)

    print("\n" + "=" * 70)
    print("  Plots Generated Successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_with_timestamp}/")
    print("Files created:")
    if separate_methods:
        print(f"  • compression_distortion_tradeoff_<method>.{format} (per method)")
        print(f"  • pairwise_distortion_<method>.{format} (per method)")
        print(f"  • rank_distortion_<method>.{format} (per method)")
        print(f"  • recall_comparison_<method>.{format} (per method)")
    else:
        print(f"  • compression_distortion_tradeoff.{format}")
        print(f"  • pairwise_distortion.{format}")
        print(f"  • rank_distortion.{format}")
        print(f"  • recall_comparison.{format}")
    print(f"  • pareto_frontier.{format}")
    print(f"  • method_radar_comparison.{format}")
    print(f"  • comparison_table.txt")


def _load_runs_from_db(db_path: str, sweep_id: Optional[str] = None) -> List[Dict]:
    """Load benchmark runs from SQLite database, optionally filtered by sweep_id."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if sweep_id:
        cursor.execute("""
            SELECT method, dataset, metrics_json, cli_command, timestamp, sweep_id
            FROM runs
            WHERE sweep_id = ?
            ORDER BY timestamp DESC
        """, (sweep_id,))
    else:
        cursor.execute("""
            SELECT method, dataset, metrics_json, cli_command, timestamp, sweep_id
            FROM runs
            ORDER BY timestamp DESC
        """)

    runs = []
    for row in cursor.fetchall():
        method, dataset, metrics_json, cli_command, timestamp, run_sweep_id = row
        metrics = json.loads(metrics_json)

        runs.append({
            "method": method,
            "dataset": dataset,
            "metrics": metrics,
            "cli_command": cli_command,
            "timestamp": timestamp,
            "sweep_id": run_sweep_id
        })

    conn.close()
    return runs


def _plot_compression_distortion(runs: List[Dict], output: str, format: str, dpi: int, separate_methods: bool = False):
    """Plot compression ratio vs reconstruction distortion."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by method
    methods = {}
    for run in runs:
        method = run["method"]
        metrics = run["metrics"]

        if "compression_ratio" not in metrics or "reconstruction_distortion" not in metrics:
            continue

        if method not in methods:
            methods[method] = {"compression": [], "distortion": []}

        methods[method]["compression"].append(metrics["compression_ratio"])
        methods[method]["distortion"].append(metrics["reconstruction_distortion"])

    markers = {"pq": "o", "sq": "s"}
    colors = {"pq": "tab:blue", "sq": "tab:orange"}

    if separate_methods:
        # Create separate plot for each method
        for method, data in methods.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            distortion_sorted = [data["distortion"][i] for i in sorted_indices]

            ax.plot(
                compression_sorted,
                distortion_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

            ax.set_xlabel("Compression Ratio", fontsize=12)
            ax.set_ylabel("Reconstruction Distortion (MSE)", fontsize=12)
            ax.set_title(f"Compression-Distortion Trade-off - {method.upper()}", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")
            ax.set_yscale("log")

            plt.tight_layout()
            plt.savefig(f"{output}/compression_distortion_tradeoff_{method}.{format}", dpi=dpi)
            plt.close()
    else:
        # Combined plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for method, data in methods.items():
            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            distortion_sorted = [data["distortion"][i] for i in sorted_indices]

            ax.plot(
                compression_sorted,
                distortion_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel("Reconstruction Distortion (MSE)", fontsize=12)
        ax.set_title("Compression-Distortion Trade-off", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(f"{output}/compression_distortion_tradeoff.{format}", dpi=dpi)
        plt.close()


def _plot_pairwise_distortion(runs: List[Dict], output: str, format: str, dpi: int, separate_methods: bool = False):
    """Plot compression ratio vs pairwise distance distortion."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by method
    methods = {}
    for run in runs:
        method = run["method"]
        metrics = run["metrics"]

        if "compression_ratio" not in metrics or "pairwise_distortion_mean" not in metrics:
            continue

        if method not in methods:
            methods[method] = {"compression": [], "pairwise": []}

        methods[method]["compression"].append(metrics["compression_ratio"])
        methods[method]["pairwise"].append(metrics["pairwise_distortion_mean"])

    markers = {"pq": "o", "sq": "s"}
    colors = {"pq": "tab:blue", "sq": "tab:orange"}

    if separate_methods:
        # Create separate plot for each method
        for method, data in methods.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            pairwise_sorted = [data["pairwise"][i] for i in sorted_indices]

            ax.plot(
                compression_sorted,
                pairwise_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

            ax.set_xlabel("Compression Ratio", fontsize=12)
            ax.set_ylabel("Pairwise Distance Distortion (Mean Relative Error)", fontsize=12)
            ax.set_title(f"Distance Preservation vs Compression - {method.upper()}", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

            plt.tight_layout()
            plt.savefig(f"{output}/pairwise_distortion_{method}.{format}", dpi=dpi)
            plt.close()
    else:
        # Combined plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for method, data in methods.items():
            # Sort by compression ratio
            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            pairwise_sorted = [data["pairwise"][i] for i in sorted_indices]

            # Plot line + markers
            ax.plot(
                compression_sorted,
                pairwise_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel("Pairwise Distance Distortion (Mean Relative Error)", fontsize=12)
        ax.set_title("Distance Preservation vs Compression", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"{output}/pairwise_distortion.{format}", dpi=dpi)
        plt.close()


def _plot_rank_distortion(runs: List[Dict], output: str, format: str, dpi: int, separate_methods: bool = False):
    """Plot compression ratio vs rank distortion."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by method
    methods = {}
    for run in runs:
        method = run["method"]
        metrics = run["metrics"]

        if "compression_ratio" not in metrics:
            continue

        # Find rank distortion metric (key varies with k)
        rank_key = None
        for key in metrics:
            if key.startswith("rank_distortion@"):
                rank_key = key
                break

        if rank_key is None:
            continue

        if method not in methods:
            methods[method] = {"compression": [], "rank": []}

        methods[method]["compression"].append(metrics["compression_ratio"])
        methods[method]["rank"].append(metrics[rank_key])

    markers = {"pq": "o", "sq": "s"}
    colors = {"pq": "tab:blue", "sq": "tab:orange"}

    if separate_methods:
        # Create separate plot for each method
        for method, data in methods.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            rank_sorted = [data["rank"][i] for i in sorted_indices]

            ax.plot(
                compression_sorted,
                rank_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

            ax.set_xlabel("Compression Ratio", fontsize=12)
            ax.set_ylabel("Rank Distortion@k (Fraction of Wrong Neighbors)", fontsize=12)
            ax.set_title(f"Ranking Quality vs Compression - {method.upper()}", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

            plt.tight_layout()
            plt.savefig(f"{output}/rank_distortion_{method}.{format}", dpi=dpi)
            plt.close()
    else:
        # Combined plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for method, data in methods.items():
            # Sort by compression ratio
            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            rank_sorted = [data["rank"][i] for i in sorted_indices]

            # Plot line + markers
            ax.plot(
                compression_sorted,
                rank_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel("Rank Distortion@k (Fraction of Wrong Neighbors)", fontsize=12)
        ax.set_title("Ranking Quality vs Compression", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"{output}/rank_distortion.{format}", dpi=dpi)
        plt.close()


def _plot_recall(runs: List[Dict], output: str, format: str, dpi: int, separate_methods: bool = False):
    """Plot compression ratio vs recall."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by method
    methods = {}
    for run in runs:
        method = run["method"]
        metrics = run["metrics"]

        if "compression_ratio" not in metrics or "recall@10" not in metrics:
            continue

        if method not in methods:
            methods[method] = {"compression": [], "recall": []}

        methods[method]["compression"].append(metrics["compression_ratio"])
        methods[method]["recall"].append(metrics["recall@10"])

    markers = {"pq": "o", "sq": "s"}
    colors = {"pq": "tab:blue", "sq": "tab:orange"}

    if separate_methods:
        # Create separate plot for each method
        for method, data in methods.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            recall_sorted = [data["recall"][i] for i in sorted_indices]

            ax.plot(
                compression_sorted,
                recall_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

            ax.set_xlabel("Compression Ratio", fontsize=12)
            ax.set_ylabel("Recall@10", fontsize=12)
            ax.set_title(f"Recall vs Compression - {method.upper()}", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")

            plt.tight_layout()
            plt.savefig(f"{output}/recall_comparison_{method}.{format}", dpi=dpi)
            plt.close()
    else:
        # Combined plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for method, data in methods.items():
            # Sort by compression ratio
            sorted_indices = np.argsort(data["compression"])
            compression_sorted = [data["compression"][i] for i in sorted_indices]
            recall_sorted = [data["recall"][i] for i in sorted_indices]

            # Plot line + markers
            ax.plot(
                compression_sorted,
                recall_sorted,
                marker=markers.get(method, "^"),
                label=method.upper(),
                linewidth=2,
                markersize=8,
                alpha=0.7,
                color=colors.get(method, None)
            )

        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel("Recall@10", fontsize=12)
        ax.set_title("Recall vs Compression", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"{output}/recall_comparison.{format}", dpi=dpi)
        plt.close()


def _generate_comparison_table(runs: List[Dict], output: str):
    """Generate text table comparing all methods."""
    with open(f"{output}/comparison_table.txt", "w") as f:
        f.write("=" * 100 + "\n")
        f.write("BENCHMARK COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")

        header = f"{'Method':<10} {'Compression':<12} {'Recon MSE':<12} {'Pairwise':<12} {'Rank Dist':<12} {'Recall@10':<12}\n"
        f.write(header)
        f.write("-" * 100 + "\n")

        for run in runs:
            method = run["method"]
            metrics = run["metrics"]

            comp = metrics.get("compression_ratio", 0)
            recon = metrics.get("reconstruction_distortion", 0)
            pairwise = metrics.get("pairwise_distortion_mean", 0)
            rank_key = next((k for k in metrics if k.startswith("rank_distortion@")), None)
            rank = metrics.get(rank_key, 0) if rank_key else 0
            recall = metrics.get("recall@10", 0)

            row = f"{method:<10} {comp:<12.2f} {recon:<12.4f} {pairwise:<12.4f} {rank:<12.4f} {recall:<12.4f}\n"
            f.write(row)

        f.write("\n" + "=" * 100 + "\n")


def _plot_pareto_frontier(runs: List[Dict], output: str, format: str, dpi: int):
    """Plot Pareto frontier showing optimal compression-recall tradeoffs."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by method
    methods = {}
    for run in runs:
        method = run["method"]
        metrics = run["metrics"]

        if "compression_ratio" not in metrics or "recall@10" not in metrics:
            continue

        if method not in methods:
            methods[method] = {"compression": [], "recall": [], "points": []}

        methods[method]["compression"].append(metrics["compression_ratio"])
        methods[method]["recall"].append(metrics["recall@10"])
        methods[method]["points"].append((metrics["compression_ratio"], metrics["recall@10"]))

    def is_pareto_optimal(point, points):
        """Check if a point is on the Pareto frontier (maximizing both compression and recall)."""
        comp, recall = point
        for other_comp, other_recall in points:
            # Another point dominates if it has both higher compression AND higher recall
            if other_comp > comp and other_recall > recall:
                return False
        return True

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"pq": "tab:blue", "opq": "tab:green", "sq": "tab:orange",
              "saq": "tab:red", "rabitq": "tab:purple"}
    markers = {"pq": "o", "opq": "^", "sq": "s", "saq": "D", "rabitq": "v"}

    all_points = []
    for method, data in methods.items():
        all_points.extend(data["points"])

        # Plot all points
        ax.scatter(
            data["compression"],
            data["recall"],
            marker=markers.get(method, "o"),
            label=method.upper(),
            s=100,
            alpha=0.6,
            color=colors.get(method, None),
            edgecolors='black',
            linewidths=1
        )

    # Identify and highlight Pareto frontier
    pareto_points = [p for p in all_points if is_pareto_optimal(p, all_points)]
    if pareto_points:
        pareto_points.sorted = sorted(pareto_points, key=lambda x: x[0])
        pareto_comp = [p[0] for p in pareto_points]
        pareto_recall = [p[1] for p in pareto_points]

        ax.scatter(
            pareto_comp,
            pareto_recall,
            s=200,
            facecolors='none',
            edgecolors='gold',
            linewidths=3,
            label='Pareto Frontier',
            zorder=10
        )

    ax.set_xlabel("Compression Ratio (higher is better)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Recall@10 (higher is better)", fontsize=13, fontweight='bold')
    ax.set_title("Pareto Frontier: Optimal Compression-Accuracy Tradeoffs",
                 fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{output}/pareto_frontier.{format}", dpi=dpi, bbox_inches='tight')
    plt.close()


def _plot_radar_chart(runs: List[Dict], output: str, format: str, dpi: int):
    """Generate radar/spider chart comparing methods across multiple dimensions."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Aggregate metrics by method (take mean of all runs per method)
    methods_data = {}
    for run in runs:
        method = run["method"]
        metrics = run["metrics"]

        if method not in methods_data:
            methods_data[method] = {
                "compression_ratios": [],
                "recalls": [],
                "pairwise_distortions": [],
                "reconstruction_distortions": []
            }

        if "compression_ratio" in metrics:
            methods_data[method]["compression_ratios"].append(metrics["compression_ratio"])
        if "recall@10" in metrics:
            methods_data[method]["recalls"].append(metrics["recall@10"])
        if "pairwise_distortion_mean" in metrics:
            methods_data[method]["pairwise_distortions"].append(metrics["pairwise_distortion_mean"])
        if "reconstruction_distortion" in metrics:
            methods_data[method]["reconstruction_distortions"].append(metrics["reconstruction_distortion"])

    # Calculate averages and normalize to 0-1 scale
    normalized_data = {}
    all_compression = []
    all_recall = []
    all_pairwise = []
    all_recon = []

    for method, data in methods_data.items():
        if data["compression_ratios"]:
            all_compression.extend(data["compression_ratios"])
        if data["recalls"]:
            all_recall.extend(data["recalls"])
        if data["pairwise_distortions"]:
            all_pairwise.extend(data["pairwise_distortions"])
        if data["reconstruction_distortions"]:
            all_recon.extend(data["reconstruction_distortions"])

    for method, data in methods_data.items():
        avg_compression = np.mean(data["compression_ratios"]) if data["compression_ratios"] else 0
        avg_recall = np.mean(data["recalls"]) if data["recalls"] else 0
        avg_pairwise = np.mean(data["pairwise_distortions"]) if data["pairwise_distortions"] else 0
        avg_recon = np.mean(data["reconstruction_distortions"]) if data["reconstruction_distortions"] else 0

        # Normalize (higher is better for all metrics after transformation)
        norm_compression = (avg_compression - min(all_compression)) / (max(all_compression) - min(all_compression)) if all_compression else 0
        norm_recall = (avg_recall - min(all_recall)) / (max(all_recall) - min(all_recall)) if all_recall else 0
        # Invert distortion metrics (lower distortion is better, so invert to make higher better)
        norm_pairwise = 1 - ((avg_pairwise - min(all_pairwise)) / (max(all_pairwise) - min(all_pairwise))) if all_pairwise else 0
        norm_recon = 1 - ((avg_recon - min(all_recon)) / (max(all_recon) - min(all_recon))) if all_recon else 0

        normalized_data[method] = {
            "Compression": norm_compression,
            "Recall": norm_recall,
            "Distance\nPreservation": norm_pairwise,
            "Reconstruction\nQuality": norm_recon
        }

    # Create radar chart
    categories = ["Compression", "Recall", "Distance\nPreservation", "Reconstruction\nQuality"]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = {"pq": "tab:blue", "opq": "tab:green", "sq": "tab:orange",
              "saq": "tab:red", "rabitq": "tab:purple"}

    for method, data in normalized_data.items():
        values = [data[cat] for cat in categories]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2,
                label=method.upper(), color=colors.get(method, None))
        ax.fill(angles, values, alpha=0.15, color=colors.get(method, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=9)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax.set_title("Multi-Dimensional Method Comparison\n(higher is better for all dimensions)",
                 fontsize=15, fontweight="bold", pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{output}/method_radar_comparison.{format}", dpi=dpi, bbox_inches='tight')
    plt.close()
