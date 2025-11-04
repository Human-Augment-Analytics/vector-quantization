#!/usr/bin/env python3
"""
Production sweep script for vector quantization benchmarking.

Configurable via command-line arguments for realistic parameter sweeps
on any supported dataset.

Usage:
    python run_sweep.py --dataset dbpedia-100k --methods pq opq sq --output-dir results/sweep1
    python run_sweep.py --dataset dbpedia-1536 --limit 500000 --methods all
    python run_sweep.py --config sweeps/dbpedia_full.yaml
"""

import os
import sys
import argparse
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set HuggingFace cache directories
# Priority: 1) Shared cache (if exists), 2) $TMPDIR, 3) Local .cache
SHARED_CACHE = "/storage/ice-shared/cs8903onl/.cache/huggingface"

if os.path.exists(SHARED_CACHE):
    # Use shared persistent cache (no re-downloads between jobs)
    hf_cache_base = SHARED_CACHE
    print(f"[INFO] Using shared persistent cache: {SHARED_CACHE}")
elif "TMPDIR" in os.environ:
    # Use fast local storage (will re-download each job)
    hf_cache_base = os.path.join(os.environ["TMPDIR"], "hf_cache")
    print(f"[INFO] Using fast local storage: $TMPDIR = {os.environ['TMPDIR']}")
    print("[INFO] Note: Dataset will re-download each job. Consider using shared cache.")
else:
    # Local development
    hf_cache_base = os.path.abspath(".cache/huggingface")
    print("[INFO] Using local cache: .cache/")

os.environ["HF_HOME"] = hf_cache_base
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_base, "datasets")

from haag_vq.data import (
    load_dbpedia_openai_1536_100k,
    load_dbpedia_openai_1536,
    load_dbpedia_openai_3072,
    load_cohere_msmarco_passages,
)
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


def load_dataset(dataset_name: str, limit: Optional[int] = None, cache_dir: Optional[str] = None):
    """Load dataset by name."""
    print(f"\n{'='*70}")
    print(f"  Loading Dataset: {dataset_name}")
    print(f"{'='*70}")

    if cache_dir is None:
        cache_dir = os.path.join(hf_cache_base, "datasets")
    os.makedirs(cache_dir, exist_ok=True)

    if dataset_name == "dbpedia-100k":
        print(f"Loading DBpedia 100K (1536-dim), limit={limit}")
        data = load_dbpedia_openai_1536_100k(cache_dir=cache_dir, limit=limit)
    elif dataset_name == "dbpedia-1536":
        print(f"Loading DBpedia 1M (1536-dim), limit={limit}")
        data = load_dbpedia_openai_1536(cache_dir=cache_dir, limit=limit)
    elif dataset_name == "dbpedia-3072":
        print(f"Loading DBpedia 1M (3072-dim), limit={limit}")
        data = load_dbpedia_openai_3072(cache_dir=cache_dir, limit=limit)
    elif dataset_name == "cohere-msmarco":
        print(f"Loading Cohere MS MARCO, limit={limit}")
        data = load_cohere_msmarco_passages(
            limit=limit or 100_000,
            streaming=True,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"✅ Dataset loaded: {data.vectors.shape}")
    print(f"   Vectors: {data.vectors.shape[0]:,}")
    print(f"   Dimensions: {data.vectors.shape[1]}")
    print(f"   Queries: {data.queries.shape[0]}")

    # Compute ground truth if needed
    if data.ground_truth is None and data.vectors.shape[0] <= 100_000:
        print("\n⏳ Computing ground truth (k-nearest neighbors)...")
        import faiss
        d = data.vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.ascontiguousarray(data.vectors, dtype=np.float32))
        k = 100
        queries_for_search = np.ascontiguousarray(data.queries, dtype=np.float32)
        distances, indices = index.search(queries_for_search, k)
        data.ground_truth = indices
        print(f"✅ Ground truth computed: top-{k} neighbors for {len(data.queries)} queries")

    return data


def get_sweep_configs(methods: List[str], dimension: int) -> List[Dict[str, Any]]:
    """Generate sweep configurations based on methods and dimension."""
    configs = []

    if "pq" in methods:
        # PQ configurations - adjust M based on dimension
        valid_Ms = [m for m in [8, 12, 16, 24, 32, 48, 64, 96] if dimension % m == 0]
        for M in valid_Ms[:4]:  # Use first 4 valid values
            for B in [6, 8]:
                configs.append({
                    "method": "pq",
                    "params": {"M": M, "B": B},
                    "name": f"PQ(M={M}, B={B})"
                })

    if "opq" in methods:
        # OPQ configurations
        valid_Ms = [m for m in [8, 12, 16, 24, 32, 48] if dimension % m == 0]
        for M in valid_Ms[:3]:  # Use first 3 valid values
            configs.append({
                "method": "opq",
                "params": {"M": M, "B": 8},
                "name": f"OPQ(M={M}, B=8)"
            })

    if "sq" in methods:
        # Scalar quantization
        configs.append({
            "method": "sq",
            "params": {},
            "name": "SQ(8-bit)"
        })

    if "saq" in methods:
        # SAQ configurations
        for bits in [4, 6, 8]:
            configs.append({
                "method": "saq",
                "params": {"num_bits": bits, "allowed_bits": [0, 2, 4, 6, 8]},
                "name": f"SAQ({bits} bits/dim)"
            })
        # Total bits configurations
        for total_bits in [dimension * 2, dimension * 4]:
            configs.append({
                "method": "saq",
                "params": {"total_bits": total_bits, "allowed_bits": [0, 2, 4, 6, 8]},
                "name": f"SAQ({total_bits} total bits)"
            })

    if "rabitq" in methods:
        # RaBitQ
        configs.append({
            "method": "rabitq",
            "params": {"metric_type": MetricType.L2},
            "name": "RaBitQ(L2)"
        })

    return configs


def create_model(method: str, params: Dict[str, Any]):
    """Create a quantization model from method name and parameters."""
    if method == "pq":
        return ProductQuantizer(**params)
    elif method == "opq":
        return OptimizedProductQuantizer(**params)
    elif method == "sq":
        return ScalarQuantizer(**params)
    elif method == "saq":
        return SAQ(**params)
    elif method == "rabitq":
        return RaBitQuantizer(**params)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_sweep(
    dataset_name: str,
    methods: List[str],
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    skip_ground_truth_metrics: bool = False,
):
    """Run a complete parameter sweep."""
    # Generate sweep ID
    sweep_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 70)
    print("  VECTOR QUANTIZATION PARAMETER SWEEP")
    print("=" * 70)
    print(f"\n🔖 Sweep ID: {sweep_id}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"📈 Methods: {', '.join(methods)}")
    if limit:
        print(f"🔢 Vector limit: {limit:,}")
    print("=" * 70)

    # Load dataset
    data = load_dataset(dataset_name, limit=limit)

    # Generate configurations
    configs = get_sweep_configs(methods, data.vectors.shape[1])
    print(f"\n🚀 Running {len(configs)} configurations")
    print("=" * 70)

    # Run each configuration
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {config['name']}")
        print("-" * 70)

        try:
            # Create and train model
            model = create_model(config["method"], config["params"])
            print("Training...")
            model.fit(data.vectors)

            # Compress
            print("Compressing...")
            compressed = model.compress(data.vectors)

            # Compute metrics
            print("Computing metrics...")
            metrics = {
                "compression_ratio": model.get_compression_ratio(data.vectors),
                "reconstruction_distortion": compute_distortion(data.vectors, compressed, model),
                "pairwise_distortion_mean": compute_pairwise_distortion(
                    data.vectors, compressed, model, num_pairs=min(500, len(data.vectors) // 2)
                )["mean"],
            }

            # Ground truth metrics (if available)
            if not skip_ground_truth_metrics and data.ground_truth is not None:
                metrics["rank_distortion@10"] = compute_rank_distortion(data, model, k=10, num_queries=min(100, len(data.queries)))
                metrics["recall@10"] = evaluate_recall(data, model, num_queries=min(100, len(data.queries)))["recall@10"]

            # Log to database
            log_run(
                method=config["method"],
                dataset=dataset_name,
                metrics=metrics,
                config=config["params"],
                sweep_id=sweep_id,
            )

            # Print summary
            print(f"✅ Success!")
            print(f"   Compression: {metrics['compression_ratio']:.1f}x")
            print(f"   MSE: {metrics['reconstruction_distortion']:.6f}")
            print(f"   Pairwise: {metrics['pairwise_distortion_mean']:.6f}")
            if "recall@10" in metrics:
                print(f"   Rank@10: {metrics.get('rank_distortion@10', 'N/A'):.4f}")
                print(f"   Recall@10: {metrics['recall@10']:.4f}")

            results.append({"config": config, "metrics": metrics, "status": "success"})

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"config": config, "error": str(e), "status": "failed"})

    # Print summary
    print("\n" + "=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)
    print(f"\n🔖 Sweep ID: {sweep_id}")
    print(f"✅ Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
    print(f"❌ Failed: {sum(1 for r in results if r['status'] == 'failed')}/{len(results)}")
    print(f"\n📁 Results logged to: logs/benchmark_runs.db")
    print(f"📊 View results: sqlite3 logs/benchmark_runs.db \"SELECT * FROM runs WHERE sweep_id='{sweep_id}';\"")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{sweep_id}_summary.txt")
        with open(output_file, "w") as f:
            f.write(f"Sweep ID: {sweep_id}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            for r in results:
                f.write(f"{r['config']['name']}: {r['status']}\n")
                if r['status'] == 'success':
                    for k, v in r['metrics'].items():
                        f.write(f"  {k}: {v}\n")
        print(f"📄 Summary saved to: {output_file}")

    return sweep_id, results


def main():
    parser = argparse.ArgumentParser(
        description="Run vector quantization parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DBpedia 100K with PQ and OPQ
  python run_sweep.py --dataset dbpedia-100k --methods pq opq

  # DBpedia 1M (first 500K vectors) with all methods
  python run_sweep.py --dataset dbpedia-1536 --limit 500000 --methods all

  # Full DBpedia 1M with specific methods
  python run_sweep.py --dataset dbpedia-1536 --methods pq opq sq saq

  # Cohere MS MARCO (100K subset)
  python run_sweep.py --dataset cohere-msmarco --limit 100000 --methods pq sq
        """
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["dbpedia-100k", "dbpedia-1536", "dbpedia-3072", "cohere-msmarco"],
        help="Dataset to use for sweep"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to benchmark (pq, opq, sq, saq, rabitq, or 'all')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of vectors to load (default: full dataset)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files (default: results)"
    )
    parser.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="Skip ground truth metrics (rank distortion, recall)"
    )

    args = parser.parse_args()

    # Handle "all" methods
    if "all" in args.methods:
        methods = ["pq", "opq", "sq", "saq", "rabitq"]
    else:
        methods = args.methods

    # Run sweep
    sweep_id, results = run_sweep(
        dataset_name=args.dataset,
        methods=methods,
        limit=args.limit,
        output_dir=args.output_dir,
        skip_ground_truth_metrics=args.skip_ground_truth,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
