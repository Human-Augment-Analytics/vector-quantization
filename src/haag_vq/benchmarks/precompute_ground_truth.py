"""
Tool for precomputing ground truth nearest neighbors for large datasets.

This should be run as a preprocessing step on the ICE cluster before
running benchmarks. It uses FAISS for efficient k-NN search on large datasets.
"""

from pathlib import Path
import numpy as np
import typer
from time import perf_counter


def precompute_ground_truth(
    vectors_path: str = typer.Option(..., help="Path to vectors file (.npy format)"),
    output_path: str = typer.Option(..., help="Path to save ground truth (.npy format)"),
    num_queries: int = typer.Option(100, help="Number of queries (taken from start of vectors)"),
    k: int = typer.Option(100, help="Number of nearest neighbors to compute"),
    use_gpu: bool = typer.Option(False, help="Use GPU for computation (if available)"),
    batch_size: int = typer.Option(1000, help="Batch size for processing queries"),
):
    """
    Precompute ground truth k-nearest neighbors using FAISS.

    This tool is designed for large-scale datasets where computing ground truth
    on-the-fly would be memory-prohibitive. Run this on ICE cluster with:

    Example:
        sbatch --mem=64G --time=4:00:00 --wrap="vq-benchmark precompute-gt \\
            --vectors-path /scratch/$USER/msmarco_vectors.npy \\
            --output-path /scratch/$USER/msmarco_ground_truth.npy \\
            --num-queries 1000 \\
            --k 100"

    Then use the precomputed file in benchmarks with:
        vq-benchmark run --ground-truth-path /scratch/$USER/msmarco_ground_truth.npy
    """
    try:
        import faiss
    except ImportError:
        print("ERROR: FAISS is required for efficient ground truth computation.")
        print("Install with: pip install faiss-cpu  (or faiss-gpu for GPU support)")
        raise typer.Exit(1)

    print("=" * 70)
    print("  Precomputing Ground Truth Nearest Neighbors")
    print("=" * 70)

    # Load vectors
    print(f"\n1. Loading vectors from: {vectors_path}")
    vectors_path = Path(vectors_path)
    if not vectors_path.exists():
        print(f"ERROR: Vectors file not found: {vectors_path}")
        raise typer.Exit(1)

    vectors = np.load(vectors_path)
    print(f"   Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")

    # Extract queries (first num_queries vectors)
    if num_queries > len(vectors):
        print(f"WARNING: num_queries ({num_queries}) > dataset size ({len(vectors)})")
        num_queries = len(vectors)

    queries = vectors[:num_queries]
    print(f"   Using first {num_queries} vectors as queries")

    # Build FAISS index
    print(f"\n2. Building FAISS index...")
    start_time = perf_counter()

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)

    if use_gpu and faiss.get_num_gpus() > 0:
        print("   Using GPU acceleration")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print("   Using CPU")

    index.add(vectors.astype(np.float32))
    build_time = perf_counter() - start_time
    print(f"   Index built in {build_time:.2f}s")

    # Search for k-nearest neighbors
    print(f"\n3. Computing {k}-nearest neighbors for {num_queries} queries...")
    search_start = perf_counter()

    # Process in batches to avoid memory issues
    all_indices = []
    all_distances = []

    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        batch_queries = queries[i:end_idx].astype(np.float32)

        distances, indices = index.search(batch_queries, k)
        all_distances.append(distances)
        all_indices.append(indices)

        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"   Processed {end_idx}/{num_queries} queries...")

    ground_truth = np.vstack(all_indices)
    distances_array = np.vstack(all_distances)

    search_time = perf_counter() - search_start
    print(f"   Search completed in {search_time:.2f}s")
    print(f"   Average query time: {(search_time / num_queries) * 1000:.2f}ms")

    # Save results
    print(f"\n4. Saving ground truth to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, ground_truth)

    # Also save distances for debugging
    distances_path = output_path.with_suffix('.distances.npy')
    np.save(distances_path, distances_array)
    print(f"   Also saved distances to: {distances_path}")

    print("\n" + "=" * 70)
    print("  Ground Truth Computation Complete!")
    print("=" * 70)
    print(f"\nGround truth shape: {ground_truth.shape}")
    print(f"Total time: {build_time + search_time:.2f}s")
    print(f"\nUse this file in benchmarks with:")
    print(f"  vq-benchmark run --ground-truth-path {output_path}")
