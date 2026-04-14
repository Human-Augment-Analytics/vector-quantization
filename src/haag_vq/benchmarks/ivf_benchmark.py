"""
IVF-aware benchmark runner for large-scale vector quantization experiments.

Loads datasets from directories containing train.npy / queries.npy,
runs multiple methods (pq_flat, sq_flat, faiss_ivfpq, saq) at a given
bits-per-dimension budget, and writes results to CSV.

Usage from slurm scripts:
    python -m haag_vq.benchmarks.ivf_benchmark \
        --dataset /path/to/dataset_dir \
        --methods faiss_ivfpq,saq \
        --bpd 4 --k 10 --K 4096 --nprobe 64 \
        --output results.csv
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import typer

try:
    import faiss
except ImportError:
    faiss = None


def _load_npy_dataset(dataset_dir: str, num_queries: int = 1000):
    """Load train.npy and queries.npy from a directory.

    If queries.npy doesn't exist, splits the last `num_queries` rows
    from train.npy as queries.
    """
    d = Path(dataset_dir)
    train_path = d / "train.npy"
    queries_path = d / "queries.npy"
    gt_path = d / "ground_truth.npy"

    if not train_path.exists():
        raise FileNotFoundError(f"train.npy not found in {d}")

    train = np.load(train_path).astype(np.float32)
    if queries_path.exists():
        queries = np.load(queries_path).astype(np.float32)
    else:
        queries = train[-num_queries:]
        train = train[:-num_queries]

    gt = None
    if gt_path.exists():
        gt = np.load(gt_path)

    return train, queries, gt


def _compute_ground_truth(train: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Brute-force k-NN ground truth via FAISS."""
    assert faiss is not None, "FAISS required for ground truth computation"
    D = train.shape[1]
    index = faiss.IndexFlatL2(D)
    index.add(train)
    _, gt_indices = index.search(queries, k)
    return gt_indices


def _recall_at_k(gt: np.ndarray, retrieved: np.ndarray, k: int) -> float:
    """Recall@k: fraction of true top-k neighbors found in retrieved top-k."""
    nq = gt.shape[0]
    hits = 0
    for i in range(nq):
        gt_set = set(gt[i, :k].tolist())
        ret_set = set(retrieved[i, :k].tolist())
        hits += len(gt_set & ret_set)
    return hits / (nq * k)


def _bpd_to_pq_M(D: int, bpd: int) -> int:
    """Convert bits-per-dimension to PQ subquantizer count M.

    With B=8 (256 centroids per subquantizer), each subquantizer
    contributes 8 bits for (D/M) dimensions. So bpd = M*8/D → M = D*bpd/8.
    """
    M = D * bpd // 8
    M = max(1, M)
    # M must divide D
    while D % M != 0 and M > 1:
        M -= 1
    return M


def _run_pq_flat(train, queries, gt, k, bpd):
    """PQ with brute-force (flat) search on reconstructed vectors."""
    from haag_vq.methods.product_quantization import ProductQuantizer

    D = train.shape[1]
    M = _bpd_to_pq_M(D, bpd)
    model = ProductQuantizer(M=M, B=8)

    t0 = perf_counter()
    model.fit(train)
    fit_time = perf_counter() - t0

    codes = model.compress(train)
    reconstructed = model.decompress(codes)

    mse = float(np.mean(np.sum((train - reconstructed) ** 2, axis=1)))

    # Flat search on reconstructed vectors
    index = faiss.IndexFlatL2(D)
    index.add(reconstructed.astype(np.float32))

    t0 = perf_counter()
    _, I = index.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)
    mem = codes.nbytes
    comp = (train.nbytes) / max(mem, 1)

    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": mse,
    }


def _run_sq_flat(train, queries, gt, k, bpd):
    """Scalar quantization with brute-force search on reconstructed vectors."""
    from haag_vq.methods.scalar_quantization import ScalarQuantizer

    D = train.shape[1]
    # SQ supports 4, 8, 16 bits; bpd maps directly
    num_bits = bpd if bpd in (4, 8, 16) else 8
    model = ScalarQuantizer(num_bits=num_bits)

    model.fit(train)
    codes = model.compress(train)
    reconstructed = model.decompress(codes)

    mse = float(np.mean(np.sum((train - reconstructed) ** 2, axis=1)))

    index = faiss.IndexFlatL2(D)
    index.add(reconstructed.astype(np.float32))

    t0 = perf_counter()
    _, I = index.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)
    mem = codes.nbytes if isinstance(codes, np.ndarray) else sys.getsizeof(codes)
    comp = train.nbytes / max(mem, 1)

    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": mse,
    }


def _run_faiss_ivfpq(train, queries, gt, k, bpd, K, nprobe):
    """FAISS IVF+PQ index: train, add, search, measure recall/QPS."""
    D = train.shape[1]
    M = _bpd_to_pq_M(D, bpd)

    factory_key = f"IVF{K},PQ{M}x8"
    print(f"  faiss_ivfpq: factory={factory_key}, nprobe={nprobe}")

    index = faiss.index_factory(D, factory_key, faiss.METRIC_L2)
    index.nprobe = nprobe

    t0 = perf_counter()
    index.train(train)
    index.add(train)
    build_time = perf_counter() - t0
    print(f"  faiss_ivfpq: index built in {build_time:.1f}s")

    t0 = perf_counter()
    _, I = index.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)

    # Memory: PQ codes (N * M bytes) + coarse centroids (K * D * 4) + overhead
    mem = train.shape[0] * M + K * D * 4
    comp = train.nbytes / max(mem, 1)

    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": "",  # IVF doesn't reconstruct all vectors cheaply
    }


def _run_rabitq(train, queries, gt, k, bpd):
    """Flat RaBitQ via native estimator (``RaBitQIndex``).

    Note: ``bpd`` is ignored. RaBitQ encodes each normalised vector at ~1 bit
    per dimension by construction — the value is accepted only so that the
    method-runner signature matches the others in ``METHOD_RUNNERS``.
    """
    from haag_vq.methods.search import RaBitQIndex

    model = RaBitQIndex()

    t0 = perf_counter()
    model.fit(train)
    fit_time = perf_counter() - t0
    print(f"  rabitq: fit in {fit_time:.1f}s")

    t0 = perf_counter()
    I = model.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)
    mem = int(model.memory_footprint())
    comp = train.nbytes / max(mem, 1)

    mse = model.reconstruction_mse(train, sample_ids=np.arange(min(1000, len(train))))

    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": mse if mse is not None else "",
    }


def _run_rabitq_ivf(train, queries, gt, k, bpd, K, nprobe):
    """IVF+RaBitQ via ``RaBitQIVFIndex``. ``bpd`` ignored (RaBitQ = 1 bpd)."""
    from haag_vq.methods.search import RaBitQIVFIndex

    model = RaBitQIVFIndex(nlist=K, nprobe=nprobe)

    t0 = perf_counter()
    model.fit(train)
    fit_time = perf_counter() - t0
    print(f"  rabitq_ivf: fit in {fit_time:.1f}s")

    t0 = perf_counter()
    I = model.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)
    mem = int(model.memory_footprint())
    comp = train.nbytes / max(mem, 1)

    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": "",
    }


def _run_opq_flat(train, queries, gt, k, bpd):
    """OPQ (rotation + PQ) with brute-force search on reconstructed vectors."""
    from haag_vq.methods.optimized_product_quantization import (
        OptimizedProductQuantizer,
    )

    D = train.shape[1]
    M = _bpd_to_pq_M(D, bpd)
    model = OptimizedProductQuantizer(M=M, B=8)

    t0 = perf_counter()
    model.fit(train)
    fit_time = perf_counter() - t0
    print(f"  opq_flat: fit in {fit_time:.1f}s")

    codes = model.compress(train)
    reconstructed = model.decompress(codes).astype(np.float32)

    mse = float(np.mean(np.sum((train - reconstructed) ** 2, axis=1)))

    index = faiss.IndexFlatL2(D)
    index.add(reconstructed)

    t0 = perf_counter()
    _, I = index.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)
    mem = int(codes.nbytes) if isinstance(codes, np.ndarray) else sys.getsizeof(codes)
    comp = train.nbytes / max(mem, 1)

    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": mse,
    }


def _run_saq(train, queries, gt, k, bpd, K, nprobe):
    """SAQ via the C++ engine (``SaqIndex``).

    The previous implementation imported a pure-Python SAQ scaffold that has
    since been removed from the repo. This path now routes through
    ``SaqIndex`` — the same wrapper used by ``run_benchmarks.py`` — so
    ``ivf_benchmark`` and ``run_benchmarks`` measure the same SAQ.
    """
    from haag_vq.methods.search import SaqIndex

    model = SaqIndex(bpd=bpd, K=K, nprobe=nprobe)

    t0 = perf_counter()
    model.fit(train)
    fit_time = perf_counter() - t0
    print(f"  saq: fit in {fit_time:.1f}s")

    t0 = perf_counter()
    I = model.search(queries, k)
    search_time = perf_counter() - t0

    recall = _recall_at_k(gt, I, k)
    qps = len(queries) / max(search_time, 1e-12)
    mem = int(model.memory_footprint())
    comp = train.nbytes / max(mem, 1)

    # SaqIndex.reconstruction_mse is intentionally None (construct() path does
    # not retain raw codes for decompression). Leave mse blank — matches
    # faiss_ivfpq row.
    return {
        "recall_at_k": recall,
        "qps": qps,
        "memory_bytes": mem,
        "compression_ratio": comp,
        "mse": "",
    }


METHOD_RUNNERS = {
    "pq_flat": lambda t, q, gt, k, bpd, K, np_: _run_pq_flat(t, q, gt, k, bpd),
    "opq_flat": lambda t, q, gt, k, bpd, K, np_: _run_opq_flat(t, q, gt, k, bpd),
    "sq_flat": lambda t, q, gt, k, bpd, K, np_: _run_sq_flat(t, q, gt, k, bpd),
    "faiss_ivfpq": _run_faiss_ivfpq,
    "saq": _run_saq,
    "rabitq": lambda t, q, gt, k, bpd, K, np_: _run_rabitq(t, q, gt, k, bpd),
    "rabitq_ivf": _run_rabitq_ivf,
}


def _utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _timestamped_output_path(path: Path, now: Optional[datetime] = None) -> Path:
    """Insert ``_YYYYMMDD_HHMMSS`` before the suffix to prevent re-run overwrites."""
    if now is None:
        now = datetime.now(timezone.utc)
    stamp = now.strftime('%Y%m%d_%H%M%S')
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def ivf_benchmark(
    dataset: str = typer.Option(..., help="Path to dataset directory containing train.npy and queries.npy"),
    methods: str = typer.Option("faiss_ivfpq,saq", help="Comma-separated methods: pq_flat,sq_flat,faiss_ivfpq,saq"),
    bpd: int = typer.Option(4, help="Bits per dimension budget"),
    k: int = typer.Option(10, help="Top-k for recall evaluation"),
    nlist: int = typer.Option(256, help="Number of IVF clusters"),
    nprobe: int = typer.Option(32, help="Number of IVF clusters to probe at search time"),
    output: str = typer.Option(..., help="Path to output CSV file"),
    num_queries: int = typer.Option(1000, help="Number of query vectors"),
    gt_k: int = typer.Option(100, help="k for ground truth computation (must be >= k)"),
):
    """Run IVF-aware benchmarks on .npy datasets and output CSV results."""

    assert faiss is not None, "FAISS is required. Install faiss-cpu or faiss-gpu."

    print(f"Loading dataset from {dataset}...")
    train, queries, gt = _load_npy_dataset(dataset, num_queries=num_queries)
    N, D = train.shape
    nq = queries.shape[0]
    print(f"  train: {train.shape}, queries: {queries.shape}")

    if gt is None:
        print(f"Computing ground truth (k={gt_k})...")
        t0 = perf_counter()
        gt = _compute_ground_truth(train, queries, gt_k)
        print(f"  Ground truth computed in {perf_counter() - t0:.1f}s")
        # Save for reuse
        gt_path = Path(dataset) / "ground_truth.npy"
        np.save(gt_path, gt)
        print(f"  Saved to {gt_path}")

    method_list = [m.strip() for m in methods.split(",")]
    results = []
    run_ts = _utc_timestamp()

    for method_name in method_list:
        runner = METHOD_RUNNERS.get(method_name)
        if runner is None:
            print(f"WARNING: Unknown method '{method_name}', skipping. "
                  f"Available: {list(METHOD_RUNNERS.keys())}")
            continue

        print(f"\nRunning {method_name} (bpd={bpd}, nlist={nlist}, nprobe={nprobe})...")
        try:
            metrics = runner(train, queries, gt, k, bpd, nlist, nprobe)
            row = {
                "method": method_name,
                "k": k,
                "N": N,
                "D": D,
                "timestamp": run_ts,
                **metrics,
            }
            results.append(row)
            print(f"  recall@{k}={metrics['recall_at_k']:.4f}  "
                  f"qps={metrics['qps']:.1f}  "
                  f"compression={metrics['compression_ratio']:.1f}x")
        except Exception as e:
            print(f"  ERROR running {method_name}: {e}")
            import traceback
            traceback.print_exc()

    # Write CSV — filename is timestamped so re-runs don't clobber.
    if results:
        out_path = _timestamped_output_path(Path(output))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["method", "recall_at_k", "qps", "memory_bytes",
                       "compression_ratio", "mse", "k", "N", "D", "timestamp"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults written to {out_path}")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    typer.run(ivf_benchmark)
