from pathlib import Path
from time import perf_counter
import os

import numpy as np
import typer

from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.metrics.performance import measure_qps, time_compress, time_decompress
from haag_vq.metrics.recall import evaluate_recall
from haag_vq.data.datasets import Dataset, load_dummy_dataset
from haag_vq.utils.run_logger import log_run


def run(
    method: str = typer.Option("pq", help="Compression method: pq or sq"),
    dataset: str = typer.Option(..., help="Dataset name: dummy, huggingface, or msmarco (REQUIRED)"),
    num_samples: int = typer.Option(10000, help="Number of samples to use"),
    dim: int = typer.Option(1024, help="Dimensionality for dummy dataset"),
    M: int = typer.Option(8, help="[PQ only] Number of subquantizers (M)"),
    B: int = typer.Option(8, help="[PQ only] Bits per subvector index (B)"),
    with_recall: bool = typer.Option(False, help="Whether to compute Recall@k metrics"),
    ground_truth_path: str = typer.Option(None, help="Path to precomputed ground truth (.npy file)"),
    codebooks_dir: str = typer.Option(None, help="Directory to save codebooks (default: ./codebooks or $CODEBOOKS_DIR)"),
    db_path: str = typer.Option(None, help="Path to SQLite database (default: logs/benchmark_runs.db or $DB_PATH)")
):
    # Determine codebooks directory (priority: CLI arg > env var > default)
    if codebooks_dir is None:
        codebooks_dir = os.getenv("CODEBOOKS_DIR")
    if codebooks_dir is None:
        codebooks_dir = Path(__file__).resolve().parents[3] / "codebooks"
    else:
        codebooks_dir = Path(codebooks_dir)
    codebooks_dir.mkdir(parents=True, exist_ok=True)

    # Load precomputed ground truth if provided
    precomputed_gt = None
    if ground_truth_path:
        print(f"Loading precomputed ground truth from: {ground_truth_path}")
        precomputed_gt = np.load(ground_truth_path)
        print(f"   Loaded ground truth shape: {precomputed_gt.shape}")

    print(f"Loading dataset: {dataset}...")
    if dataset == "dummy":
        # For dummy dataset, pass skip_ground_truth=True if no precomputed GT and large dataset
        skip_gt = (precomputed_gt is None) and (num_samples > 100000)
        data = load_dummy_dataset(num_samples=num_samples, dim=dim)
        # Override ground truth if provided
        if precomputed_gt is not None:
            data.ground_truth = precomputed_gt
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Currently supported: dummy")

    if method == "pq":
        print(f"Fitting PQ (M={M}, B={B})...")
        model = ProductQuantizer(M=M, B=B)
    elif method == "sq":
        print("Fitting SQ...")
        model = ScalarQuantizer()
    else:
        raise ValueError(f"Unsupported method: {method}")

    X = data.vectors
    fit_start = perf_counter()
    model.fit(X)
    fit_time = perf_counter() - fit_start

    X_compressed, compression_time = time_compress(model, X)
    X_reconstructed, decompression_time = time_decompress(model, X_compressed)

    distortion = compute_distortion(
        X,
        X_compressed,
        model
    )
    compression = model.get_compression_ratio(X)
    quantization_time = fit_time + compression_time

    try:
        export_result = model.save_codebooks(
            codes=X_compressed,
            output_dir=codebooks_dir,
            codebook_filename=f"{method}_{dataset}_codebook.fvecs",
            codes_filename=f"{method}_{dataset}_codes.ivecs",
        )
        print(f"Saved codebook to: {export_result['codebook']}")
        if "codes" in export_result:
            print(f"Saved codes to   : {export_result['codes']}")
        codebook_vectors = export_result.get("codebook_vectors")
    except RuntimeError as exc:
        print(f"Warning: FAISS export skipped ({exc})")
        codebook_vectors = None
    except Exception as exc:
        print(f"Warning: Failed to export codebook ({exc})")
        codebook_vectors = None

    metrics = {
        "distortion": distortion,
        "compression": compression,
        "fit_latency_ms": fit_time * 1000.0,
        "compression_latency_ms": compression_time * 1000.0,
        "decompression_latency_ms": decompression_time * 1000.0,
        "quantization_latency_ms": quantization_time * 1000.0,
    }

    qps_metrics = None
    # Always attempt QPS measurement. For RaBitQ, measure_qps falls back to
    # timing model.compress and does not require a codebook.
    try:
        qps_metrics = measure_qps(
            data.queries,
            model=model,
            codebook_vectors=codebook_vectors,
        )
        metrics.update(qps_metrics)
    except Exception as exc:
        print(f"Warning: Failed to measure QPS ({exc})")

    if with_recall:
        recall_metrics = evaluate_recall(data, model)
        metrics.update(recall_metrics)

    log_run(method=method, dataset=dataset, metrics=metrics, db_path=db_path)

    print("\nResults:")
    max_key_len = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k.ljust(max_key_len)} : {v:.4f}" if isinstance(v, float) else f"  {k.ljust(max_key_len)} : {v}")
