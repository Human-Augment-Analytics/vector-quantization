from pathlib import Path

import typer

from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.metrics.recall import evaluate_recall
from haag_vq.data.datasets import Dataset, load_dummy_dataset
from haag_vq.utils.run_logger import log_run

CODEBOOKS_DIR = Path(__file__).resolve().parents[3] / "codebooks"


def run(
    method: str = typer.Option("pq", help="Compression method: pq or sq"),
    dataset: str = typer.Option("dummy", help="Dataset name: dummy"),
    num_samples: int = typer.Option(10000),
    dim: int = typer.Option(1024),
    num_chunks: int = typer.Option(8),
    num_clusters: int = typer.Option(256),
    with_recall: bool = typer.Option(False, help="Whether to compute Recall@k metrics")
):
    print(f"Loading dataset: {dataset}...")
    if dataset == "dummy":
        data = load_dummy_dataset(num_samples=num_samples, dim=dim)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if method == "pq":
        print(f"Fitting PQ (chunks={num_chunks}, clusters={num_clusters})...")
        model = ProductQuantizer(num_chunks=num_chunks, num_clusters=num_clusters)
    elif method == "sq":
        print("Fitting SQ...")
        model = ScalarQuantizer()
    else:
        raise ValueError(f"Unsupported method: {method}")

    X = data.vectors
    model.fit(X)
    X_compressed = model.compress(X)

    distortion = compute_distortion(X, X_compressed, model)
    compression = model.get_compression_ratio(X)

    try:
        export_result = model.save_codebooks(
            codes=X_compressed,
            output_dir=CODEBOOKS_DIR,
            codebook_filename=f"{method}_{dataset}_codebook.fvecs",
            codes_filename=f"{method}_{dataset}_codes.ivecs",
        )
        print(f"Saved codebook to: {export_result['codebook']}")
        if "codes" in export_result:
            print(f"Saved codes to   : {export_result['codes']}")
    except RuntimeError as exc:
        print(f"Warning: FAISS export skipped ({exc})")
    except Exception as exc:
        print(f"Warning: Failed to export codebook ({exc})")

    metrics = {
        "distortion": distortion,
        "compression": compression,
    }

    if with_recall:
        recall_metrics = evaluate_recall(data, model)
        metrics.update(recall_metrics)

    log_run(method=method, dataset=dataset, metrics=metrics)

    print("\nResults:")
    max_key_len = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k.ljust(max_key_len)} : {v:.4f}" if isinstance(v, float) else f"  {k.ljust(max_key_len)} : {v}")
