import click
import math
import typer

from haag_vq.methods.optimized_product_quantization import OptimizedProductQuantizer
from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.rabit_quantization import RaBitQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.methods.variance_adaptive_quantization import VarianceAdaptiveQuantizer
from haag_vq.metrics.distortion import compute_distortion
from haag_vq.metrics.faiss import MetricType
from haag_vq.metrics.recall import evaluate_recall
from haag_vq.data.datasets import Dataset, load_dummy_dataset
from haag_vq.utils.run_logger import log_run

def run(
    method: str = typer.Option("pq", help="Compression method: pq or sq"),
    dataset: str = typer.Option("dummy", help="Dataset name: dummy"),
    num_samples: int = typer.Option(10000),
    dim: int = typer.Option(1024),
    num_chunks: int = typer.Option(8),
    num_clusters: int = typer.Option(256),
    with_distortion: bool = typer.Option(True, help="Whether to compute distortion"),
    with_recall: bool = typer.Option(False, help="Whether to compute Recall@k metrics"),
    distance_metric: str = typer.Option(
        MetricType.L2.name,
        click_type=click.Choice(MetricType._member_names_, case_sensitive=False),
        help="Distance metric type enum for FAISS",
    ),
    min_bits_per_subs: int = typer.Option(1),
    max_bits_per_subs: int = typer.Option(16),
    percent_var_explained: float = typer.Option(1),
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
    elif method == "opq":
        B = int(math.ceil(math.log2(num_clusters)))
        assert 1 << B == num_clusters, f"number of clusters for FAISS OPQ should be a integer power of 2"
        print(f"Fitting OPQ (M={num_chunks} B={B})")
        model = OptimizedProductQuantizer(num_chunks, B)
    elif method == "rabitq":
        model = RaBitQuantizer(MetricType[distance_metric])
    elif method == "vaq":
        model = VarianceAdaptiveQuantizer(num_clusters, num_chunks, min_bits_per_subs, max_bits_per_subs, percent_var_explained)
    else:
        raise ValueError(f"Unsupported method: {method}")

    X = data.vectors
    model.fit(X)
    X_compressed = model.compress(X)

    metrics = dict()
    metrics["compression"] = model.get_compression_ratio(X, X_compressed)
    if with_distortion:
        metrics["distortion"] = compute_distortion(X, X_compressed, model)

    if with_recall:
        recall_metrics = evaluate_recall(data, model)
        metrics.update(recall_metrics)

    log_run(method=method, dataset=dataset, metrics=metrics)

    print("\nResults:")
    max_key_len = max(len(k) for k in metrics)
    for k, v in metrics.items():
        print(f"  {k.ljust(max_key_len)} : {v:.4f}" if isinstance(v, float) else f"  {k.ljust(max_key_len)} : {v}")