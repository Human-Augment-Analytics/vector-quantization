# src/haag_vq/benchmarks/run_benchmarks.py
"""Benchmark runner — dispatches all VQ methods through BaseSearchIndex."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from haag_vq.benchmarks.search_bench import (
    benchmark_index,
    compare_methods,
    compute_ground_truth,
    pareto_plot,
    sweep_bpd,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_fvecs(path: Path) -> np.ndarray:
    """Load a .fvecs file into (N, D) float32 array."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    d = int(data[0].view(np.int32))
    return data.reshape(-1, d + 1)[:, 1:].copy()


def _load_ivecs(path: Path) -> np.ndarray:
    """Load a .ivecs file into (N, k) int32 array."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    d = int(data[0])
    return data.reshape(-1, d + 1)[:, 1:].copy()


def load_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load X_train, X_query, and optionally ground truth from dataset_path.

    Accepts:
    - A directory containing train.npy / queries.npy / [groundtruth.npy]
    - A directory containing base.fvecs / query.fvecs / [groundtruth.ivecs]
    - The string 'synthetic' to generate a small synthetic dataset.

    Returns:
        (X_train, X_query, gt_ids) — gt_ids is None when not found on disk.
    """
    if dataset_path == 'synthetic':
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((2000, 64)).astype(np.float32)
        X_query = rng.standard_normal((100, 64)).astype(np.float32)
        return X_train, X_query, None

    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {p}")

    # .npy format
    if (p / 'train.npy').exists():
        X_train = np.load(p / 'train.npy').astype(np.float32)
        X_query = np.load(p / 'queries.npy').astype(np.float32)
        gt: np.ndarray | None = None
        if (p / 'groundtruth.npy').exists():
            gt = np.load(p / 'groundtruth.npy').astype(np.int64)
        return X_train, X_query, gt

    # .fvecs format
    if (p / 'base.fvecs').exists():
        X_train = _load_fvecs(p / 'base.fvecs')
        X_query = _load_fvecs(p / 'query.fvecs')
        gt = None
        if (p / 'groundtruth.ivecs').exists():
            gt = _load_ivecs(p / 'groundtruth.ivecs').astype(np.int64)
        return X_train, X_query, gt

    raise ValueError(
        f"Could not detect dataset format in {p}. "
        "Expected train.npy+queries.npy or base.fvecs+query.fvecs."
    )


# ---------------------------------------------------------------------------
# Method construction
# ---------------------------------------------------------------------------

AVAILABLE_METHODS = ('pq_flat', 'sq_flat', 'pq_ivf', 'faiss_ivfpq', 'saq')


def build_method_configs(
    method_names: list[str],
    D: int,
    bpd: float,
    K: int = 1024,
    nprobe: int = 64,
) -> dict[str, object]:
    """Instantiate BaseSearchIndex objects for the requested methods.

    Args:
        method_names: Subset of AVAILABLE_METHODS.
        D:            Vector dimensionality (needed to derive PQ sub-spaces).
        bpd:          Target bits per dimension (used to parameterise PQ/SQ).
        K:            Number of IVF centroids.
        nprobe:       IVF nprobe at search time.

    Returns:
        Dict of method_name -> unfitted BaseSearchIndex.
    """
    configs: dict[str, object] = {}

    for name in method_names:
        if name == 'pq_flat':
            try:
                from haag_vq.methods.product_quantization import ProductQuantizer
                from haag_vq.methods.search import FlatQuantizedIndex
                # M = total_bits / bits_per_subcode; clamp to valid range
                nbits_per_sub = 8
                total_bits = int(bpd * D)
                M = max(1, total_bits // nbits_per_sub)
                M = min(M, D)  # M <= D
                configs['pq_flat'] = FlatQuantizedIndex(
                    ProductQuantizer(M=M, B=nbits_per_sub)
                )
            except ImportError as e:
                print(f"WARNING: pq_flat unavailable ({e})", file=sys.stderr)

        elif name == 'sq_flat':
            try:
                from haag_vq.methods.scalar_quantization import ScalarQuantizer
                from haag_vq.methods.search import FlatQuantizedIndex
                # Map bpd to nearest supported bit depth (4, 8, 16)
                if bpd <= 4.5:
                    nb = 4
                elif bpd <= 12:
                    nb = 8
                else:
                    nb = 16
                configs['sq_flat'] = FlatQuantizedIndex(ScalarQuantizer(num_bits=nb))
            except ImportError as e:
                print(f"WARNING: sq_flat unavailable ({e})", file=sys.stderr)

        elif name == 'pq_ivf':
            try:
                from haag_vq.methods.product_quantization import ProductQuantizer
                from haag_vq.methods.search import IvfQuantizedIndex
                nbits_per_sub = 8
                total_bits = int(bpd * D)
                M = max(1, total_bits // nbits_per_sub)
                M = min(M, D)
                configs['pq_ivf'] = IvfQuantizedIndex(
                    quantizer_factory=lambda M=M: ProductQuantizer(M=M, B=8),
                    K=K,
                    nprobe=nprobe,
                )
            except ImportError as e:
                print(f"WARNING: pq_ivf unavailable ({e})", file=sys.stderr)

        elif name == 'faiss_ivfpq':
            try:
                from haag_vq.methods.search import FaissIvfPqIndex
                nbits_per_sub = 8
                total_bits = int(bpd * D)
                m = max(1, total_bits // nbits_per_sub)
                m = min(m, D)
                configs['faiss_ivfpq'] = FaissIvfPqIndex(
                    K=K, m=m, nbits=nbits_per_sub, nprobe=nprobe
                )
            except ImportError as e:
                print(f"WARNING: faiss_ivfpq unavailable ({e})", file=sys.stderr)

        elif name == 'saq':
            try:
                from haag_vq.methods.search import SaqIndex  # type: ignore[attr-defined]
                configs['saq'] = SaqIndex(bpd=bpd, K=K, nprobe=nprobe)
            except (ImportError, AttributeError) as e:
                print(f"WARNING: saq unavailable ({e})", file=sys.stderr)

        else:
            print(f"WARNING: unknown method '{name}' — skipped", file=sys.stderr)

    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='VQ benchmark harness — compares BaseSearchIndex methods.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        default='synthetic',
        help=(
            "Path to dataset directory (expects train.npy+queries.npy or "
            "base.fvecs+query.fvecs), or 'synthetic' for a small generated dataset."
        ),
    )
    parser.add_argument(
        '--methods',
        default=','.join(('pq_flat', 'sq_flat', 'pq_ivf', 'faiss_ivfpq')),
        help='Comma-separated list of methods to benchmark.',
    )
    parser.add_argument(
        '--bpd',
        type=float,
        default=8.0,
        help='Bits per dimension (used to parameterise method compression level).',
    )
    parser.add_argument(
        '--sweep-bpd',
        dest='sweep_bpd_values',
        type=str,
        default=None,
        help=(
            'Comma-separated bpd values for a sweep (e.g. "2,4,8,16"). '
            'When set, a single method must be given via --methods and '
            '--bpd is ignored.'
        ),
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of neighbors for recall and search.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Path to save results CSV. Omit to skip saving.',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show (or save) Pareto plot after benchmarking.',
    )
    parser.add_argument(
        '--plot-save',
        default=None,
        help='File path to save the Pareto plot image (implies --plot).',
    )
    parser.add_argument(
        '--K',
        type=int,
        default=1024,
        help='Number of IVF centroids for IVF-based methods.',
    )
    parser.add_argument(
        '--nprobe',
        type=int,
        default=64,
        help='Number of IVF cells probed at search time.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load data
    print(f"Loading dataset: {args.dataset!r} ...")
    X_train, X_query, gt = load_dataset(args.dataset)
    N, D = X_train.shape
    print(f"  X_train={X_train.shape}  X_query={X_query.shape}")

    # Ground truth
    if gt is None:
        print(f"  Computing brute-force ground truth (k={args.k}) ...")
        gt = compute_ground_truth(X_train, X_query, k=args.k)
    else:
        print(f"  Ground truth loaded: {gt.shape}")

    method_names = [m.strip() for m in args.methods.split(',') if m.strip()]

    # Sweep mode
    if args.sweep_bpd_values is not None:
        if len(method_names) != 1:
            print(
                "ERROR: --sweep-bpd requires exactly one method via --methods.",
                file=sys.stderr,
            )
            sys.exit(1)

        bpd_values = [float(v) for v in args.sweep_bpd_values.split(',')]
        method_name = method_names[0]

        def _factory(bpd: float):
            cfg = build_method_configs(
                [method_name], D=D, bpd=bpd, K=args.K, nprobe=args.nprobe
            )
            if method_name not in cfg:
                raise RuntimeError(
                    f"Could not instantiate method '{method_name}' — see warnings above."
                )
            return cfg[method_name]

        print(f"\nSweeping bpd={bpd_values} for method '{method_name}' ...")
        df = sweep_bpd(_factory, bpd_values, X_train, X_query, gt, k=args.k)

    else:
        # Compare mode
        configs = build_method_configs(
            method_names, D=D, bpd=args.bpd, K=args.K, nprobe=args.nprobe
        )
        if not configs:
            print("ERROR: no methods could be instantiated.", file=sys.stderr)
            sys.exit(1)

        print(f"\nBenchmarking: {list(configs.keys())} (k={args.k}, bpd={args.bpd})")
        df = compare_methods(configs, X_train, X_query, gt, k=args.k)

    # Print results
    print("\n--- Results ---")
    display_cols = [c for c in ('method', 'bpd', 'recall_at_k', 'qps', 'memory_bytes', 'compression_ratio', 'mse') if c in df.columns]
    print(df[display_cols].to_string(index=False))

    # Save CSV
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nSaved results to {out}")

    # Pareto plot
    if args.plot or args.plot_save:
        save = args.plot_save  # may be None → plt.show()
        pareto_plot(df, save_path=save)


if __name__ == '__main__':
    main()
