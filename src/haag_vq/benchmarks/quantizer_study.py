# src/haag_vq/benchmarks/quantizer_study.py
"""Driver: loop (method, bpd), exact normalized-IP search, emit metrics.

Public entrypoints:
- run_study_arrays(X, Q, ...) -> DataFrame   (in-memory; used by tests)
- run_study(config) -> DataFrame             (loads dataset from StudyConfig)
- main()                                     (CLI)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from haag_vq.benchmarks.exact_search import (
    build_scaled_ip_index,
    compute_exact_norms,
    normalized_ground_truth,
    reconstruction_mse,
    recall_at_ks,
    search_index,
)
from haag_vq.benchmarks.quantizer_adapters import Quantizer
from haag_vq.benchmarks.study_config import StudyConfig, load_study_config


def _build_quantizer(method: str, bpd: float, D: int) -> Quantizer:
    from haag_vq.benchmarks.method_registry import build_quantizer
    return build_quantizer(method, bpd=bpd, D=D)


def run_study_arrays(
    X: np.ndarray,
    Q: np.ndarray,
    methods: Sequence[str],
    bpd_values: Sequence[float],
    ks: Tuple[int, ...] = (1, 10, 100),
    chunk_size: int = 50_000,
    mse_sample: int = 100_000,
) -> pd.DataFrame:
    X = np.ascontiguousarray(X, dtype=np.float32)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    n, D = X.shape
    nq = Q.shape[0]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    norms = compute_exact_norms(X)
    gt = normalized_ground_truth(X, Q, k=max(ks), norms=norms, chunk=chunk_size)

    rng = np.random.default_rng(0)
    if n > mse_sample:
        sample_ids = rng.choice(n, mse_sample, replace=False).astype(np.uint32)
    else:
        sample_ids = np.arange(n, dtype=np.uint32)

    rows: List[dict] = []
    for method in methods:
        for bpd in bpd_values:
            q = _build_quantizer(method, bpd, D)
            q.fit(X)

            index = build_scaled_ip_index(
                q.reconstruct, n=n, d=D, norms=norms, chunk=chunk_size
            )
            _, ids = search_index(index, Q, k=max(ks))
            rec = recall_at_ks(ids, gt, ks=ks)
            mse = reconstruction_mse(X, q.reconstruct, sample_ids, chunk=chunk_size)

            code_bytes = q.code_bytes()
            compression = (n * D * 4) / code_bytes if code_bytes else float("inf")

            row = {
                "method": method,
                "bpd": bpd,
                "compression_factor": compression,
                "code_bytes": code_bytes,
                "mse": mse,
                "n_db": n,
                "n_queries": nq,
                "D": D,
                "timestamp": ts,
            }
            for k in ks:
                row[f"recall_at_{k}"] = rec[k]
            rows.append(row)
            del index, q
    return pd.DataFrame(rows)


def _load_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"_load_fvecs: file is empty: {path}")
    d = int(data[0].view(np.int32))
    try:
        return data.reshape(-1, d + 1)[:, 1:].copy()
    except ValueError:
        raise ValueError(
            f"_load_fvecs: size {data.size} floats not divisible by (d+1)={d + 1}: {path}"
        )


def run_study(config: StudyConfig) -> pd.DataFrame:
    X = _load_fvecs(config.dataset["base_fvecs"])
    Q = _load_fvecs(config.dataset["query_fvecs"])
    n_queries = int(config.dataset.get("n_queries", Q.shape[0]))
    Q = Q[:n_queries]
    df = run_study_arrays(
        X, Q,
        methods=config.methods,
        bpd_values=config.bpd,
        ks=tuple(config.ks),
        chunk_size=config.chunk_size,
        mse_sample=config.mse_sample,
    )
    df.insert(0, "dataset", config.dataset.get("name", "unknown"))
    return df


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="VQ quantizer benchmark study")
    ap.add_argument("--config", required=True, help="Path to study YAML config")
    ap.add_argument("--plot", action="store_true", help="Also write Pareto plots")
    args = ap.parse_args(argv)

    cfg = load_study_config(args.config)
    df = run_study(cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"results_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    print(df.to_string(index=False))

    if args.plot:
        from haag_vq.benchmarks.study_plots import pareto_curves
        pareto_curves(df, out_dir / f"pareto_{stamp}.png", ks=tuple(cfg.ks))
        print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
