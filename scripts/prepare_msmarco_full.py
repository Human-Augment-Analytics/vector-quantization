"""Prepare MS MARCO full-corpus benchmark inputs.

Reads the official query set from `queries_parquet/queries.parquet` and writes:
  - datasets/msmarco_full/queries.npy       (1677 x 1024 float32)
  - datasets/msmarco_full/ground_truth.npy  (1677 x 1000 int64, from top1k_offsets)

The corpus itself is left as the existing 60-shard float16 directory at
datasets/msmarco-v2.1-embed-english-v3/passages_npy/. The shard loader
(haag_vq.utils.shard_loader) reads it lazily at job start.

Run from any compute or head node; lightweight (reads ~57 MB parquet).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DEFAULT_DATASETS_ROOT = Path(
    "/storage/ice-shared/cs8903onl/vector_quantization/datasets"
)
SOURCE_DIR = DEFAULT_DATASETS_ROOT / "msmarco-v2.1-embed-english-v3"
QUERIES_PARQUET = SOURCE_DIR / "queries_parquet" / "queries.parquet"
PASSAGES_NPY = SOURCE_DIR / "passages_npy"
OUTPUT_DIR = DEFAULT_DATASETS_ROOT / "msmarco_full"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries-parquet",
        type=Path,
        default=QUERIES_PARQUET,
        help="Path to queries.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write queries.npy and ground_truth.npy",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Trim ground truth to this many neighbors per query (max 1000)",
    )
    args = parser.parse_args()

    print(f"Reading {args.queries_parquet} ...")
    table = pq.read_table(
        args.queries_parquet, columns=["emb", "top1k_offsets"]
    )
    n_queries = table.num_rows
    print(f"  {n_queries} queries")

    emb_list = table["emb"].to_pylist()
    queries = np.asarray(emb_list, dtype=np.float32)
    assert queries.shape == (n_queries, 1024), (
        f"unexpected query shape: {queries.shape}"
    )

    offsets_list = table["top1k_offsets"].to_pylist()
    gt = np.asarray(offsets_list, dtype=np.int64)
    assert gt.shape == (n_queries, 1000), (
        f"unexpected gt shape: {gt.shape}"
    )
    if args.top_k < 1000:
        gt = gt[:, : args.top_k].copy()

    norms = np.linalg.norm(queries, axis=1)
    print(
        f"  query norms: min={norms.min():.6f} "
        f"max={norms.max():.6f} mean={norms.mean():.6f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    q_path = args.output_dir / "queries.npy"
    gt_path = args.output_dir / "ground_truth.npy"

    np.save(q_path, queries)
    np.save(gt_path, gt)

    print(f"Wrote {q_path} ({queries.shape}, {queries.dtype})")
    print(f"Wrote {gt_path} ({gt.shape}, {gt.dtype})")

    shards_link = args.output_dir / "train_shards"
    if not shards_link.exists():
        shards_link.symlink_to(PASSAGES_NPY)
        print(f"Linked {shards_link} -> {PASSAGES_NPY}")
    else:
        print(f"Shard symlink already exists at {shards_link}")


if __name__ == "__main__":
    main()
