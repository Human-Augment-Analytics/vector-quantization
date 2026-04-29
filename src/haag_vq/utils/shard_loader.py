"""Load a directory of .npy shards into a single contiguous float32 ndarray.

Designed for MS MARCO-style corpora where the embeddings are split across many
.npy files (often float16 to save disk) and the consumer needs a single
in-memory float32 matrix (faiss.train/add, SAQ construct, etc.).

Memory: peak is full_size_float32 + one_shard_float32 (~7-9 GB transient).
Wall time: dominated by sequential disk read; ~30 min for 113M x 1024 on PACE
shared storage.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np


def list_shards(shard_dir: Path) -> list[Path]:
    """Return sorted list of .npy files in shard_dir (lexicographic order)."""
    shards = sorted(p for p in Path(shard_dir).iterdir() if p.suffix == ".npy")
    if not shards:
        raise FileNotFoundError(f"No .npy shards found in {shard_dir}")
    return shards


def shard_metadata(shards: list[Path]) -> Tuple[int, int, np.dtype]:
    """Inspect shard headers (memmap, no full read) to get total_rows, dim, dtype."""
    total_rows = 0
    dim = None
    dtype = None
    for path in shards:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{path} has shape {arr.shape}, expected 2D")
        if dim is None:
            dim = arr.shape[1]
            dtype = arr.dtype
        elif arr.shape[1] != dim:
            raise ValueError(
                f"{path} has dim {arr.shape[1]}, expected {dim}"
            )
        total_rows += arr.shape[0]
    assert dim is not None and dtype is not None
    return total_rows, dim, dtype


def load_sharded_corpus(
    shard_dir: str | Path,
    out_dtype: np.dtype = np.float32,
    verbose: bool = True,
) -> np.ndarray:
    """Concatenate all .npy shards in shard_dir into one in-memory ndarray.

    Args:
        shard_dir: Path to directory containing .npy shards (sorted by name).
        out_dtype: Output dtype. Conversion happens shard-by-shard.
        verbose: Print per-shard progress.

    Returns:
        ndarray of shape (total_rows, dim), dtype=out_dtype, contiguous.
    """
    shards = list_shards(Path(shard_dir))
    total_rows, dim, src_dtype = shard_metadata(shards)

    out_bytes = total_rows * dim * np.dtype(out_dtype).itemsize
    if verbose:
        print(
            f"shard_loader: {len(shards)} shards, total {total_rows:,} x {dim} "
            f"({src_dtype} -> {out_dtype}), allocating "
            f"{out_bytes / 1e9:.1f} GB"
        )

    out = np.empty((total_rows, dim), dtype=out_dtype)

    cursor = 0
    t_start = perf_counter()
    for i, path in enumerate(shards):
        t0 = perf_counter()
        shard = np.load(path)  # full read; one shard ~3-4 GB
        n = shard.shape[0]
        out[cursor : cursor + n] = shard.astype(out_dtype, copy=False)
        cursor += n
        if verbose:
            elapsed = perf_counter() - t0
            total_elapsed = perf_counter() - t_start
            print(
                f"  [{i + 1}/{len(shards)}] {path.name}: {n:,} rows in "
                f"{elapsed:.1f}s (cumulative {total_elapsed / 60:.1f} min)"
            )
        del shard

    assert cursor == total_rows, (
        f"row count mismatch: filled {cursor}, expected {total_rows}"
    )
    if verbose:
        print(
            f"shard_loader: done in {(perf_counter() - t_start) / 60:.1f} min"
        )
    return out
