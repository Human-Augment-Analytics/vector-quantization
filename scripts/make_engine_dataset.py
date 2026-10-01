"""Convert a harness dataset into SAQ-engine format for flat (K=1, no-IVF) GPU runs.

Inputs: harness fvecs (vectors.fvecs, queries.fvecs), the exact-PCA cache
(npz: mu, V, var — pca_moments.py output / VQ_PCA_CACHE format), and the
harness GT cache (npz with gt ids — run_full_benchmark's gt_cache). GT ids
transfer because the data is unit-norm: the harness's normalized-IP ranking
equals the engine's L2 ranking, and PCA rotation preserves both.

Outputs in --out-dir (engine conventions):
    vectors_pca.fvecs    (x - mu) @ V, float32, chunked
    queries_pca.fvecs    first --n-queries query rows, transformed
    variances_pca.fvecs  1 x D row = var (PCA-basis variances)
    centroids_1_pca.fvecs  single zero centroid (PCA output is centered)
    cluster_ids_1.ivecs    all-zero assignments
    groundtruth.ivecs      GT ids from the harness cache

Usage:
    python scripts/make_engine_dataset.py \
        --data-dir $VQ_DATA_ROOT/msmarco_2m --pca-cache .../pca_msmarco_2m.npz \
        --gt-cache .../gt_cache_n2000000_q20000_k100_mse100000.npz \
        --out-dir $ENGINE_DATA_ROOT/msmarco_2m --n-queries 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHUNK = 100_000


def transform_fvecs(src: Path, dst: Path, mu: np.ndarray, V: np.ndarray,
                    max_rows: int | None = None) -> int:
    d = int(np.fromfile(src, dtype=np.int32, count=1)[0])
    assert d == len(mu), f"{src}: dim {d} != PCA dim {len(mu)}"
    done = 0
    hdr = np.array([d], dtype=np.int32)
    Vf = V.astype(np.float32)
    muf = mu.astype(np.float32)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while max_rows is None or done < max_rows:
            take = CHUNK if max_rows is None else min(CHUNK, max_rows - done)
            raw = np.fromfile(fi, dtype=np.float32, count=take * (d + 1))
            if raw.size == 0:
                break
            rows = raw.reshape(-1, d + 1)
            assert np.all(rows[:, :1].view(np.int32) == d), f"{src}: corrupt header"
            xp = (rows[:, 1:] - muf) @ Vf
            out = np.empty((xp.shape[0], d + 1), dtype=np.float32)
            out[:, :1] = hdr.view(np.float32)
            out[:, 1:] = xp
            out.tofile(fo)
            done += xp.shape[0]
    return done


def write_fvecs_row(dst: Path, row: np.ndarray) -> None:
    with open(dst, "wb") as f:
        np.array([len(row)], dtype=np.int32).tofile(f)
        row.astype(np.float32).tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--pca-cache", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-queries", required=True, type=int)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.pca_cache)
    mu, V, var = z["mu"], z["V"], z["var"]
    D = len(mu)
    assert D % 64 == 0, f"D={D} not a multiple of the engine's 64-dim padding"

    n = transform_fvecs(args.data_dir / "vectors.fvecs", out / "vectors_pca.fvecs", mu, V)
    print(f"vectors_pca.fvecs: {n} rows x {D}")
    nq = transform_fvecs(args.data_dir / "queries.fvecs", out / "queries_pca.fvecs",
                         mu, V, max_rows=args.n_queries)
    assert nq == args.n_queries, f"only {nq} queries available, wanted {args.n_queries}"
    print(f"queries_pca.fvecs: {nq} rows")

    write_fvecs_row(out / "variances_pca.fvecs", var)
    write_fvecs_row(out / "centroids_1_pca.fvecs", np.zeros(D))  # PCA output is centered

    ids = np.zeros((n, 2), dtype=np.int32)
    ids[:, 0] = 1  # ivecs row header: dim=1
    ids.tofile(out / "cluster_ids_1.ivecs")
    print("cluster_ids_1.ivecs, centroids_1_pca.fvecs, variances_pca.fvecs written")

    gt = np.load(args.gt_cache)["gt"]
    assert gt.shape[0] >= args.n_queries, f"GT has {gt.shape[0]} queries < {args.n_queries}"
    gt = gt[: args.n_queries].astype(np.int32)
    k = gt.shape[1]
    rows = np.empty((gt.shape[0], k + 1), dtype=np.int32)
    rows[:, 0] = k
    rows[:, 1:] = gt
    rows.tofile(out / "groundtruth.ivecs")
    print(f"groundtruth.ivecs: {gt.shape[0]} x {k}")


if __name__ == "__main__":
    main()
