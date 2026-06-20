"""Build an MSMARCO retrieval benchmark dir from the local Cohere v2.1 raw .npy
shards (3.75M passages, 1024-d, float16, unit-normalized). Offline, no download.

base  = first N passages (the corpus)
query = next M passages, held out (pseudo-queries, disjoint from base)
Writes base.fvecs + query.fvecs (float32) in dbpedia-style layout. Ground truth
is computed at run time by the harness (normalized_ground_truth).

Usage:
  python scripts/prep_msmarco_bench.py --n 1000000 --queries 10000 \
      --out /mnt/e/.../SAQ/data/datasets/msmarco_1m
Memory-careful: shards are mmap'd float16; rows are cast to float32 and written
in chunks (no multi-GB peak beyond one chunk + the open file).
"""
import argparse
import glob
import os

import numpy as np

SHARDS_DIR = ("/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/research-hub/experiments/"
              "2026-05-29-sizing-validation/data/hf_cache/"
              "datasets--CohereLabs--msmarco-v2.1-embed-english-v3/snapshots/"
              "e78737fe92ac1b783211b705c12207ca75fcc9b7/passages_npy")


def _open_shards():
    shards = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(SHARDS_DIR + "/*.npy"))]
    total = sum(s.shape[0] for s in shards)
    dim = shards[0].shape[1]
    return shards, total, dim


def _row(shards, i):
    for s in shards:
        if i < s.shape[0]:
            return s[i]
        i -= s.shape[0]
    raise IndexError


def _write_fvecs(path, shards, start, count, dim, chunk=50000):
    """Stream rows [start, start+count) as float32 fvecs (no large peak)."""
    with open(path, "wb") as f:
        written = 0
        while written < count:
            m = min(chunk, count - written)
            block = np.empty((m, dim), dtype=np.float32)
            for j in range(m):
                block[j] = np.asarray(_row(shards, start + written + j), dtype=np.float32)
            out = np.empty((m, dim + 1), dtype=np.float32)
            out[:, 0] = np.frombuffer(np.int32(dim).tobytes(), dtype=np.float32)[0]
            out[:, 1:] = block
            f.write(out.tobytes())
            written += m
            print(f"  {path}: {written}/{count}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000, help="corpus size (base)")
    ap.add_argument("--queries", type=int, default=10_000, help="held-out pseudo-queries")
    ap.add_argument("--out", required=True, help="output dataset dir")
    args = ap.parse_args()

    shards, total, dim = _open_shards()
    print(f"shards: {total} rows x {dim}, need {args.n}+{args.queries}={args.n + args.queries}", flush=True)
    assert args.n + args.queries <= total, "not enough passages for n+queries"
    os.makedirs(args.out, exist_ok=True)

    print("writing base.fvecs ...", flush=True)
    _write_fvecs(os.path.join(args.out, "base.fvecs"), shards, 0, args.n, dim)
    print("writing query.fvecs (held-out) ...", flush=True)
    _write_fvecs(os.path.join(args.out, "query.fvecs"), shards, args.n, args.queries, dim)
    with open(os.path.join(args.out, "metadata.txt"), "w") as f:
        f.write(f"source=CohereLabs/msmarco-v2.1-embed-english-v3 raw .npy shards\n"
                f"n_base={args.n} n_queries={args.queries} dim={dim} "
                f"queries=held-out-pseudo unit_normalized=yes dtype=float32\n")
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
