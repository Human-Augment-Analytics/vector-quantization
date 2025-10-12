#!/usr/bin/env bash
set -euo pipefail

REPO=~/scratch/vector-quantization
DATA=~/scratch/msmarco

mkdir -p "$REPO/data/msmarco"
ln -sf "$DATA/collection.tsv" "$REPO/data/msmarco/collection.tsv"

echo "✅ Symlink created:"
echo "$REPO/data/msmarco/collection.tsv -> $DATA/collection.tsv"
