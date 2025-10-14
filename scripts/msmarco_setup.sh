#!/usr/bin/env bash
set -euo pipefail

# Configure paths via environment variables or defaults
REPO=${VQ_REPO:-~/scratch/vector-quantization}
DATA=${MSMARCO_DATA:-~/scratch/msmarco}

mkdir -p "$REPO/data/msmarco"
ln -sf "$DATA/collection.tsv" "$REPO/data/msmarco/collection.tsv"

echo "Symlink created:"
echo "$REPO/data/msmarco/collection.tsv -> $DATA/collection.tsv"

echo ""
echo "To customize paths, set environment variables:"
echo "  export VQ_REPO=/your/custom/path"
echo "  export MSMARCO_DATA=/your/data/path"
