#!/bin/bash
# Example parameter sweeps for DBpedia 100K dataset (1536 dimensions)
# This script shows how to run comprehensive benchmarks across all 5 quantization methods

# Set up paths (optional - use defaults if not set)
export CODEBOOKS_DIR="./codebooks"
export DB_PATH="./logs/benchmark_runs.db"

# Note: The DBpedia 100K dataset will be auto-downloaded to ../datasets/ on first run

echo "=========================================="
echo "Running Parameter Sweeps on DBpedia 100K"
echo "=========================================="
echo ""

# 1. PRODUCT QUANTIZATION (PQ)
# Vary number of subquantizers (M) and bits per subquantizer (B)
# M must divide 1536 evenly: common values are 8, 12, 16, 24, 32
echo "1. Running PQ sweep..."
vq-benchmark sweep \
    --method pq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --pq-subquantizers "8,16,32" \
    --pq-bits "6,8" \
    --with-recall \
    --with-pairwise \
    --with-rank

echo ""
echo "PQ sweep complete!"
echo ""

# 2. OPTIMIZED PRODUCT QUANTIZATION (OPQ)
# Same parameters as PQ but with learned rotation
echo "2. Running OPQ sweep..."
vq-benchmark sweep \
    --method opq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --opq-quantizers "8,16,32" \
    --opq-bits "8" \
    --with-recall \
    --with-pairwise \
    --with-rank

echo ""
echo "OPQ sweep complete!"
echo ""

# 3. SCALAR QUANTIZATION (SQ)
# Sweep over different bit widths
echo "3. Running SQ sweep..."
vq-benchmark sweep \
    --method sq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --sq-bits "4,8" \
    --with-recall \
    --with-pairwise \
    --with-rank

echo ""
echo "SQ sweep complete!"
echo ""

# 4. SEGMENTED CAQ (SAQ)
# Option A: Sweep over fixed per-dimension bitwidths
echo "4a. Running SAQ sweep (fixed bits)..."
vq-benchmark sweep \
    --method saq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --saq-num-bits "4,6,8" \
    --saq-allowed-bits "0,2,4,6,8" \
    --with-recall \
    --with-pairwise \
    --with-rank

echo ""

# Option B: Sweep over total bit budgets (more flexible)
echo "4b. Running SAQ sweep (bit budgets)..."
vq-benchmark sweep \
    --method saq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --saq-num-bits "" \
    --saq-total-bits "3072,6144,9216" \
    --saq-allowed-bits "0,2,4,6,8" \
    --with-recall \
    --with-pairwise \
    --with-rank

echo ""
echo "SAQ sweeps complete!"
echo ""

# 5. RaBitQ
# Sweep over distance metrics (if applicable)
echo "5. Running RaBitQ sweep..."
vq-benchmark sweep \
    --method rabitq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --rabitq-metric-type "L2" \
    --with-recall \
    --with-pairwise \
    --with-rank

echo ""
echo "RaBitQ sweep complete!"
echo ""

echo "=========================================="
echo "All sweeps complete!"
echo "=========================================="
echo ""
echo "Generate plots with:"
echo "  vq-benchmark plot"
echo ""
echo "Results saved to: $DB_PATH"
