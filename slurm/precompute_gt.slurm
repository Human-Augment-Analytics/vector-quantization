#!/bin/bash
#SBATCH --job-name=precompute_gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/precompute_gt_%j.out
#SBATCH --error=/scratch/%u/logs/precompute_gt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@gatech.edu

# =============================================================================
# Precompute Ground Truth for Vector Quantization Benchmarks
# =============================================================================
# This job precomputes k-nearest neighbors for queries using FAISS.
# Run this ONCE before running benchmarks that require recall metrics.
#
# Usage:
#   1. Update YOUR_EMAIL above
#   2. Update DATA_PATH and OUTPUT_PATH below
#   3. Submit: sbatch slurm/precompute_gt.slurm
# =============================================================================

# Load required modules
module load python/3.11

# Activate virtual environment if using one
# source ~/venv/bin/activate

# Set paths (CUSTOMIZE THESE)
DATA_PATH="/scratch/$USER/data/msmarco_vectors.npy"
OUTPUT_PATH="/scratch/$USER/data/msmarco_gt.npy"
NUM_QUERIES=1000
K=100

# Ensure output directory exists
mkdir -p $(dirname $OUTPUT_PATH)
mkdir -p /scratch/$USER/logs

# Print job info
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================================"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Num queries: $NUM_QUERIES"
echo "K neighbors: $K"
echo "================================================================"

# Run ground truth computation
vq-benchmark precompute-gt \
    --vectors-path $DATA_PATH \
    --output-path $OUTPUT_PATH \
    --num-queries $NUM_QUERIES \
    --k $K \
    --batch-size 5000

EXIT_CODE=$?

echo "================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Ground truth saved to $OUTPUT_PATH"
    ls -lh $OUTPUT_PATH*
else
    echo "FAILED: Check error log for details"
fi

exit $EXIT_CODE
