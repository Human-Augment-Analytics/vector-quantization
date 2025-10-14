#!/bin/bash
#SBATCH --job-name=vq_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/%u/logs/benchmark_%j.out
#SBATCH --error=/scratch/%u/logs/benchmark_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@gatech.edu

# =============================================================================
# Single Vector Quantization Benchmark
# =============================================================================
# This job runs a single benchmark configuration.
#
# Usage:
#   1. Update YOUR_EMAIL above
#   2. Update parameters in "Benchmark parameters" section below
#   3. Submit: sbatch slurm/benchmark.slurm
# =============================================================================

# Load required modules
module load python/3.11

# Activate virtual environment if using one
# source ~/venv/bin/activate

# Set environment variables for paths
export CODEBOOKS_DIR="/scratch/$USER/codebooks"
export DB_PATH="/scratch/$USER/logs/benchmark_$SLURM_JOB_ID.db"

# Ensure directories exist
mkdir -p $CODEBOOKS_DIR
mkdir -p $(dirname $DB_PATH)
mkdir -p /scratch/$USER/logs

# =============================================================================
# Benchmark parameters (CUSTOMIZE THESE)
# =============================================================================
DATASET="dummy"  # Options: dummy, huggingface, msmarco
NUM_SAMPLES=1000000
METHOD="pq"  # Options: pq, sq
NUM_CHUNKS=8  # PQ only
NUM_CLUSTERS=256  # PQ only
GROUND_TRUTH_PATH="/scratch/$USER/data/msmarco_gt.npy"
WITH_RECALL="--with-recall"  # Remove to skip recall metrics

# Print job info
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"
echo "================================================================"
echo "Dataset: $DATASET"
echo "Samples: $NUM_SAMPLES"
echo "Method: $METHOD"
echo "Chunks: $NUM_CHUNKS"
echo "Clusters: $NUM_CLUSTERS"
echo "================================================================"

# Run benchmark
vq-benchmark run \
    --dataset $DATASET \
    --num-samples $NUM_SAMPLES \
    --method $METHOD \
    --num-chunks $NUM_CHUNKS \
    --num-clusters $NUM_CLUSTERS \
    --ground-truth-path $GROUND_TRUTH_PATH \
    $WITH_RECALL \
    --codebooks-dir $CODEBOOKS_DIR \
    --db-path $DB_PATH

EXIT_CODE=$?

echo "================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Results saved to $DB_PATH"
    echo "Codebooks saved to $CODEBOOKS_DIR"
else
    echo "FAILED: Check error log for details"
fi

exit $EXIT_CODE
