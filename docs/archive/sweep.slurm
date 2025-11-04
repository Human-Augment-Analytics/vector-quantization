#!/bin/bash
#SBATCH --job-name=vq_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/logs/sweep_%j.out
#SBATCH --error=/scratch/%u/logs/sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@gatech.edu

# =============================================================================
# Parameter Sweep for Vector Quantization
# =============================================================================
# This job runs a parameter sweep across multiple configurations.
# Results are logged with a unique sweep ID for easy visualization.
#
# Usage:
#   1. Update YOUR_EMAIL above
#   2. Update parameters below
#   3. Submit: sbatch slurm/sweep.slurm
#   4. After completion, visualize with:
#      vq-benchmark plot --sweep-id <sweep_id>
# =============================================================================

# Load required modules
module load python/3.11

# Activate virtual environment if using one
# source ~/venv/bin/activate

# Set environment variables
export CODEBOOKS_DIR="/scratch/$USER/codebooks"
export DB_PATH="/scratch/$USER/logs/sweeps.db"

# Ensure directories exist
mkdir -p $CODEBOOKS_DIR
mkdir -p $(dirname $DB_PATH)
mkdir -p /scratch/$USER/logs

# =============================================================================
# Sweep parameters (CUSTOMIZE THESE)
# =============================================================================
DATASET="dummy"  # Options: dummy, huggingface, msmarco
NUM_SAMPLES=500000
METHOD="pq"  # Options: pq, sq
PQ_CHUNKS="4,8,16,32"  # Comma-separated
PQ_CLUSTERS="64,128,256,512"  # Comma-separated
GROUND_TRUTH_PATH="/scratch/$USER/data/msmarco_gt.npy"

# Metric flags (use --no-with-X to disable)
WITH_RECALL="--with-recall"
WITH_PAIRWISE="--with-pairwise"
WITH_RANK="--with-rank"

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
echo "PQ Chunks: $PQ_CHUNKS"
echo "PQ Clusters: $PQ_CLUSTERS"
echo "================================================================"

# Run parameter sweep
vq-benchmark sweep \
    --method $METHOD \
    --dataset $DATASET \
    --num-samples $NUM_SAMPLES \
    --pq-chunks $PQ_CHUNKS \
    --pq-clusters $PQ_CLUSTERS \
    $WITH_RECALL \
    $WITH_PAIRWISE \
    $WITH_RANK \
    --codebooks-dir $CODEBOOKS_DIR \
    --db-path $DB_PATH

EXIT_CODE=$?

echo "================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Results saved to $DB_PATH"
    echo ""
    echo "To visualize this sweep:"
    echo "  1. Copy DB to local machine:"
    echo "     scp $USER@login-ice.pace.gatech.edu:$DB_PATH ./"
    echo "  2. Generate plots:"
    echo "     vq-benchmark plot --db-path $(basename $DB_PATH)"
else
    echo "FAILED: Check error log for details"
fi

exit $EXIT_CODE
