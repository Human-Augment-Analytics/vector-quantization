#!/bin/bash
#SBATCH --job-name=vq-msmarco
#SBATCH --output=logs/msmarco_%j.out
#SBATCH --error=logs/msmarco_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=10G
#SBATCH -C localNVMe

# Slurm job script for MS MARCO streaming compression (53M vectors)
#
# Usage:
#   sbatch slurm_msmarco.sh
#
# Resource specifications:
#   --mem=12G         : 12GB RAM (enough for 1M training + batches)
#   --cpus-per-task=8 : 8 CPU cores for parallel processing
#   --tmp=10G         : 10GB local NVMe storage for cache
#   -C localNVMe      : Fast NVMe local storage
#   --time=12:00:00   : Max 12 hours (for full 53M dataset)
#
# Strategy:
#   1. Train on 1M vector subset (~5 GB memory)
#   2. Stream and compress full 53M in batches (~50 MB per batch)
#   3. Save compressed batches to $TMPDIR (fast local storage)
#   4. Copy results back to network storage at end

# Create logs directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "MS MARCO Streaming Compression"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Temp directory: $TMPDIR"
echo "Allocated memory: $SLURM_MEM_PER_NODE MB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="
echo

# Load Python
module load python/3.12

# Activate venv
source .venv312/bin/activate

# Print environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo
echo "HuggingFace cache: $TMPDIR/hf_cache"
echo "Output directory: compressed_msmarco/"
echo "=========================================="
echo

# Run streaming compression
# Modify the demo script to process full dataset
python examples/demo_msmarco_streaming.py

echo
echo "=========================================="
echo "Compression complete at: $(date)"
echo "=========================================="
echo
echo "Results:"
echo "  Compressed batches: compressed_msmarco/"
echo "  Database: logs/benchmark_runs.db"
echo
echo "Next steps:"
echo "  1. Analyze compression ratios in database"
echo "  2. Use compressed batches for downstream tasks"
echo "  3. Compute search metrics on compressed representation"
