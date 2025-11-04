#!/bin/bash
#SBATCH --job-name=vq-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=4G
#SBATCH -C localNVMe

# Slurm job script for running vector quantization parameter sweeps on PACE/ICE
#
# Usage:
#   sbatch slurm_sweep.sh
#
# For different datasets, adjust resources:
#   DBpedia 100K (1536-dim):  --mem=4G  --tmp=2G
#   DBpedia 1M (1536-dim):    --mem=8G  --tmp=6G
#   DBpedia 1M (3072-dim):    --mem=16G --tmp=12G

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "PACE/ICE Vector Quantization Sweep"
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

# Load Python module (adjust version as needed)
module load python/3.12

# Activate virtual environment
source .venv312/bin/activate

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo
echo "Key packages:"
pip list | grep -E "numpy|faiss|datasets|torch|scikit" || true
echo
echo "HuggingFace cache will use: $TMPDIR/hf_cache"
echo "Dataset cache will use: $TMPDIR/hf_cache/datasets"
echo "=========================================="
echo

# Run the sweep
# The demo.py script will automatically use $TMPDIR for caching
python examples/demo.py

echo
echo "=========================================="
echo "Sweep completed at: $(date)"
echo "=========================================="
echo
echo "Results:"
echo "  Database: logs/benchmark_runs.db"
echo "  Plots: demo_plots/"
echo
echo "To view results:"
echo "  sqlite3 logs/benchmark_runs.db 'SELECT * FROM runs ORDER BY timestamp DESC LIMIT 10;'"
