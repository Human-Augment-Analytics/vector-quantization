#!/bin/bash
#SBATCH --job-name=vq-demo
#SBATCH --output=logs/demo_%j.out
#SBATCH --error=logs/demo_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --tmp=2G
#SBATCH -C localNVMe

# Slurm job script for running vector quantization demo on PACE/ICE
#
# Usage:
#   sbatch slurm_demo.sh
#
# Resource specifications:
#   --mem=4G          : Request 4GB RAM (enough for DBpedia 100K + processing)
#   --cpus-per-task=4 : Use 4 CPU cores (FAISS can use multiple cores)
#   --tmp=2G          : Guarantee 2GB local disk space in $TMPDIR
#   -C localNVMe      : Request node with fast NVMe local storage
#   --time=02:00:00   : Max runtime 2 hours

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Temp directory: $TMPDIR"
echo "=========================================="
echo

# Load Python module (adjust version as needed)
module load python/3.12

# Activate virtual environment
source .venv312/bin/activate

# Print Python and package info
echo "Python version:"
python --version
echo
echo "Key packages:"
pip list | grep -E "numpy|faiss|datasets|torch"
echo
echo "=========================================="
echo

# Run the demo
# $TMPDIR will be automatically used for HuggingFace cache (see demo.py)
python examples/demo.py

# Print completion info
echo
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
