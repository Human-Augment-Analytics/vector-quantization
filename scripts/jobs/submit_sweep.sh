#!/bin/bash
#SBATCH --job-name=vq-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# Production Slurm script for vector quantization parameter sweeps
#
# Usage:
#   sbatch submit_sweep.sh <dataset> <methods> [limit]
#
# Examples:
#   sbatch submit_sweep.sh dbpedia-100k "pq opq sq"
#   sbatch submit_sweep.sh dbpedia-1536 "all" 500000
#   sbatch submit_sweep.sh dbpedia-3072 "pq opq" 100000
#
# Or use wrapper scripts:
#   ./sweep_dbpedia_100k.sh
#   ./sweep_dbpedia_1m.sh

# Parse arguments
DATASET=${1:-"dbpedia-100k"}
METHODS=${2:-"all"}
LIMIT=${3:-""}

# Set resources based on dataset
case "$DATASET" in
    "dbpedia-100k")
        MEM="4G"
        TMP="2G"
        TIME="02:00:00"
        CPUS="4"
        ;;
    "dbpedia-1536")
        if [ -n "$LIMIT" ] && [ "$LIMIT" -le 500000 ]; then
            MEM="8G"
            TMP="4G"
        else
            MEM="12G"
            TMP="8G"
        fi
        TIME="06:00:00"
        CPUS="8"
        ;;
    "dbpedia-3072")
        if [ -n "$LIMIT" ] && [ "$LIMIT" -le 500000 ]; then
            MEM="12G"
            TMP="8G"
        else
            MEM="20G"
            TMP="16G"
        fi
        TIME="12:00:00"
        CPUS="8"
        ;;
    "cohere-msmarco")
        MEM="8G"
        TMP="6G"
        TIME="04:00:00"
        CPUS="8"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# Update SBATCH directives dynamically
#SBATCH --mem=$MEM
#SBATCH --tmp=$TMP
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=$CPUS
#SBATCH -C localNVMe

# Create logs and results directories
mkdir -p logs results

# Print job info
echo "=========================================="
echo "VECTOR QUANTIZATION PARAMETER SWEEP"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Temp directory: $TMPDIR"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Methods: $METHODS"
echo "  Limit: ${LIMIT:-'full dataset'}"
echo "  Memory: $MEM"
echo "  CPUs: $CPUS"
echo "  Temp storage: $TMP"
echo "  Time limit: $TIME"
echo "=========================================="
echo ""

# Load Python module
module load python/3.12

# Activate virtual environment
source .venv312/bin/activate

# Print environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "HuggingFace cache: ${TMPDIR}/hf_cache"
echo "=========================================="
echo ""

# Build command
CMD="python scripts/run_sweep.py --dataset $DATASET --methods $METHODS --output-dir results"
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Run sweep
echo "Running: $CMD"
echo ""
eval $CMD
EXIT_CODE=$?

# Print completion info
echo ""
echo "=========================================="
echo "Sweep completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Sweep successful!"
    echo ""
    echo "Results:"
    echo "  📁 Database: logs/benchmark_runs.db"
    echo "  📁 Summary: results/"
    echo ""
    echo "View results:"
    echo "  sqlite3 logs/benchmark_runs.db \"SELECT method, compression_ratio, reconstruction_distortion FROM runs ORDER BY timestamp DESC LIMIT 20;\""
else
    echo "❌ Sweep failed! Check logs/sweep_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
