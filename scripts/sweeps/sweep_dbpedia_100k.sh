#!/bin/bash
# Quick sweep on DBpedia 100K with all methods
sbatch --mem=4G --tmp=2G --time=02:00:00 --cpus-per-task=4 -C localNVMe \
    --job-name=vq-dbp100k \
    --output=logs/dbp100k_%j.out \
    --error=logs/dbp100k_%j.err \
    --wrap="source .venv312/bin/activate && python scripts/run_sweep.py --dataset dbpedia-100k --methods all --output-dir results"
