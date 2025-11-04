#!/bin/bash
# Sweep on DBpedia 1M (500K subset) with PQ and OPQ
sbatch --mem=8G --tmp=6G --time=04:00:00 --cpus-per-task=8 -C localNVMe \
    --job-name=vq-dbp1m-500k \
    --output=logs/dbp1m_500k_%j.out \
    --error=logs/dbp1m_500k_%j.err \
    --wrap="source .venv312/bin/activate && python scripts/run_sweep.py --dataset dbpedia-1536 --limit 500000 --methods pq opq sq --output-dir results"
