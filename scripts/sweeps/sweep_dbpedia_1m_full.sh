#!/bin/bash
# Full sweep on DBpedia 1M (1536-dim) with all methods
sbatch --mem=12G --tmp=8G --time=08:00:00 --cpus-per-task=8 -C localNVMe \
    --job-name=vq-dbp1m-full \
    --output=logs/dbp1m_full_%j.out \
    --error=logs/dbp1m_full_%j.err \
    --wrap="source .venv312/bin/activate && python scripts/run_sweep.py --dataset dbpedia-1536 --methods all --output-dir results"
