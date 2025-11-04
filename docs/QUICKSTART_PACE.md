# PACE/ICE Quickstart Guide

**Get your first sweep running in 5 minutes.**

## Step 1: SSH to ICE

```bash
ssh <your-username>@login-ice.pace.gatech.edu
```

## Step 2: Navigate to Project

```bash
cd /storage/ice-shared/cs8903onl/vector_quantization/vector-quantization
```

## Step 3: Setup Environment (First Time Only)

```bash
# Load Python 3.12
module load python/3.12

# Check if venv exists
if [ ! -d ".venv312" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv312
    source .venv312/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    echo "Virtual environment already exists"
    source .venv312/bin/activate
fi

# Create directories
mkdir -p logs results
```

## Step 4: Run Your First Sweep

```bash
# Submit quick test sweep (100K vectors, ~2 hours)
./sweep dbpedia-100k
```

You'll see output like:
```
Submitted batch job 12345678
```

## Step 5: Monitor Your Job

```bash
# Check if it's running
squeue -u $USER

# Watch live output
tail -f logs/dbp100k_*.out

# Press Ctrl+C to stop watching (job keeps running)
```

## Step 6: View Results (After Completion)

```bash
# Check if job finished
squeue -u $USER  # Should be empty when done

# View results
sqlite3 logs/benchmark_runs.db "
SELECT method, compression_ratio, reconstruction_distortion
FROM runs
ORDER BY timestamp DESC
LIMIT 10;
"
```

## Done! 🎉

You've just run a complete parameter sweep across 5 quantization methods on 100K vectors.

---

## What Just Happened?

1. **Submitted Slurm job** requesting compute node with:
   - 4 GB RAM
   - 2 GB fast NVMe local storage
   - 4 CPU cores
   - 2 hour time limit

2. **Automatically**:
   - Downloaded DBpedia 100K dataset to fast local storage
   - Trained and benchmarked PQ, OPQ, SQ, SAQ, and RaBitQ
   - Computed all metrics (compression, distortion, recall)
   - Saved results to SQLite database

3. **Generated results** in:
   - `logs/benchmark_runs.db` - All metrics
   - `results/<sweep_id>_summary.txt` - Summary
   - `logs/dbp100k_<jobid>.out` - Full output log

---

## Next Steps

### Run Larger Sweeps

```bash
# 500K vectors (~4 hours)
./sweep dbpedia-1m-subset

# Full 1M vectors (~8 hours)
./sweep dbpedia-1m-full
```

### Custom Sweep

```bash
# Example: Test specific methods on 300K vectors
sbatch --mem=8G --tmp=6G --time=04:00:00 --cpus-per-task=8 -C localNVMe \
    --job-name=my-sweep \
    --output=logs/my-sweep_%j.out \
    --wrap="source .venv312/bin/activate && python scripts/run_sweep.py --dataset dbpedia-1536 --limit 300000 --methods pq opq sq"
```

### View All Options

```bash
python scripts/run_sweep.py --help
```

### Generate Plots

```bash
# After sweep completes, generate visualizations
vq-benchmark plot --output-dir plots/
```

---

## Common Commands

```bash
# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# Check resource usage (after completion)
seff <job_id>

# View latest results
sqlite3 logs/benchmark_runs.db "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 20;"

# Watch live output
tail -f logs/<job_output_file>.out
```

---

## Troubleshooting

### "Command not found"
```bash
# Make sure you're in the right directory
cd /storage/ice-shared/cs8903onl/vector_quantization/vector-quantization

# Make scripts executable
chmod +x sweep_*.sh run_sweep.py
```

### "Python module not found"
```bash
# Load Python and activate venv
module load python/3.12
source .venv312/bin/activate

# Reinstall if needed
pip install -e .
```

### Job killed/Out of memory
```bash
# Use smaller dataset or more memory
python scripts/run_sweep.py --dataset dbpedia-100k --limit 50000 --methods pq sq
```

### Want to test quickly first?
```bash
# Interactive session for testing (1 hour)
srun --mem=4G --tmp=2G --time=1:00:00 --cpus-per-task=4 -C localNVMe --pty bash

# Then run interactively
module load python/3.12
source .venv312/bin/activate
python scripts/run_sweep.py --dataset dbpedia-100k --limit 10000 --methods pq
```

---

## Full Documentation

- **[RUNNING_SWEEPS.md](RUNNING_SWEEPS.md)** - Complete sweep guide
- **[RUNNING_ON_PACE.md](documentation/RUNNING_ON_PACE.md)** - PACE/ICE details
- **[MEMORY_OPTIMIZATIONS.md](documentation/MEMORY_OPTIMIZATIONS.md)** - Memory details
- **[DATASETS.md](documentation/DATASETS.md)** - Dataset information

---

## Quick Reference Card

| Task | Command |
|------|---------|
| **Quick sweep** | `./sweep dbpedia-100k` |
| **Check status** | `squeue -u $USER` |
| **Watch output** | `tail -f logs/dbp100k_*.out` |
| **View results** | `sqlite3 logs/benchmark_runs.db "SELECT * FROM runs LIMIT 10;"` |
| **Cancel job** | `scancel <jobid>` |
| **Custom sweep** | `python scripts/run_sweep.py --dataset <name> --methods <list>` |
| **Help** | `python scripts/run_sweep.py --help` |

---

That's it! You're ready to run production sweeps on PACE/ICE. 🚀
