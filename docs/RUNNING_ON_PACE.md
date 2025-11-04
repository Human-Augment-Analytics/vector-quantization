# Running on PACE/ICE Cluster

This guide covers how to run vector quantization benchmarks on Georgia Tech's PACE/ICE computing cluster with optimal performance and resource usage.

## Quick Start

```bash
# 1. Submit demo job
sbatch slurm_demo.sh

# 2. Check job status
squeue -u $USER

# 3. View output (once complete)
cat logs/demo_<jobid>.out
```

## Key Optimizations for PACE/ICE

### 1. Local Storage ($TMPDIR)

**All code automatically uses `$TMPDIR` when available**, which provides:

- ✅ **Fast NVMe/SAS local disk** (much faster than network storage)
- ✅ **No permission issues** (you own the temp directory)
- ✅ **No file locking conflicts** with shared storage
- ✅ **Automatic cleanup** after job completion

The code detects `$TMPDIR` and automatically stores:
- HuggingFace dataset cache
- Downloaded parquet files
- Temporary processing files

### 2. Memory Pre-allocation

All data loaders use pre-allocated numpy arrays instead of Python lists, reducing peak memory usage by ~50%.

See [MEMORY_OPTIMIZATIONS.md](MEMORY_OPTIMIZATIONS.md) for details.

### 3. FAISS Ground Truth

Ground truth k-NN computation uses FAISS instead of sklearn, reducing memory usage by >99% for this step.

## Resource Requirements

### By Dataset

| Dataset | Vectors | Dimensions | Memory | Temp Disk | Time (estimate) |
|---------|---------|------------|--------|-----------|-----------------|
| DBpedia 100K | 100,000 | 1536 | 4 GB | 2 GB | 1-2 hours |
| DBpedia 1M (1536) | 1,000,000 | 1536 | 8 GB | 6 GB | 3-5 hours |
| DBpedia 1M (3072) | 1,000,000 | 3072 | 16 GB | 12 GB | 5-8 hours |
| **MS MARCO (streaming)** | 53,000,000 | ~1024 | 12 GB | 10 GB | 8-12 hours |

**Note**: MS MARCO uses a **streaming approach** - trains on 1M subset, then compresses full 53M in batches.

**Formula for custom limits:**
- Memory: `num_vectors × dimensions × 4 bytes × 1.5 (overhead) / 1024³ GB`
- Temp disk: `~same as memory` (for dataset cache)

### Storage Types

Request specific storage types with Slurm constraints:

```bash
# Fast NVMe storage (recommended)
#SBATCH -C localNVMe

# SAS storage (alternative)
#SBATCH -C localSAS
```

See [ICE Resources](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094) for node specifications.

## Slurm Scripts

### slurm_demo.sh

Runs the complete demo with all 5 methods on DBpedia 100K.

**Resources:**
- Memory: 4 GB
- CPUs: 4
- Time: 2 hours
- Local storage: 2 GB NVMe

**Usage:**
```bash
sbatch slurm_demo.sh
```

**Customize for different datasets:**
```bash
# For DBpedia 1M (1536-dim)
#SBATCH --mem=8G
#SBATCH --tmp=6G
#SBATCH --time=04:00:00

# For DBpedia 1M (3072-dim)
#SBATCH --mem=16G
#SBATCH --tmp=12G
#SBATCH --time=08:00:00
```

### slurm_sweep.sh

Same as demo but with more resources for larger sweeps.

**Resources:**
- Memory: 8 GB
- CPUs: 8
- Time: 4 hours
- Local storage: 4 GB NVMe

### slurm_msmarco.sh

**Streaming compression for MS MARCO (53M vectors)**

Uses a two-phase approach:
1. **Training phase**: Load 1M representative vectors (~5 GB)
2. **Compression phase**: Stream full 53M dataset in 10K-vector batches

**Resources:**
- Memory: 12 GB
- CPUs: 8
- Time: 12 hours
- Local storage: 10 GB NVMe

**Usage:**
```bash
sbatch slurm_msmarco.sh
```

**Output**: Compressed batches saved to `compressed_msmarco/` directory

See [demo_msmarco_streaming.py](../examples/demo_msmarco_streaming.py) for implementation details.

## Interactive Jobs

For testing and debugging, request an interactive session:

```bash
# DBpedia 100K
srun --mem=4G --cpus-per-task=4 --tmp=2G -C localNVMe --time=1:00:00 --pty bash

# DBpedia 1M (1536-dim)
srun --mem=8G --cpus-per-task=8 --tmp=6G -C localNVMe --time=2:00:00 --pty bash

# Then run
module load python/3.12
source .venv312/bin/activate
python examples/demo.py
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View live output
```bash
tail -f logs/demo_<jobid>.out
```

### Check resource usage
```bash
seff <jobid>
```

### Cancel job
```bash
scancel <jobid>
```

## Common Issues and Solutions

### Issue: Job killed during loading

**Symptom:** Process killed with no error message during "Loading vectors"

**Causes:**
1. Insufficient memory allocated
2. Insufficient temp disk space
3. Running on login node (restricted resources)

**Solutions:**
```bash
# 1. Increase memory and temp disk
#SBATCH --mem=8G
#SBATCH --tmp=6G

# 2. Or reduce dataset size in demo.py
limit=50_000  # Half the vectors

# 3. Always use compute nodes (sbatch/srun), never login nodes
```

### Issue: Permission denied on cache files

**Symptom:** `PermissionError: [Errno 13] Permission denied: .../.lock`

**Cause:** Not using $TMPDIR (trying to write to shared storage)

**Solution:** Code should automatically detect $TMPDIR. If not, check:
```bash
echo $TMPDIR  # Should be set in Slurm jobs
```

### Issue: Slow dataset download

**Symptom:** Dataset download takes very long or times out

**Solutions:**
```bash
# 1. Download once to shared storage (one-time setup)
python -c "from haag_vq.data import load_dbpedia_openai_1536_100k; \
    load_dbpedia_openai_1536_100k(cache_dir='../.cache/datasets')"

# 2. Then in Slurm job, copy to $TMPDIR
cp -r ../.cache/datasets $TMPDIR/

# 3. Or increase timeout in Slurm script
#SBATCH --time=04:00:00
```

### Issue: Out of temp disk space

**Symptom:** `OSError: [Errno 28] No space left on device`

**Cause:** $TMPDIR too small for dataset

**Solution:** Request more temp disk:
```bash
#SBATCH --tmp=10G  # Increase from default
```

## Best Practices

### 1. Test Locally First
```bash
# Quick test with small subset
python -c "
from haag_vq.data import load_dbpedia_openai_1536_100k
data = load_dbpedia_openai_1536_100k(limit=1000, cache_dir='../.cache/datasets')
print(f'✅ Loaded {data.vectors.shape}')
"
```

### 2. Use Interactive Session for First Run
```bash
srun --mem=4G --tmp=2G -C localNVMe --time=1:00:00 --pty bash
# ... then test interactively
```

### 3. Submit Batch Jobs for Production
```bash
sbatch slurm_demo.sh
```

### 4. Monitor Resource Usage
```bash
# After job completes
seff <jobid>
```

Check if you over/under-allocated resources and adjust for next run.

### 5. Use Array Jobs for Multiple Configurations
```bash
#!/bin/bash
#SBATCH --array=1-5
#SBATCH --mem=4G

# Run different configs in parallel
python examples/demo.py --config-id $SLURM_ARRAY_TASK_ID
```

## Example Workflow

### Full Parameter Sweep on DBpedia 100K

```bash
# 1. Clone repo and setup
git clone <repo>
cd vector-quantization
module load python/3.12
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -e .

# 2. Create logs directory
mkdir -p logs

# 3. Submit job
sbatch slurm_demo.sh

# 4. Monitor
squeue -u $USER
tail -f logs/demo_*.out

# 5. View results (after completion)
sqlite3 logs/benchmark_runs.db "SELECT method, compression_ratio, reconstruction_distortion FROM runs ORDER BY timestamp DESC LIMIT 10;"

# 6. Download plots
scp pace:/path/to/vector-quantization/demo_plots/ .
```

## Performance Tips

1. **Use NVMe storage** (`-C localNVMe`) for best I/O performance
2. **Request multiple CPUs** (`--cpus-per-task=8`) - FAISS can parallelize
3. **Pre-download datasets** to shared storage, then copy to $TMPDIR in job
4. **Use appropriate memory** - don't over-allocate (wastes resources) or under-allocate (job killed)
5. **Cache compiled FAISS** - first run compiles, subsequent runs faster

## Troubleshooting

### Check environment variables
```bash
echo "TMPDIR: $TMPDIR"
echo "HF_HOME: $HF_HOME"
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
```

### Check disk space
```bash
df -h $TMPDIR
du -sh $TMPDIR/*
```

### Check memory usage
```bash
# During job (from another terminal)
sstat -j <jobid> --format=JobID,MaxRSS,AveCPU

# After job
seff <jobid>
```

## Additional Resources

- [PACE Documentation](https://docs.pace.gatech.edu/)
- [ICE Resources](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094)
- [Slurm Quick Start](https://slurm.schedmd.com/quickstart.html)
- [Memory Optimizations](MEMORY_OPTIMIZATIONS.md)
- [Dataset Guide](DATASETS.md)

## Getting Help

1. Check logs: `logs/demo_*.err`
2. Check job status: `squeue -u $USER`
3. Check resource usage: `seff <jobid>`
4. Contact PACE support: pace-support@oit.gatech.edu
5. Open GitHub issue for code problems
