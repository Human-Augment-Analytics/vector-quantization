# Running Production Sweeps on PACE/ICE

Complete guide for running realistic parameter sweeps on the ICE cluster.

## Quick Start

```bash
# 1. SSH to ICE
ssh <username>@login-ice.pace.gatech.edu

# 2. Navigate to project
cd /storage/ice-shared/cs8903onl/vector_quantization/vector-quantization

# 3. Load Python and activate venv
module load python/3.12
source .venv312/bin/activate

# 4. Submit a sweep
./sweep dbpedia-100k

# 5. Monitor
squeue -u $USER
tail -f logs/dbp100k_*.out
```

## Available Sweep Scripts

### 1. DBpedia 100K (Quick Testing)
```bash
./sweep dbpedia-100k
```
- **Vectors**: 100,000 (1536-dim)
- **Methods**: All 5 methods (PQ, OPQ, SQ, SAQ, RaBitQ)
- **Resources**: 4GB RAM, 2GB NVMe, 4 CPUs
- **Time**: ~2 hours
- **Use for**: Quick validation, testing new methods

### 2. DBpedia 1M Subset (500K vectors)
```bash
./sweep dbpedia-1m-subset
```
- **Vectors**: 500,000 (1536-dim)
- **Methods**: PQ, OPQ, SQ
- **Resources**: 8GB RAM, 6GB NVMe, 8 CPUs
- **Time**: ~4 hours
- **Use for**: Balanced testing, good quality results

### 3. DBpedia 1M Full (Production)
```bash
./sweep dbpedia-1m-full
```
- **Vectors**: 1,000,000 (1536-dim)
- **Methods**: All 5 methods
- **Resources**: 12GB RAM, 8GB NVMe, 8 CPUs
- **Time**: ~8 hours
- **Use for**: Final benchmarks, publication-quality results

## Custom Sweeps

### Using Python Directly

```bash
# Specific methods on DBpedia 100K
python scripts/run_sweep.py --dataset dbpedia-100k --methods pq opq sq

# DBpedia 1M with 300K limit
python scripts/run_sweep.py --dataset dbpedia-1536 --limit 300000 --methods all

# Skip ground truth metrics (faster)
python scripts/run_sweep.py --dataset dbpedia-1536 --methods pq opq --skip-ground-truth

# Help
python scripts/run_sweep.py --help
```

### Custom Slurm Job

```bash
# Submit with custom parameters
sbatch --mem=8G --tmp=6G --time=04:00:00 --cpus-per-task=8 -C localNVMe \
    --job-name=my-sweep \
    --output=logs/my-sweep_%j.out \
    --error=logs/my-sweep_%j.err \
    --wrap="source .venv312/bin/activate && python scripts/run_sweep.py --dataset dbpedia-1536 --limit 500000 --methods pq opq sq"
```

## Resource Guidelines

### By Dataset

| Dataset | Full Size | Memory | Temp Disk | CPUs | Time | Recommended Limit |
|---------|-----------|--------|-----------|------|------|-------------------|
| **dbpedia-100k** | 100K | 4 GB | 2 GB | 4 | 2 hrs | Use full |
| **dbpedia-1536** | 1M | 12 GB | 8 GB | 8 | 8 hrs | 500K for testing |
| **dbpedia-3072** | 1M | 20 GB | 16 GB | 8 | 12 hrs | 300K for testing |
| **cohere-msmarco** | 53M | 8 GB | 6 GB | 8 | 4 hrs | 100K subset |

### When to Use --limit

- **Testing**: Always use limit (e.g., 10K-100K)
- **Development**: Use 100K-500K
- **Production**: Use full dataset or large subset (500K-1M)

### Resource Calculation

```
Memory needed ≈ num_vectors × dimensions × 4 bytes × 1.5 / 1024³ GB
Temp disk ≈ memory + 2 GB (for cache)
Time ≈ num_configs × (training_time + metrics_time)
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### Watch live output
```bash
tail -f logs/<job_output>.out
```

### Check resource usage (after completion)
```bash
seff <job_id>
```

### Cancel job
```bash
scancel <job_id>
```

## Viewing Results

### Quick View (Latest Results)
```bash
sqlite3 logs/benchmark_runs.db "
SELECT method, compression_ratio, reconstruction_distortion, recall@10
FROM runs
ORDER BY timestamp DESC
LIMIT 20;
"
```

### By Sweep ID
```bash
# Get latest sweep ID
SWEEP_ID=$(sqlite3 logs/benchmark_runs.db "SELECT sweep_id FROM runs ORDER BY timestamp DESC LIMIT 1;")

# View all results from that sweep
sqlite3 logs/benchmark_runs.db "
SELECT method, config, compression_ratio, reconstruction_distortion
FROM runs
WHERE sweep_id='$SWEEP_ID'
ORDER BY compression_ratio DESC;
"
```

### Generate Plots
```bash
# After sweep completes
vq-benchmark plot --sweep-id <sweep_id> --output-dir plots/

# Or plot all recent results
vq-benchmark plot --output-dir plots/
```

### Export to CSV
```bash
sqlite3 -header -csv logs/benchmark_runs.db "
SELECT * FROM runs WHERE sweep_id='<sweep_id>';
" > results/sweep_results.csv
```

## Typical Workflow

### 1. Initial Testing (10 minutes)
```bash
# Quick test with tiny subset
python scripts/run_sweep.py --dataset dbpedia-100k --limit 10000 --methods pq sq --output-dir results/test
```

### 2. Development (1-2 hours)
```bash
# Realistic test with 100K vectors
./sweep dbpedia-100k
```

### 3. Production (8-12 hours)
```bash
# Full sweep for publication
./sweep dbpedia-1m-full

# Or 3072-dim for high-dimensional testing
sbatch --mem=20G --tmp=16G --time=12:00:00 --cpus-per-task=8 -C localNVMe \
    --job-name=vq-dbp-3072 \
    --wrap="source .venv312/bin/activate && python scripts/run_sweep.py --dataset dbpedia-3072 --methods pq opq sq --output-dir results"
```

## Configuration Details

### `run_sweep.py` automatically:
- ✅ Uses `$TMPDIR` for fast local NVMe storage
- ✅ Generates sweep-specific configurations based on dimension
- ✅ Computes ground truth if dataset ≤ 100K vectors
- ✅ Logs all results to database
- ✅ Handles errors gracefully (continues on failure)
- ✅ Saves summary to results directory

### Method Configurations Generated

**PQ (Product Quantization)**:
- M values: First 4 divisors of dimension (e.g., 8, 12, 16, 24 for 1536)
- B values: 6, 8 bits per code

**OPQ (Optimized PQ)**:
- M values: First 3 divisors of dimension
- B: 8 bits

**SQ (Scalar Quantization)**:
- Default 8-bit quantization

**SAQ (Segmented CAQ)**:
- Per-dimension: 4, 6, 8 bits
- Total bits: 2×dim, 4×dim

**RaBitQ**:
- L2 metric (default)

## Troubleshooting

### Job Killed / Out of Memory
```bash
# Increase memory
sbatch --mem=16G ...

# Or reduce dataset
python scripts/run_sweep.py --dataset dbpedia-1536 --limit 300000 ...
```

### Slow Download
```bash
# Pre-download dataset (one-time)
python -c "from haag_vq.data import load_dbpedia_openai_1536_100k; load_dbpedia_openai_1536_100k(cache_dir='.cache/datasets')"
```

### Job Timeout
```bash
# Increase time limit
sbatch --time=12:00:00 ...

# Or reduce methods
python scripts/run_sweep.py --methods pq sq  # Skip slower methods
```

### Permission Denied
```bash
# Ensure logs and results directories exist
mkdir -p logs results

# Check you're on compute node (not login node)
echo $TMPDIR  # Should be set
```

## Advanced Usage

### Array Jobs (Parallel Sweeps)
```bash
# Run multiple datasets in parallel
sbatch --array=1-3 --wrap="
case \$SLURM_ARRAY_TASK_ID in
    1) DATASET=dbpedia-100k ;;
    2) DATASET=dbpedia-1536 LIMIT=500000 ;;
    3) DATASET=dbpedia-3072 LIMIT=300000 ;;
esac
source .venv312/bin/activate
python scripts/run_sweep.py --dataset \$DATASET --limit \$LIMIT --methods all
"
```

### Custom Method Configuration

Edit `run_sweep.py` and modify `get_sweep_configs()` function to add/remove configurations.

### Email Notifications
```bash
sbatch --mail-type=END,FAIL --mail-user=your-email@gatech.edu ./sweep dbpedia-100k
```

## Best Practices

1. **Start small**: Test with `--limit 10000` first
2. **Use wrapper scripts**: They have correct resource allocations
3. **Monitor first job**: Watch output to catch issues early
4. **Check resource usage**: Use `seff` to optimize future jobs
5. **Save sweep IDs**: Record them for plotting and analysis
6. **Regular backups**: Copy `logs/benchmark_runs.db` periodically

## File Organization

```
vector-quantization/
├── run_sweep.py              # Main sweep script
├── sweep_dbpedia_100k.sh     # Quick sweep (2 hrs)
├── sweep_dbpedia_1m_subset.sh # Medium sweep (4 hrs)
├── sweep_dbpedia_1m_full.sh   # Full sweep (8 hrs)
├── logs/
│   ├── benchmark_runs.db     # All results (SQLite)
│   ├── dbp100k_*.out         # Job outputs
│   └── dbp100k_*.err         # Job errors
├── results/
│   └── <sweep_id>_summary.txt # Sweep summaries
└── plots/                    # Generated visualizations
```

## Getting Help

1. **Check job logs**: `cat logs/*_<jobid>.err`
2. **Resource usage**: `seff <jobid>`
3. **Queue status**: `squeue -u $USER`
4. **PACE docs**: https://docs.pace.gatech.edu/
5. **This README**: You're reading it!

## Quick Reference

```bash
# Submit quick test
./sweep dbpedia-100k

# Submit production run
./sweep dbpedia-1m-full

# Check status
squeue -u $USER

# Watch progress
tail -f logs/dbp100k_*.out

# View results
sqlite3 logs/benchmark_runs.db "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 10;"

# Cancel job
scancel <jobid>

# Check resource usage
seff <jobid>
```

## Next Steps

After sweep completes:
1. View results in database
2. Generate plots with `vq-benchmark plot`
3. Analyze trade-offs (compression vs quality)
4. Run additional sweeps with different parameters
5. Export data for paper/presentation
