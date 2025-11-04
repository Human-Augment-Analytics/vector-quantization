# SLURM Guide for ICE Cluster

This guide provides instructions for running large-scale vector quantization benchmarks on Georgia Tech's ICE cluster using SLURM.

> **Note:** For complete CLI reference, see [USAGE.md](USAGE.md). This guide focuses on SLURM-specific configuration and job submission.

---

## Quick Start

```bash
# 1. SSH to ICE
ssh <username>@login-ice.pace.gatech.edu

# 2. Clone repo to your home directory
cd ~
git clone <repo-url>
cd vector-quantization

# 3. Set up environment
module load python/3.11
pip install --user -e .

# 4. Set paths
export CODEBOOKS_DIR=/scratch/$USER/codebooks
export DB_PATH=/scratch/$USER/logs/benchmarks.db

# 5. Submit job
sbatch slurm/benchmark.slurm
```

---

## Directory Structure on ICE

Recommended organization:

```
/home/<username>/
└── vector-quantization/          # Code repo (git tracked)

/scratch/<username>/
├── data/
│   ├── msmarco_vectors.npy       # Your dataset vectors
│   ├── msmarco_queries.npy       # Query vectors
│   └── msmarco_gt.npy            # Precomputed ground truth
├── codebooks/                     # Generated codebooks
│   ├── pq_msmarco_codebook.fvecs
│   └── pq_msmarco_codes.ivecs
└── logs/
    ├── benchmarks.db              # SQLite results database
    └── slurm_*.out                # SLURM job outputs
```

**Why this structure?**
- `/home/` - Small quota, for code only
- `/scratch/` - Large quota, for data and results
- SLURM can access both locations

---

## Environment Setup

### 1. Create Module Load Script

Create `~/vector-quantization/setup_env.sh`:

```bash
#!/bin/bash
# Load required modules
module load python/3.11
module load cuda/11.8  # If using GPU

# Set environment variables
export CODEBOOKS_DIR=/scratch/$USER/codebooks
export DB_PATH=/scratch/$USER/logs/benchmarks.db
export PYTHONPATH=$HOME/vector-quantization/src:$PYTHONPATH

# Optional: Set FAISS to use all CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

### 2. Add to `.bashrc` (optional)

```bash
echo "source ~/vector-quantization/setup_env.sh" >> ~/.bashrc
```

---

## SLURM Job Scripts

### Template 1: Precompute Ground Truth

`slurm/precompute_gt.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=precompute_gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/$USER/logs/precompute_gt_%j.out
#SBATCH --error=/scratch/$USER/logs/precompute_gt_%j.err

# Load environment
source ~/vector-quantization/setup_env.sh

# Ensure output directory exists
mkdir -p /scratch/$USER/logs

# Run ground truth computation
vq-benchmark precompute-gt \
    --vectors-path /scratch/$USER/data/msmarco_vectors.npy \
    --output-path /scratch/$USER/data/msmarco_gt.npy \
    --num-queries 1000 \
    --k 100 \
    --batch-size 5000

echo "Ground truth computation complete!"
echo "Output: /scratch/$USER/data/msmarco_gt.npy"
```

**Submit:**
```bash
sbatch slurm/precompute_gt.slurm
```

---

### Template 2: Single Benchmark Run

`slurm/benchmark.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=vq_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/$USER/logs/benchmark_%j.out
#SBATCH --error=/scratch/$USER/logs/benchmark_%j.err

# Load environment
source ~/vector-quantization/setup_env.sh

# Create output directories
mkdir -p /scratch/$USER/logs
mkdir -p /scratch/$USER/codebooks

# Set database path to include job ID (for uniqueness)
export DB_PATH=/scratch/$USER/logs/benchmark_$SLURM_JOB_ID.db

# Run benchmark
vq-benchmark run \
    --dataset dummy \
    --num-samples 1000000 \
    --method pq \
    --num-chunks 8 \
    --num-clusters 256 \
    --ground-truth-path /scratch/$USER/data/msmarco_gt.npy \
    --with-recall \
    --codebooks-dir $CODEBOOKS_DIR \
    --db-path $DB_PATH

echo "Benchmark complete!"
echo "Results: $DB_PATH"
echo "Codebooks: $CODEBOOKS_DIR"
```

**Submit:**
```bash
sbatch slurm/benchmark.slurm
```

---

### Template 3: Parameter Sweep

`slurm/sweep.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=vq_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/$USER/logs/sweep_%j.out
#SBATCH --error=/scratch/$USER/logs/sweep_%j.err

# Load environment
source ~/vector-quantization/setup_env.sh

# Create output directories
mkdir -p /scratch/$USER/logs
mkdir -p /scratch/$USER/codebooks

# Run parameter sweep
vq-benchmark sweep \
    --method pq \
    --dataset dummy \
    --num-samples 500000 \
    --pq-chunks "4,8,16,32" \
    --pq-clusters "64,128,256,512" \
    --with-recall \
    --with-pairwise \
    --with-rank \
    --codebooks-dir $CODEBOOKS_DIR \
    --db-path $DB_PATH

echo "Sweep complete! Check logs for sweep ID."
```

**Submit:**
```bash
sbatch slurm/sweep.slurm
```

---

### Template 4: Job Array for Parallel Sweeps

`slurm/sweep_array.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=vq_sweep_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=1-12
#SBATCH --output=/scratch/$USER/logs/sweep_array_%A_%a.out
#SBATCH --error=/scratch/$USER/logs/sweep_array_%A_%a.err

# Load environment
source ~/vector-quantization/setup_env.sh

# Create output directories
mkdir -p /scratch/$USER/logs

# Define parameter combinations
CHUNKS=(4 4 4 8 8 8 16 16 16 32 32 32)
CLUSTERS=(128 256 512 128 256 512 128 256 512 128 256 512)

# Get parameters for this task
IDX=$(($SLURM_ARRAY_TASK_ID - 1))
CHUNK=${CHUNKS[$IDX]}
CLUSTER=${CLUSTERS[$IDX]}

# Unique DB for this task
DB_FILE="/scratch/$USER/logs/sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.db"

echo "Running: chunks=$CHUNK, clusters=$CLUSTER"

# Run benchmark
vq-benchmark run \
    --dataset dummy \
    --num-samples 1000000 \
    --method pq \
    --num-chunks $CHUNK \
    --num-clusters $CLUSTER \
    --ground-truth-path /scratch/$USER/data/msmarco_gt.npy \
    --with-recall \
    --db-path $DB_FILE

echo "Task complete: $DB_FILE"
```

**Submit:**
```bash
sbatch slurm/sweep_array.slurm
```

**Merge results later:**
```bash
# Copy all DBs to one master DB
sqlite3 /scratch/$USER/logs/master.db <<EOF
ATTACH '/scratch/$USER/logs/sweep_12345_1.db' AS db1;
INSERT INTO runs SELECT * FROM db1.runs;
-- Repeat for all task DBs
EOF
```

---

## Monitoring Jobs

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j <job_id>

# View job details
scontrol show job <job_id>
```

### Check Job Output

```bash
# Tail live output
tail -f /scratch/$USER/logs/benchmark_<job_id>.out

# View errors
cat /scratch/$USER/logs/benchmark_<job_id>.err
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel array job
scancel <array_job_id>  # Cancels all tasks
scancel <array_job_id>_<task_id>  # Cancel specific task
```

---

## Resource Guidelines

### Memory Requirements

| Dataset Size | Recommended Memory |
|--------------|-------------------|
| 100K vectors | 8-16GB |
| 1M vectors | 32-64GB |
| 10M vectors | 128-256GB |
| Ground truth computation | 2x dataset memory |

### Time Limits

Typical runtimes:
- **Precompute GT** (1M vectors, 1K queries): 2-4 hours
- **Single benchmark** (1M vectors): 30-60 minutes
- **Parameter sweep** (12 configs, 1M vectors): 6-12 hours

**Set generous time limits** - jobs killed by timeout lose all results!

### CPU Allocation

- **PQ fitting**: CPU-bound, use 16-32 cores
- **Ground truth**: FAISS scales well, use 32+ cores
- **More cores** ≠ faster quantization (diminishing returns > 32)

### GPU Usage

Currently not utilized, but ground truth computation can use GPU:

```bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

vq-benchmark precompute-gt \
    --use-gpu \
    ...
```

---

## Best Practices

### 1. Test Locally First

```bash
# Test with small dataset before submitting large job
vq-benchmark run \
    --dataset dummy \
    --num-samples 1000 \
    --method pq
```

### 2. Use Scratch Space

- **Never** write large files to `/home/` (small quota)
- **Always** use `/scratch/$USER/` for data and results
- Backup important results (scratch is not backed up)

### 3. Unique Database Paths

For job arrays, use unique DB files:
```bash
export DB_PATH=/scratch/$USER/logs/run_$SLURM_JOB_ID.db
```

### 4. Check Disk Space

```bash
# Check scratch usage
du -sh /scratch/$USER/*

# Check scratch quota
quota -s
```

### 5. Debug Interactively

Request interactive node:
```bash
salloc --nodes=1 --cpus-per-task=8 --mem=32G --time=2:00:00
vq-benchmark run --dataset dummy --num-samples 10000
exit
```

---

## Troubleshooting

### Job Fails Immediately

**Check:**
1. Module loads correctly: `module list`
2. Python package installed: `pip show haag-vq`
3. SLURM script syntax: `sbatch --test-only script.slurm`

### Out of Memory Error

**Solutions:**
- Increase `--mem` in SLURM script
- Reduce `--num-samples`
- Use precomputed ground truth
- Skip ground truth: remove `--with-recall`

### Job Timeout

**Solutions:**
- Increase `--time` in SLURM script
- Reduce parameter sweep size
- Split into multiple jobs

### Permission Denied

**Check:**
- File paths are in `/scratch/$USER/`, not `/scratch/`
- Directories exist: `mkdir -p /scratch/$USER/logs`
- File permissions: `chmod +x script.sh`

### FAISS Not Found

```bash
# Install FAISS
pip install --user faiss-cpu

# For GPU support
pip install --user faiss-gpu
```

---

## Example Workflow

### Full Pipeline on ICE

```bash
# 1. Prepare data (one-time)
sbatch slurm/precompute_gt.slurm
# Wait for job: squeue -u $USER

# 2. Run sweeps (multiple times, iterate)
sbatch slurm/sweep.slurm

# 3. Download results to local machine
scp <user>@login-ice.pace.gatech.edu:/scratch/$USER/logs/*.db ./

# 4. Visualize locally
vq-benchmark plot --db-path ./benchmarks.db
```

---

## Additional Resources

- **ICE Documentation**: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042088
- **SLURM Commands**: https://slurm.schedmd.com/quickstart.html
- **PACE Support**: pace-support@oit.gatech.edu

---

## Tips for Long Runs

1. **Use email notifications:**
   ```bash
   #SBATCH --mail-type=END,FAIL
   #SBATCH --mail-user=<your-email>@gatech.edu
   ```

2. **Save intermediate results:**
   Results are logged continuously to SQLite DB - safe to inspect mid-run

3. **Monitor resource usage:**
   ```bash
   seff <job_id>  # After job completes
   ```

4. **Use persistent terminal:**
   ```bash
   screen  # Or tmux - keeps session alive if SSH disconnects
   ```
