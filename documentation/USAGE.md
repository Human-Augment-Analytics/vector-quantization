# Usage Guide

This guide covers the command-line interface for the HAAG Vector Quantization benchmarking tool.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd vector-quantization
pip install -e .
```

## CLI Commands

The tool provides four main commands:

```bash
vq-benchmark run           # Run single benchmark
vq-benchmark sweep         # Run parameter sweep
vq-benchmark precompute-gt # Precompute ground truth k-NN
vq-benchmark plot          # Visualize results
```

---

## Command: `run`

Run a single benchmark configuration.

### Required Parameters

- `--dataset`: Dataset to use (required - no default)
  - Options: `dummy`, `huggingface`, or `msmarco` (when implemented)

### Common Parameters

```bash
--method pq                    # Quantization method (pq or sq)
--num-samples 10000            # Number of samples to use
--num-chunks 8                 # PQ: number of chunks
--num-clusters 256             # PQ: clusters per chunk
--with-recall                  # Compute recall metrics (requires ground truth)
```

### Path Configuration (for SLURM/ICE)

```bash
--codebooks-dir /path/to/dir   # Where to save codebooks
--db-path /path/to/db.db       # SQLite database path
--ground-truth-path /path.npy  # Precomputed ground truth
```

**Environment Variable Support:**
- `$CODEBOOKS_DIR` - Default codebooks directory
- `$DB_PATH` - Default database path

### Examples

**Basic run:**
```bash
vq-benchmark run --dataset dummy --method pq
```

**With custom paths (for SLURM):**
```bash
export CODEBOOKS_DIR=/scratch/$USER/codebooks
export DB_PATH=/scratch/$USER/logs/run.db

vq-benchmark run \
    --dataset dummy \
    --num-samples 100000 \
    --method pq \
    --num-chunks 8 \
    --num-clusters 256 \
    --ground-truth-path /scratch/$USER/msmarco_gt.npy \
    --with-recall
```

---

## Command: `sweep`

Run parameter sweeps to explore compression-distortion trade-offs.

### Parameters

```bash
--method pq                        # Method to sweep
--dataset dummy                    # Dataset to use (required)
--pq-chunks "4,8,16"              # Comma-separated chunk values
--pq-clusters "128,256,512"       # Comma-separated cluster values
--with-recall                      # Compute recall (default: true)
--with-pairwise                    # Compute pairwise distortion (default: true)
--with-rank                        # Compute rank distortion (default: true)
```

### Example

```bash
vq-benchmark sweep \
    --method pq \
    --dataset dummy \
    --num-samples 10000 \
    --pq-chunks "4,8,16,32" \
    --pq-clusters "128,256,512" \
    --with-recall
```

Each sweep gets a unique ID for tracking:
```
🔖 Sweep ID: sweep_20251014_112541_98df3ade
```

---

## Command: `precompute-gt`

Precompute ground truth k-nearest neighbors for large datasets using FAISS.

**Why?** Computing ground truth on-the-fly requires `O(queries × vectors)` memory, which causes OOM errors on large datasets. Precompute once, reuse many times.

### Required Parameters

```bash
--vectors-path /path/to/vectors.npy   # Input vectors (.npy file)
--output-path /path/to/output.npy     # Where to save ground truth
```

### Optional Parameters

```bash
--num-queries 100     # Number of queries (from start of vectors)
--k 100               # Number of nearest neighbors
--use-gpu             # Use GPU if available
--batch-size 1000     # Batch size for processing
```

### Example (for ICE cluster)

```bash
# Step 1: Precompute ground truth (one-time, memory-intensive)
sbatch --mem=64G --time=4:00:00 --wrap="
vq-benchmark precompute-gt \
    --vectors-path /scratch/$USER/msmarco_vectors.npy \
    --output-path /scratch/$USER/msmarco_gt.npy \
    --num-queries 1000 \
    --k 100
"

# Step 2: Use in benchmarks
vq-benchmark run \
    --dataset dummy \
    --ground-truth-path /scratch/$USER/msmarco_gt.npy \
    --with-recall
```

**Output files:**
- `output.npy` - Ground truth indices (shape: num_queries × k)
- `output.distances.npy` - Corresponding distances

---

## Command: `plot`

Generate visualizations from logged benchmark results.

### Parameters

```bash
--output plots                 # Output directory
--db-path logs/benchmark_runs.db  # Database to read from
--format png                   # Format (png, pdf, svg)
--dpi 300                      # Resolution
--sweep-id <id>                # Filter to specific sweep
--separate-methods             # Create separate plots per method
```

### Example

```bash
# Plot all results
vq-benchmark plot

# Plot specific sweep
vq-benchmark plot --sweep-id sweep_20251014_112541_98df3ade

# High-res PDFs for papers
vq-benchmark plot --format pdf --dpi 600
```

**Generated files:**
- `compression_distortion_tradeoff.png`
- `pairwise_distortion.png`
- `rank_distortion.png`
- `recall_comparison.png`
- `comparison_table.txt`

---

## Configuration Priority

For paths, the tool uses this priority order:

1. **CLI argument** (highest priority)
2. **Environment variable** (e.g., `$CODEBOOKS_DIR`)
3. **Default value** (lowest priority)

### Example

```bash
# CLI arg overrides env var
export CODEBOOKS_DIR=/scratch/default
vq-benchmark run --codebooks-dir /scratch/override ...  # Uses /scratch/override

# Env var overrides default
export CODEBOOKS_DIR=/scratch/custom
vq-benchmark run ...  # Uses /scratch/custom

# No CLI arg or env var
vq-benchmark run ...  # Uses ./codebooks (default)
```

---

## Large-Scale Dataset Workflow

For datasets with millions of vectors:

### 1. Precompute Ground Truth (one-time)

```bash
sbatch ground_truth.slurm
```

### 2. Run Benchmarks (many times)

```bash
sbatch benchmark.slurm
```

### 3. Visualize Results

```bash
vq-benchmark plot --db-path /scratch/$USER/logs/merged.db
```

See [SLURM_GUIDE.md](SLURM_GUIDE.md) for detailed ICE cluster instructions.

---

## Tips

### Dataset Size Guidelines

| Dataset Size | Ground Truth Strategy | Memory Required |
|--------------|----------------------|-----------------|
| < 10K vectors | Compute on-the-fly | < 1GB |
| 10K - 100K | Compute on-the-fly or precompute | 1-10GB |
| > 100K vectors | **Must precompute** | > 10GB |
| > 1M vectors | **Must precompute** | Use ICE cluster |

### Recall Metrics Require Ground Truth

If you want `--with-recall`, you must either:
- Have small dataset (< 100K) - computed automatically
- Provide `--ground-truth-path` for large datasets
- Skip recall metrics for large datasets without precomputed GT

### Environment Variables for SLURM

Add to your `~/.bashrc` on ICE:

```bash
export CODEBOOKS_DIR=/scratch/$USER/codebooks
export DB_PATH=/scratch/$USER/logs/benchmarks.db
```

---

## Getting Help

```bash
vq-benchmark --help
vq-benchmark run --help
vq-benchmark sweep --help
vq-benchmark precompute-gt --help
vq-benchmark plot --help
```
