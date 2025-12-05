# Vector Quantization Benchmarking

Benchmark 5 vector quantization methods with full parameter sweep support. Clean, tested, production-ready.

---

## What This Does

Systematically test vector compression methods across parameter ranges:
- **5 methods**: PQ, OPQ, SQ, SAQ, RaBitQ  
- **4 datasets**: DBpedia (100K-1M), MS MARCO (53M)
- **Parameter sweeps**: Automatic multi-configuration testing
- **PACE/ICE ready**: Slurm integration, NVMe caching, environment variables

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd vector-quantization
pip install -e .
```

### 2. Run Locally

```bash
# Quick test - PQ with default parameters
vq-benchmark sweep --dataset dbpedia-100k --method pq

# Test multiple PQ configurations
vq-benchmark sweep --dataset dbpedia-100k --method pq --pq-subquantizers "8,16,32"

# Limit dataset size
vq-benchmark sweep --dataset dbpedia-1536 --dataset-limit 500000 --method opq
```

### 3. Run on PACE with Slurm

```bash
# Quick test (100K, 2 hrs, 4GB RAM, 5GB local disk on NVMe)
sbatch --mem=4G --time=2:00:00 --tmp=5G -C localNVMe \
    --wrap="vq-benchmark sweep --dataset dbpedia-100k --method pq"

# PQ parameter sweep (500K, 4 hrs, 8GB RAM, 10GB NVMe)
sbatch --mem=8G --time=4:00:00 --tmp=10G -C localNVMe \
    --wrap="vq-benchmark sweep --dataset dbpedia-1536 --dataset-limit 500000 --method pq --pq-subquantizers '8,16,32'"

# Full production (1M, 8 hrs, 12GB RAM, 20GB NVMe)
sbatch --mem=12G --time=8:00:00 --tmp=20G -C localNVMe \
    --wrap="vq-benchmark sweep --dataset dbpedia-1536 --method opq --opq-quantizers '8,16,32'"
```

### 4. View Results

```bash
sqlite3 logs/benchmark_runs.db
sqlite> SELECT method, config, compression_ratio, mse, recall_at_10
        FROM benchmark_runs ORDER BY timestamp DESC LIMIT 10;
```

---

## Available Datasets

| Name | Vectors | Dims | Memory | Use Case |
|------|---------|------|--------|----------|

| Name | Vectors | Dims | Memory | Use Case |
|------|---------|------|--------|----------|
| `dbpedia-100k` | 100K | 1536 | ~1 GB | Quick testing |
| `dbpedia-1536` | 1M | 1536 | ~6 GB | Production |
| `dbpedia-3072` | 1M | 3072 | ~12 GB | High-dimensional |
| `cohere-msmarco` | 53M | 1024 | ~200 GB | Full 53M: use `streaming-sweep` |

**Common options:**
- `--dataset-limit INT`: Limit vectors loaded (for regular `sweep` command)
- `--cache-dir PATH`: HuggingFace cache (default: `../datasets`)

**MS MARCO Options:**
1. **Subset testing**: Use `sweep` with `--dataset-limit 100000` (up to ~1M fits in memory)
2. **Full 53M dataset**: Use `streaming-sweep` command (batch compression, no memory limit)

---

## Methods & Parameters

### PQ (Product Quantization)

Splits vectors into M subvectors, quantizes each with k-means.

```bash
vq-benchmark sweep --dataset dbpedia-100k --method pq \
    --pq-subquantizers "8,16,32" \    # Number of subvectors (M)
    --pq-bits "8"                      # Bits per subvector (B=8 means 256 clusters)
```

**Parameters:**
- `--pq-subquantizers`: M values, comma-separated (default: `"8,16,32"`)
  - M must divide dimension evenly
  - Higher M = more compression, less accuracy
- `--pq-bits`: B values, comma-separated (default: `"8"`)

**Compression:** 32-512:1 | **Paper:** [Jégou et al., 2011](https://hal.inria.fr/inria-00514462v2/document)

### OPQ (Optimized Product Quantization)

PQ with learned rotation matrix for better accuracy.

```bash
vq-benchmark sweep --dataset dbpedia-100k --method opq \
    --opq-quantizers "8,16,32" \      # Number of quantizers (M)
    --opq-bits "8"                     # Bits per quantizer
```

**Parameters:**
- `--opq-quantizers`: M values (default: `"8,16,32"`)
- `--opq-bits`: Bits (default: `"8"`)

**Compression:** Same as PQ, better accuracy | **Paper:** [Ge et al., 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf)

### SQ (Scalar Quantization)

Quantizes each dimension independently to 8 bits.

```bash
vq-benchmark sweep --dataset dbpedia-100k --method sq \
    --sq-bits "8"                      # Bits per dimension
```

**Parameters:**
- `--sq-bits`: Bits per dimension (default: `"8"`)

**Compression:** 4:1

### SAQ (Segmented Adaptive Quantization)

Adaptive bit allocation based on dimension variance.

```bash
vq-benchmark sweep --dataset dbpedia-100k --method saq \
    --saq-num-bits "4,6,8" \          # Default bits per dimension
    --saq-allowed-bits "0,2,4,6,8" \  # Allowed bit values
    --saq-segments "4,8"               # Number of segments
```

**Parameters:**
- `--saq-num-bits`: Default bitwidth sweep (default: `"4,8"`)
- `--saq-total-bits`: Total bit budget per vector (overrides num-bits)
- `--saq-allowed-bits`: Discrete allowed bitwidths (default: `"0,2,4,6,8"`)
- `--saq-segments`: Segment counts (default: auto)

**Compression:** 8-32:1 | **Paper:** [Zhou et al., 2024](https://arxiv.org/abs/2410.06482)

### RaBitQ

Bit-level quantization with FAISS.

```bash
vq-benchmark sweep --dataset dbpedia-100k --method rabitq \
    --rabitq-metric-type "L2"         # Distance metric
```

**Parameters:**
- `--rabitq-metric-type`: `"L2"` or `"IP"` (inner product)

**Compression:** 100:1+ | **Paper:** [Gao & Long, 2024](https://dl.acm.org/doi/pdf/10.1145/3654970)

---

## Evaluation Options

Control metrics computed:

```bash
vq-benchmark sweep ... \
    --with-recall / --no-with-recall           # k-NN recall (default: on)
    --with-pairwise / --no-with-pairwise       # Pairwise dist (default: on)
    --with-rank / --no-with-rank               # Rank distortion (default: on)
    --num-pairs 1000                           # Pairs for pairwise
    --rank-k 10                                # k for rank distortion
```

---

## PACE/ICE Cluster Integration

### Automatic Environment Detection

The tool automatically detects and uses:
- **Cache priority**: `$TMPDIR` (fast local NVMe) > `/storage/ice-shared/cs8903onl/.cache/huggingface` (shared) > local `.cache`
- **Codebooks**: `$CODEBOOKS_DIR` or `./codebooks`
- **Database**: `$DB_PATH` or `logs/benchmark_runs.db`

**Why $TMPDIR?** ICE compute nodes have fast local NVMe storage (`$TMPDIR`) that's automatically cleared after jobs. This is MUCH faster than network storage for dataset loading. The loaders automatically use it when available.

### Resource Recommendations

| Dataset Size | Time | Memory | Temp Disk | Command |
|--------------|------|--------|-----------|---------|
| 100K | 2 hrs | 4 GB | 5 GB | `sbatch --mem=4G --time=2:00:00 --tmp=5G -C localNVMe --wrap="vq-benchmark sweep --dataset dbpedia-100k --method pq"` |
| 500K | 4 hrs | 8 GB | 10 GB | `sbatch --mem=8G --time=4:00:00 --tmp=10G -C localNVMe --wrap="vq-benchmark sweep --dataset dbpedia-1536 --dataset-limit 500000 --method pq"` |
| 1M | 8 hrs | 12 GB | 20 GB | `sbatch --mem=12G --time=8:00:00 --tmp=20G -C localNVMe --wrap="vq-benchmark sweep --dataset dbpedia-1536 --method pq"` |
| 1M (3072-dim) | 16 hrs | 16 GB | 30 GB | `sbatch --mem=16G --time=16:00:00 --tmp=30G -C localNVMe --wrap="vq-benchmark sweep --dataset dbpedia-3072 --method pq"` |

**Key Slurm flags:**
- `--tmp=<size>G`: Request local disk space
- `-C localNVMe`: Request NVMe storage (faster)
- `-C localSAS`: Request SAS storage (slower but available on more nodes)

**Reference:** [PACE ICE Local Disk Documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094)

### Monitor Jobs

```bash
squeue -u $USER
tail -f slurm-<job_id>.out
ls -lh logs/
```

---

## Understanding Results

Results → `logs/benchmark_runs.db` (SQLite)

### Metrics

- **compression_ratio**: How much smaller (32:1 = 32x)
- **mse**: Mean squared error (lower = better)
- **pairwise_distortion**: Relative distance preservation (lower = better)
- **recall@10**: % true k-NN found (higher = better)

### Query Examples

```sql
-- View recent runs
SELECT method, config, compression_ratio, mse, recall_at_10
FROM benchmark_runs ORDER BY timestamp DESC LIMIT 20;

-- Compare PQ configurations
SELECT config, compression_ratio, recall_at_10
FROM benchmark_runs
WHERE method='pq' AND dataset='dbpedia-1536'
ORDER BY compression_ratio DESC;

-- Best compression-accuracy tradeoff
SELECT method, config, compression_ratio, recall_at_10
FROM benchmark_runs
WHERE dataset='dbpedia-100k'
ORDER BY (compression_ratio * recall_at_10) DESC LIMIT 10;

-- Method comparison
SELECT method, AVG(compression_ratio) as avg_comp, AVG(recall_at_10) as avg_recall
FROM benchmark_runs WHERE dataset='dbpedia-100k' GROUP BY method;
```

---

## Example Workflows

### Test all methods locally

```bash
# Runs on your local machine (no Slurm)
for method in pq opq sq saq rabitq; do
  vq-benchmark sweep --dataset dbpedia-100k --method $method
done

# Visualize - creates comprehensive analysis plots:
# • Compression-distortion tradeoffs
# • Pareto frontier showing optimal configurations
# • Radar chart for multi-dimensional comparison
# • Recall and rank distortion analysis
vq-benchmark plot
```

### MS MARCO Subset sweep (100K-1M vectors)

**Locally:**
```bash
# Run all methods (100K subset)
for method in pq opq sq saq rabitq; do
  vq-benchmark sweep --dataset cohere-msmarco --dataset-limit 100000 --method $method
done

# Visualize results
vq-benchmark plot --dataset cohere-msmarco
```

**On PACE (parallel jobs):**
```bash
# Submit 5 jobs (run in parallel)
for method in pq opq sq saq rabitq; do
  sbatch --mem=8G --time=4:00:00 --tmp=10G -C localNVMe \
    --wrap="vq-benchmark sweep --dataset cohere-msmarco --dataset-limit 100000 --method $method"
done

# After jobs complete
vq-benchmark plot --dataset cohere-msmarco
```

### MS MARCO FULL 53M dataset (streaming batch compression)

**Locally:**
```bash
# Run streaming compression (trains on 1M, compresses all 53M in batches)
vq-benchmark streaming-sweep --method pq --training-size 1000000 --batch-size 10000

# Try different methods
vq-benchmark streaming-sweep --method opq --training-size 1000000
vq-benchmark streaming-sweep --method sq --training-size 500000
```

**On PACE (recommended for full 53M):**
```bash
# PQ on full 53M (~24 hours, 16GB RAM, 50GB NVMe for compressed batches)
sbatch --mem=16G --time=24:00:00 --tmp=50G -C localNVMe \
    --wrap="vq-benchmark streaming-sweep --method pq --training-size 1000000 --batch-size 10000"

# OPQ on full 53M (slower training)
sbatch --mem=20G --time=30:00:00 --tmp=50G -C localNVMe \
    --wrap="vq-benchmark streaming-sweep --method opq --training-size 1000000 --batch-size 10000"
```

**Streaming sweep options:**
- `--training-size`: Vectors to train quantizer (default: 1M)
- `--batch-size`: Batch size for compression (default: 10K)
- `--max-batches`: Limit batches (default: None = all ~5300 batches)
- `--output-dir`: Where to save compressed batches (default: `compressed_msmarco/`)
```

### Deep PQ parameter sweep

```bash
vq-benchmark sweep --dataset dbpedia-1536 --method pq \
    --pq-subquantizers "4,8,12,16,24,32" \
    --pq-bits "6,8" \
    --dataset-limit 500000
```

### Compare PQ vs OPQ

```bash
vq-benchmark sweep --dataset dbpedia-1536 --method pq --pq-subquantizers "8,16,32"
vq-benchmark sweep --dataset dbpedia-1536 --method opq --opq-quantizers "8,16,32"

# Query comparison
sqlite3 logs/benchmark_runs.db "
SELECT method, config, compression_ratio, recall_at_10
FROM benchmark_runs WHERE method IN ('pq','opq') ORDER BY method, config"
```

---

## Ground Truth for Recall Metrics

Ground truth k-nearest neighbors are required for computing recall and rank distortion metrics.

### Automatic Ground Truth (Default)

For datasets ≤100K vectors, ground truth is computed automatically using FAISS:

```bash
# dbpedia-100k: Ground truth computed automatically (fast with FAISS)
vq-benchmark sweep --dataset dbpedia-100k --method pq
```

### Large Datasets (>100K vectors)

For large datasets (1M+), ground truth is **skipped by default** to save memory.

**Option 1: Precompute ground truth separately (recommended)**

```bash
# 1. Save dataset vectors to .npy file first
python -c "
from haag_vq.data import load_dbpedia_openai_1536
import numpy as np
data = load_dbpedia_openai_1536(limit=None)
np.save('dbpedia_1536_vectors.npy', data.vectors)
"

# 2. Precompute ground truth using FAISS (efficient, GPU-accelerated if available)
vq-benchmark precompute-gt \
    --vectors-path dbpedia_1536_vectors.npy \
    --output-path dbpedia_1536_ground_truth.npy \
    --num-queries 100 \
    --k 100

# 3. Run sweep with precomputed ground truth
vq-benchmark sweep --dataset dbpedia-1536 --method pq \
    --ground-truth-path dbpedia_1536_ground_truth.npy
```

**Option 2: Skip recall metrics**

```bash
# Run without recall/rank distortion metrics
vq-benchmark sweep --dataset dbpedia-1536 --method pq \
    --with-recall false --with-rank false
```

**PACE Example (precompute ground truth with GPU):**

```bash
# Use GPU for faster ground truth computation on large datasets
sbatch --mem=32G --time=4:00:00 --gres=gpu:1 \
    --wrap="vq-benchmark precompute-gt \
        --vectors-path /scratch/\$USER/msmarco_vectors.npy \
        --output-path /scratch/\$USER/msmarco_gt.npy \
        --num-queries 1000 \
        --k 100 \
        --use-gpu"
```

---

## Repository Structure

```
vector-quantization/
├── README.md                 # This file (complete documentation)
├── src/haag_vq/
│   ├── cli.py                # CLI entry point (vq-benchmark)
│   ├── benchmarks/
│   │   ├── sweep.py          # Parameter sweep implementation
│   │   └── precompute_ground_truth.py
│   ├── data/                 # Dataset loaders
│   │   ├── dbpedia_loader.py
│   │   └── cohere_msmarco_loader.py
│   ├── methods/              # Quantization implementations
│   │   ├── product_quantization.py        # PQ
│   │   ├── optimized_product_quantization.py  # OPQ
│   │   ├── scalar_quantization.py         # SQ
│   │   ├── saq.py                         # SAQ
│   │   └── rabit_quantization.py          # RaBitQ
│   ├── metrics/              # Evaluation
│   │   ├── distortion.py
│   │   ├── pairwise_distortion.py
│   │   ├── recall.py
│   │   └── rank_distortion.py
│   └── utils/
│       └── run_logger.py     # SQLite logging
└── logs/
    └── benchmark_runs.db     # Results
```

---

## Troubleshooting

**Command not found: vq-benchmark**
```bash
pip install -e .
```

**Out of memory**
```bash
vq-benchmark sweep --dataset dbpedia-100k --method pq  # Use smaller dataset
vq-benchmark sweep --dataset dbpedia-1536 --dataset-limit 100000 --method pq  # Or limit
```

**Slow download**
- First run downloads from HuggingFace (takes time)
- Subsequent runs use cache
- PACE: Auto-uses `$TMPDIR` (NVMe) then `/storage/ice-shared/cs8903onl/.cache/huggingface`

**Python/FAISS**
- Requires Python 3.9+
- Install: `pip install faiss-cpu`
- PACE: `module load python/3.12`

---

## HAAG Research Resources

- 👥 [Roster](https://gtvault.sharepoint.com/:x:/s/HAAG/EbRWUBbmh3pPpGuh9HF34DgBPnJQEdtMQoBTtANXCxOg9Q?e=B8ykCV)
- 📄 [Weekly Report](https://gtvault.sharepoint.com/:w:/s/HAAG/EcKDOtAbNKZEr3KrZDfRlZ4BD_IMA-4hTSc7ll52J6_79A)
- 🎤 [Presentations](https://gtvault-my.sharepoint.com/:x:/g/personal/byu321_gatech_edu/EUB3IKLuDwdLkG5dlPwJoccByYUJ9XJgcngZMbOa8pwq0A)
- 💬 Slack: `#vector-quantization`

### Learning
- 📝 [Project Doc](https://gtvault-my.sharepoint.com/:w:/r/personal/smussmann3_gatech_edu/_layouts/15/Doc.aspx?sourcedoc=%7B805CAAA2-48BB-42CD-A20D-C04F2DA3CA41%7D)
- 📺 [VQ Intro](https://www.youtube.com/watch?v=c36lUUr864M)
- 🔍 [FAISS PQ Chart](https://raw.githubusercontent.com/wiki/facebookresearch/faiss/PQ_variants_Faiss_annotated.png)
- 📚 [Qdrant VQ](https://qdrant.tech/articles/what-is-vector-quantization/)

---
