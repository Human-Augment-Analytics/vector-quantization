# MS MARCO Streaming Compression Guide

This guide covers how to process the massive MS MARCO dataset (53M vectors) using a streaming approach.

## The Challenge

MS MARCO v2.1 contains **53.2 million passage vectors** (~1024 dimensions each):
- **Full dataset in memory**: `53M × 1024 × 4 bytes = ~210 GB`
- **Not feasible** to load entirely into RAM on most systems

## The Solution: Streaming with Two-Phase Processing

### Phase 1: Train on Representative Subset
Load a manageable subset (100K-1M vectors) to train the quantizer:
- **100K vectors**: ~410 MB (quick testing)
- **1M vectors**: ~4 GB (production quality)

### Phase 2: Compress Full Dataset in Batches
Stream the full 53M dataset and compress in small batches:
- **Batch size**: 10K vectors (~40 MB per batch)
- **Total batches**: 5,300 batches for full dataset
- **Peak memory**: Training subset + one batch (~4-5 GB total)

## Quick Start

### Demo (First 100K vectors)
```bash
python examples/demo_msmarco_streaming.py
```

This will:
1. Train on 100K vectors (~500 MB)
2. Compress first 100K in 10 batches (demo mode)
3. Save compressed batches to `compressed_msmarco/`

### Full Dataset (53M vectors)
```bash
sbatch slurm_msmarco.sh
```

Or modify the demo script:
```python
demo_msmarco_streaming(
    training_size=1_000_000,      # Train on 1M (~5 GB)
    compression_batch_size=10_000, # Process 10K at a time
    max_compression_batches=None,  # Process all 53M
)
```

## Implementation Details

### Training Phase

```python
# Load training subset (streaming to avoid downloading all 53M)
training_data = load_cohere_msmarco_passages(
    limit=1_000_000,     # Only load 1M
    streaming=True,      # Stream from HuggingFace
    cache_dir=cache_dir,
)

# Train quantizer
model = ProductQuantizer(M=16, B=8)
model.fit(training_data.vectors)
```

**Memory**: ~5 GB for 1M training vectors

### Compression Phase

```python
# Stream full dataset
ds = load_dataset(
    "Cohere/msmarco-v2.1-embed-english-v3",
    "passages",
    streaming=True,  # Don't load all 53M!
)

# Process in batches
batch_size = 10_000
for i, batch in enumerate(batched(ds, batch_size)):
    # Convert batch to numpy
    batch_array = np.array([item['emb'] for item in batch])

    # Compress
    compressed = model.compress(batch_array)

    # Save batch
    np.save(f"compressed_batch_{i:04d}.npy", compressed)
```

**Memory**: ~40 MB per batch + model overhead

## Resource Requirements

### For Full 53M Dataset

| Resource | Requirement | Reason |
|----------|-------------|--------|
| **Memory** | 12 GB | 5 GB training + 1 GB batch + 6 GB overhead |
| **Local Storage** | 10 GB | HuggingFace cache + compressed output |
| **Time** | 8-12 hours | ~5300 batches × 5-10 sec/batch |
| **CPUs** | 8 cores | Parallel compression (FAISS) |
| **Storage Type** | NVMe | Fast I/O for streaming |

### For Testing (100K subset)

| Resource | Requirement |
|----------|-------------|
| **Memory** | 2 GB |
| **Local Storage** | 2 GB |
| **Time** | 30 min |
| **CPUs** | 4 cores |

## Memory Breakdown

### Training Subset (1M vectors)
```
Vectors: 1,000,000 × 1,024 × 4 bytes = 4.1 GB
Queries: 100 × 1,024 × 4 bytes      = ~0.4 MB
Python overhead                     = ~1 GB
Total                               ≈ 5 GB
```

### Compression Batch (10K vectors)
```
Input batch:  10,000 × 1,024 × 4 bytes  = 40 MB
Compressed:   10,000 × (M×B bits)       = variable
Python overhead                          = ~10 MB
Total per batch                          ≈ 50 MB
```

### Peak Memory During Run
```
Training vectors in memory  = 5 GB
+ Current batch            = 50 MB
+ Model parameters         = ~100 MB
+ Python overhead          = ~1 GB
+ Buffer                   = ~1 GB
Total peak                 ≈ 7-8 GB
```

Request **12 GB** to be safe.

## Output Format

Compressed batches are saved as:
```
compressed_msmarco/
├── pq_compressed_batch_0000.npy   # First 10K vectors
├── pq_compressed_batch_0001.npy   # Next 10K vectors
├── ...
└── pq_compressed_batch_5299.npy   # Last batch (53M / 10K = 5300)
```

Each `.npy` file contains compressed codes for one batch:
- **PQ**: Shape `(10000, M)` where M is number of subspaces
- **SQ**: Shape `(10000, dims)` with quantized values

## Slurm Job Script

See [slurm_msmarco.sh](../slurm_msmarco.sh):

```bash
#!/bin/bash
#SBATCH --job-name=vq-msmarco
#SBATCH --mem=12G
#SBATCH --cpus-per-task=8
#SBATCH --tmp=10G
#SBATCH -C localNVMe
#SBATCH --time=12:00:00

module load python/3.12
source .venv312/bin/activate
python examples/demo_msmarco_streaming.py
```

Submit with:
```bash
sbatch slurm_msmarco.sh
```

Monitor with:
```bash
squeue -u $USER
tail -f logs/msmarco_*.out
```

## Performance Tips

### 1. Optimize Batch Size
```python
# Smaller batches = less memory, more I/O overhead
batch_size = 5_000   # ~20 MB, safer

# Larger batches = more memory, less I/O overhead
batch_size = 50_000  # ~200 MB, faster
```

### 2. Use Local Storage
The code automatically uses `$TMPDIR` on PACE/ICE:
- Fast NVMe disk for HuggingFace cache
- Fast writes for compressed batches
- Automatic cleanup after job

### 3. Parallel Processing
Request multiple CPUs for FAISS parallelization:
```bash
#SBATCH --cpus-per-task=16  # More cores = faster compression
```

### 4. Save to Scratch Storage
For large compressed output:
```python
output_dir = "/scratch/username/compressed_msmarco"
os.makedirs(output_dir, exist_ok=True)
```

## Downstream Usage

### Load Compressed Batches

```python
import numpy as np
from glob import glob

# Load all compressed batches
compressed_files = sorted(glob("compressed_msmarco/pq_compressed_batch_*.npy"))
compressed_codes = [np.load(f) for f in compressed_files]

# Concatenate if needed (watch memory!)
# all_codes = np.concatenate(compressed_codes)  # 53M rows
```

### Search with Compressed Representation

```python
import faiss

# Create index from compressed codes
# (Requires product quantizer trained earlier)
index = faiss.IndexPQ(dimension, M, B)
index.pq = trained_pq  # Use trained PQ

# Add compressed codes
for codes_batch in compressed_codes:
    index.add_codes(codes_batch)

# Search
D, I = index.search(query_vectors, k=10)
```

## Troubleshooting

### Issue: "Out of Memory during training"

**Solution**: Reduce training size
```python
training_size=500_000  # Down from 1M
```

### Issue: "Streaming is very slow"

**Solution**: Check network/storage
```bash
# Use fast local storage
echo $TMPDIR  # Should be set on compute nodes

# Check I/O speed
dd if=/dev/zero of=$TMPDIR/test bs=1M count=1000
```

### Issue: "Job killed after 12 hours"

**Solution**: Increase time limit or reduce dataset
```bash
#SBATCH --time=24:00:00  # 24 hours

# Or process fewer vectors
max_compression_batches=2000  # Only 20M vectors
```

### Issue: "Permission denied writing compressed batches"

**Solution**: Write to writable location
```python
output_dir = os.path.join(os.environ.get("TMPDIR", "."), "compressed")
os.makedirs(output_dir, exist_ok=True)
```

## Comparison: Training Size vs Quality

| Training Size | Memory | Training Time | Compression Quality |
|---------------|--------|---------------|---------------------|
| 100K | 500 MB | ~1 min | Good for testing |
| 500K | 2 GB | ~5 min | Good |
| 1M | 4 GB | ~10 min | Very good |
| 5M | 20 GB | ~1 hour | Excellent (not practical) |

**Recommendation**: Use 1M for production, 100K for quick testing.

## Methods Supported

All quantization methods work with streaming:

1. **Product Quantization (PQ)**: Best for streaming (efficient compression)
2. **Optimized PQ (OPQ)**: Works but slower training
3. **Scalar Quantization (SQ)**: Very fast, good for initial tests
4. **SAQ**: Works but requires more memory during training
5. **RaBitQ**: Experimental, may need adjustments

## Complete Example Workflow

```bash
# 1. Test with small subset locally
python examples/demo_msmarco_streaming.py

# 2. Check output
ls compressed_msmarco/
sqlite3 logs/benchmark_runs.db "SELECT * FROM runs WHERE dataset='msmarco-streaming';"

# 3. Submit full job to cluster
sbatch slurm_msmarco.sh

# 4. Monitor progress
squeue -u $USER
tail -f logs/msmarco_*.out

# 5. After completion, analyze
python -c "
import numpy as np
from glob import glob
files = sorted(glob('compressed_msmarco/*.npy'))
print(f'Compressed {len(files)} batches')
print(f'Total size: {sum(os.path.getsize(f) for f in files) / 1024**3:.2f} GB')
"
```

## Future Improvements

Potential optimizations for even larger datasets:

1. **Distributed compression**: Split across multiple compute nodes
2. **GPU acceleration**: Use FAISS GPU for faster compression
3. **Incremental training**: Update quantizer as more data is seen
4. **On-the-fly compression**: Compress during download (no storage needed)

## Questions?

- Check logs: `logs/msmarco_*.err`
- Verify resources: `seff <jobid>`
- Review code: [demo_msmarco_streaming.py](../examples/demo_msmarco_streaming.py)
- See general guide: [RUNNING_ON_PACE.md](RUNNING_ON_PACE.md)
