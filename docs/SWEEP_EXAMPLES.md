# Parameter Sweep Examples Guide

This guide provides recommended parameter sweep configurations for all implemented quantization methods on common datasets.

---

## Quick Start: DBpedia 100K (1536-dim)

Run the comprehensive benchmark script:
```bash
bash example_sweeps_dbpedia100k.sh
```

Or run individual method sweeps as shown below.

---

## Dataset Considerations

### DBpedia 100K (1536 dimensions)
- **Size**: 100K vectors, manageable for all operations
- **Dimension**: 1536 (divisible by 8, 12, 16, 24, 32, 48, 64)
- **Good for**: Quick experiments, development, testing
- **Ground truth**: Can compute for smaller subsets

### DBpedia 1M (1536 or 3072 dimensions)
- **Size**: 1M vectors, requires ground truth precomputation
- **Use**: Production benchmarks, final results
- **Recommendation**: Test on 100K first, then scale to 1M

### Cohere MS MARCO (1024 dimensions)
- **Size**: 53.2M passages
- **Dimension**: 1024 (divisible by 8, 16, 32, 64)
- **Use**: Large-scale retrieval benchmarks
- **Recommendation**: Use streaming, precompute ground truth

---

## Method 1: Product Quantization (PQ)

### Key Parameters
- **M** (subquantizers): Number of chunks to split vector into
  - Must divide dimension evenly
  - Lower M = higher accuracy, lower compression
  - Higher M = lower accuracy, higher compression
- **B** (bits): Bits per subquantizer index
  - Common: 8 bits (256 clusters)
  - Can use 4-8 bits

### Recommended Sweep for 1536-dim

**Conservative (focus on accuracy):**
```bash
vq-benchmark sweep \
    --method pq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --pq-subquantizers "8,12,16" \
    --pq-bits "8" \
    --with-recall --with-pairwise --with-rank
```
- M=8: 192 dims/chunk, high accuracy
- M=12: 128 dims/chunk, balanced
- M=16: 96 dims/chunk, good compression

**Aggressive (focus on compression):**
```bash
vq-benchmark sweep \
    --method pq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --pq-subquantizers "16,24,32" \
    --pq-bits "6,8" \
    --with-recall --with-pairwise --with-rank
```
- M=32, B=6: Very high compression (192 bytes → 24 bytes = 8x)
- M=16, B=8: Balanced (192 bytes → 16 bytes = 12x)

**Full exploration:**
```bash
vq-benchmark sweep \
    --method pq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --pq-subquantizers "8,12,16,24,32" \
    --pq-bits "6,8" \
    --with-recall --with-pairwise --with-rank
```
- 10 configurations (5 M values × 2 B values)
- Runtime: ~30-60 minutes on typical hardware

---

## Method 2: Optimized Product Quantization (OPQ)

### Key Parameters
- Same as PQ, but adds learned rotation matrix
- Training is slower than PQ
- Generally achieves better accuracy than PQ with same M, B

### Recommended Sweep for 1536-dim

**Standard sweep:**
```bash
vq-benchmark sweep \
    --method opq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --opq-quantizers "8,16,32" \
    --opq-bits "8" \
    --with-recall --with-pairwise --with-rank
```
- Fewer points than PQ (training is slower)
- Focus on B=8 (most common)

**Comparison with PQ:**
```bash
# Run both to compare
vq-benchmark sweep --method pq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --pq-subquantizers "8,16" --pq-bits "8"

vq-benchmark sweep --method opq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --opq-quantizers "8,16" --opq-bits "8"

# Generate comparison plot
vq-benchmark plot
```

---

## Method 3: Scalar Quantization (SQ)

### Key Parameters
- **bits**: Bits per dimension
  - 4 bits: 4x compression
  - 8 bits: 4x compression (most common)
  - 16 bits: 2x compression (high accuracy)

### Recommended Sweep for 1536-dim

**Standard sweep:**
```bash
vq-benchmark sweep \
    --method sq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --sq-bits "4,8" \
    --with-recall --with-pairwise --with-rank
```
- Fast to run (2 configurations)
- Good baseline for comparison

**Extended sweep:**
```bash
vq-benchmark sweep \
    --method sq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --sq-bits "2,4,8,16" \
    --with-recall --with-pairwise --with-rank
```
- Explores full range of compression ratios

---

## Method 4: Segmented CAQ (SAQ)

### Key Parameters
- **num_bits**: Default per-dimension bitwidth (if not using total_bits)
- **total_bits**: Global bit budget per vector
- **allowed_bits**: Discrete bitwidths allowed for segments
- **n_segments**: Number of segments (auto-determined if None)

### Recommended Sweep for 1536-dim

**Option A: Fixed per-dimension bits**
```bash
vq-benchmark sweep \
    --method saq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --saq-num-bits "4,6,8" \
    --saq-allowed-bits "0,2,4,6,8" \
    --with-recall --with-pairwise --with-rank
```
- Simple approach
- 3 configurations

**Option B: Total bit budgets (recommended)**
```bash
vq-benchmark sweep \
    --method saq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --saq-num-bits "" \
    --saq-total-bits "3072,4608,6144,7680" \
    --saq-allowed-bits "0,2,4,6,8" \
    --with-recall --with-pairwise --with-rank
```
- More flexible bit allocation
- 3072 bits = 2 bits/dim avg (384 bytes, 16x compression)
- 4608 bits = 3 bits/dim avg (576 bytes, ~10x compression)
- 6144 bits = 4 bits/dim avg (768 bytes, 8x compression)
- 7680 bits = 5 bits/dim avg (960 bytes, 6.4x compression)

**Fine-grained sweep:**
```bash
vq-benchmark sweep \
    --method saq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --saq-total-bits "2048,3072,4096,5120,6144,8192" \
    --saq-allowed-bits "0,2,4,6,8" \
    --saq-segments "8,16,32" \
    --with-recall --with-pairwise --with-rank
```
- Explores different segment counts
- 18 configurations (6 budgets × 3 segment counts)

---

## Method 5: RaBitQ

### Key Parameters
- **metric_type**: Distance metric (L2 or IP)
- RaBitQ is mostly automatic (few hyperparameters)

### Recommended Sweep for 1536-dim

**Standard sweep:**
```bash
vq-benchmark sweep \
    --method rabitq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --rabitq-metric-type "L2" \
    --with-recall --with-pairwise --with-rank
```
- Single configuration (if using L2 distance)
- Very fast to run

**With metric comparison:**
```bash
vq-benchmark sweep \
    --method rabitq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --rabitq-metric-type "L2,IP" \
    --with-recall --with-pairwise --with-rank
```
- Compares L2 vs Inner Product distance

---

## Complete Cross-Method Comparison

To compare all methods on DBpedia 100K:

```bash
#!/bin/bash
# Complete benchmark suite

# PQ baseline
vq-benchmark sweep --method pq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --pq-subquantizers "8,16,32" --pq-bits "8"

# OPQ (improved PQ)
vq-benchmark sweep --method opq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --opq-quantizers "8,16,32" --opq-bits "8"

# SQ baseline
vq-benchmark sweep --method sq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --sq-bits "4,8"

# SAQ (adaptive)
vq-benchmark sweep --method saq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --saq-total-bits "3072,6144,9216"

# RaBitQ (extreme compression)
vq-benchmark sweep --method rabitq \
    --dataset dummy --num-samples 100000 --dim 1536 \
    --rabitq-metric-type "L2"

# Generate comparison plots
vq-benchmark plot
```

**Total configurations:** ~13
**Estimated runtime:** 1-3 hours depending on hardware

---

## Using the Actual DBpedia 100K Dataset

The examples above use `--dataset dummy` with `--num-samples 100000 --dim 1536` to simulate the dataset. To use the **actual DBpedia 100K dataset**, you'll need to integrate it into the CLI. Currently, the CLI only supports `dummy`, `huggingface`, and `msmarco` datasets.

### Option 1: Load and Save, Then Reference

```bash
# 1. Load DBpedia 100K and save to .npz
python -m haag_vq.data.dbpedia_loader \
    --dim 1536 \
    --use-100k \
    --cache-dir ../datasets \
    --out data/dbpedia-100k.npz

# 2. Update CLI to support loading from .npz files
# (Would require code changes to add --vectors-path argument)
```

### Option 2: Use Python API Directly

For now, the easiest way to use the actual dataset is via Python:

```python
from haag_vq.data import load_dbpedia_openai_1536_100k
from haag_vq.methods import ProductQuantizer
from haag_vq.metrics import compute_distortion

# Load dataset
ds = load_dbpedia_openai_1536_100k(cache_dir="../datasets")

# Run benchmark
pq = ProductQuantizer(M=8, B=8)
pq.fit(ds.vectors)
codes = pq.compress(ds.vectors)
reconstructed = pq.decompress(codes)

# Evaluate
distortion = compute_distortion(ds.vectors, reconstructed)
ratio = pq.get_compression_ratio(ds.vectors)

print(f"Compression: {ratio:.1f}x")
print(f"Distortion: {distortion:.6f}")
```

---

## Tips for Effective Sweeps

### 1. Start Small, Scale Up
```bash
# Test with 10K first
--num-samples 10000

# Then 100K
--num-samples 100000

# Finally full dataset
```

### 2. Choose M Values Carefully (PQ/OPQ)
For dimension D, M must divide D evenly:
- **D=1024**: Use M ∈ {8, 16, 32, 64}
- **D=1536**: Use M ∈ {8, 12, 16, 24, 32, 48}
- **D=3072**: Use M ∈ {8, 12, 16, 24, 32, 48, 64}

### 3. Use Reasonable Bit Budgets (SAQ)
For dimension D:
- **Conservative**: 4-6 bits/dim (D × 4 to D × 6 total bits)
- **Aggressive**: 2-4 bits/dim (D × 2 to D × 4 total bits)
- **Extreme**: 1-2 bits/dim (D × 1 to D × 2 total bits)

### 4. Precompute Ground Truth for Large Datasets
```bash
# For datasets > 100K
vq-benchmark precompute-gt \
    --vectors-path data/my-vectors.npy \
    --output-path data/my-gt.npy \
    --num-queries 1000

# Then use in sweeps
vq-benchmark sweep ... --ground-truth-path data/my-gt.npy
```

### 5. Organize Results by Sweep
Each sweep gets a unique ID. Use it to filter plots:
```bash
vq-benchmark plot --sweep-id sweep_20250130_143022_abc123de
```

---

## Troubleshooting

### "Dimension must be divisible by M"
Choose M values that divide your dimension evenly.

### Out of Memory
- Reduce `--num-samples`
- Use `--streaming` for dataset loading
- Precompute ground truth separately

### Sweeps Taking Too Long
- Start with fewer parameter combinations
- Use smaller `--num-samples` for initial testing
- Focus on one method at a time

---

## Next Steps

After running sweeps:
1. **Visualize results**: `vq-benchmark plot`
2. **Analyze trade-offs**: Look at compression vs distortion curves
3. **Select best method**: Based on your accuracy/compression requirements
4. **Scale up**: Run on larger datasets or more configurations

See [USAGE.md](USAGE.md) for complete CLI reference.
