# Quick Start: Running Parameter Sweeps

This guide shows you how to quickly run comprehensive benchmarks on the DBpedia 100K dataset.

## Automated Sweep Script

The easiest way to run all methods:

```bash
./example_sweeps_dbpedia100k.sh
```

This will:
1. Auto-download the DBpedia 100K dataset (on first run)
2. Run parameter sweeps for all 5 methods (PQ, OPQ, SQ, SAQ, RaBitQ)
3. Save results to `./logs/benchmark_runs.db`

**Total runtime:** ~1-3 hours depending on hardware

---

## Manual Commands

### Individual Method Examples

**PQ (Product Quantization):**
```bash
vq-benchmark sweep \
    --method pq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --pq-subquantizers "8,16,32" \
    --pq-bits "8"
```

**OPQ (Optimized Product Quantization):**
```bash
vq-benchmark sweep \
    --method opq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --opq-quantizers "8,16,32" \
    --opq-bits "8"
```

**SQ (Scalar Quantization):**
```bash
vq-benchmark sweep \
    --method sq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --sq-bits "4,8"
```

**SAQ (Segmented CAQ):**
```bash
vq-benchmark sweep \
    --method saq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets \
    --saq-total-bits "3072,6144,9216"
```

**RaBitQ:**
```bash
vq-benchmark sweep \
    --method rabitq \
    --dataset dbpedia-100k \
    --cache-dir ../datasets
```

---

## Visualizing Results

After running sweeps, generate plots:

```bash
vq-benchmark plot
```

This creates:
- `plots/compression_distortion_tradeoff.png`
- `plots/pairwise_distortion.png`
- `plots/rank_distortion.png`
- `plots/recall_comparison.png`
- `plots/comparison_table.txt`

---

## Using Other Datasets

### Cohere MS MARCO (53M passages, 1024-dim)
```bash
vq-benchmark sweep \
    --method pq \
    --dataset cohere-msmarco \
    --dataset-limit 100000 \
    --cache-dir ../datasets \
    --pq-subquantizers "8,16,32" \
    --pq-bits "8"
```

### DBpedia 1M (1536-dim)
```bash
vq-benchmark sweep \
    --method pq \
    --dataset dbpedia-1536 \
    --dataset-limit 100000 \
    --cache-dir ../datasets \
    --pq-subquantizers "8,16,32" \
    --pq-bits "8"
```

### DBpedia 1M (3072-dim)
```bash
vq-benchmark sweep \
    --method pq \
    --dataset dbpedia-3072 \
    --dataset-limit 100000 \
    --cache-dir ../datasets \
    --pq-subquantizers "8,16,32" \
    --pq-bits "8"
```

---

## Parameter Guidelines

### For 1536-dim datasets (DBpedia):
- **PQ M values:** 8, 12, 16, 24, 32 (must divide 1536)
- **PQ B values:** 6, 8
- **SAQ bit budgets:** 3072 (2 bits/dim), 6144 (4 bits/dim), 9216 (6 bits/dim)

### For 1024-dim datasets (Cohere MS MARCO):
- **PQ M values:** 8, 16, 32, 64 (must divide 1024)
- **PQ B values:** 6, 8
- **SAQ bit budgets:** 2048 (2 bits/dim), 4096 (4 bits/dim), 6144 (6 bits/dim)

### For 3072-dim datasets (DBpedia):
- **PQ M values:** 8, 12, 16, 24, 32, 48 (must divide 3072)
- **PQ B values:** 6, 8
- **SAQ bit budgets:** 6144 (2 bits/dim), 12288 (4 bits/dim), 18432 (6 bits/dim)

---

## Next Steps

1. **Run the automated script:** `./example_sweeps_dbpedia100k.sh`
2. **Generate plots:** `vq-benchmark plot`
3. **Analyze results:** Look at compression vs. distortion trade-offs
4. **Scale up:** Try larger datasets or more parameter combinations

For detailed documentation, see:
- [SWEEP_EXAMPLES.md](documentation/SWEEP_EXAMPLES.md) - Comprehensive sweep guide
- [USAGE.md](documentation/USAGE.md) - CLI reference
- [METHODS.md](documentation/METHODS.md) - Method details
