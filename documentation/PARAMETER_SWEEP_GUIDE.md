# Parameter Sweep Guide

## What is a Parameter Sweep?

A parameter sweep systematically runs benchmarks across different hyperparameter configurations to explore the **compression-quality trade-off space**.

---

## Product Quantization (PQ) Parameters

PQ has two main hyperparameters that control the trade-off:

### 1. `num_chunks` - How Many Pieces to Split Vectors Into

```
Original 128-dim vector: [x1, x2, x3, ..., x128]

With num_chunks=4:
  Chunk 1: [x1...x32]   → Compress to 1 byte (cluster ID)
  Chunk 2: [x33...x64]  → Compress to 1 byte
  Chunk 3: [x65...x96]  → Compress to 1 byte
  Chunk 4: [x97...x128] → Compress to 1 byte

  Total: 4 bytes instead of 512 bytes (128 × 4 bytes for float32)
  Compression ratio: 512 / 4 = 128x
```

**Effect:**
- ✅ More chunks = **Higher compression** (fewer bytes)
- ❌ More chunks = **More distortion** (each chunk smaller, less information)

---

### 2. `num_clusters` - How Many Representative Vectors Per Chunk

For each chunk, we learn a "codebook" of representative sub-vectors:

```
With num_clusters=256:
  - Learn 256 centroids for each chunk
  - Store cluster ID (0-255) = 1 byte

With num_clusters=128:
  - Learn 128 centroids for each chunk
  - Store cluster ID (0-127) = still 1 byte (but less precise)
```

**Effect:**
- ✅ More clusters = **Better quality** (finer-grained quantization)
- ❌ More clusters = **Larger codebook** (but still 1 byte per chunk if ≤256)

---

## Demo Sweep Configuration

The demo tests 4 configurations exploring different trade-offs:

```python
configs = [
    {"num_chunks": 4, "num_clusters": 128},   # Config 1
    {"num_chunks": 8, "num_clusters": 128},   # Config 2
    {"num_chunks": 8, "num_clusters": 256},   # Config 3
    {"num_chunks": 16, "num_clusters": 256},  # Config 4
]
```
---

## How Compression Ratio is Calculated

```python
# Original size
original_bytes = num_dimensions × 4 bytes (float32)
                = 128 × 4 = 512 bytes

# Compressed size (PQ)
compressed_bytes = num_chunks × 1 byte
                 = 8 × 1 = 8 bytes

# Compression ratio
ratio = original_bytes / compressed_bytes
      = 512 / 8 = 64x
```

---

## What the Sweep Demonstrates

By testing multiple configurations, we can:

1. **See the trade-off**: Higher compression → Lower quality
2. **Find sweet spots**: Config 3 (8 chunks, 256 clusters) balances both
3. **Generate trade-off curves**: Plot compression ratio vs. distortion/recall
4. **Make informed decisions**: Choose best config for your use case

---

## Customizing the Sweep

### For More Aggressive Compression:
```python
configs = [
    {"num_chunks": 16, "num_clusters": 256},
    {"num_chunks": 32, "num_clusters": 256},
    {"num_chunks": 64, "num_clusters": 256},
]
```

### For Higher Quality:
```python
configs = [
    {"num_chunks": 4, "num_clusters": 256},
    {"num_chunks": 4, "num_clusters": 512},
    {"num_chunks": 8, "num_clusters": 512},
]
```

**Note:** num_clusters > 256 requires 2 bytes per chunk instead of 1

---

## Using the CLI for Sweeps

Instead of manually coding configurations, use the CLI:

```bash
# Sweep over chunks (keeping clusters fixed)
vq-benchmark sweep --method pq --pq-chunks "4,8,16,32" --pq-clusters "256"

# Sweep over both (Cartesian product: 3 × 2 = 6 configs)
vq-benchmark sweep --method pq --pq-chunks "4,8,16" --pq-clusters "128,256"

# Sweep scalar quantization with different bit depths
vq-benchmark sweep --method sq --sq-bits "4,8,16"

# On real data
vq-benchmark sweep --method pq --dataset huggingface --pq-chunks "8,16"
```

**Each sweep automatically generates a unique Sweep ID:**
```
🔖 Sweep ID: sweep_20251003_143052_a3b9f2c1
   Use this ID to filter plots: vq-benchmark plot --sweep-id sweep_20251003_143052_a3b9f2c1
```

---

## Interpreting Sweep Results

After running a sweep, check:

1. **Database**: `sqlite3 logs/benchmark_runs.db`
   ```sql
   -- View all runs
   SELECT config_json,
          json_extract(metrics_json, '$.compression_ratio') as compression,
          json_extract(metrics_json, '$.recall@10') as recall
   FROM runs;

   -- Filter by specific sweep
   SELECT config_json, metrics_json
   FROM runs
   WHERE sweep_id = 'sweep_20251003_143052_a3b9f2c1';
   ```

2. **Plots**: Run `vq-benchmark plot` to see:
   - Compression vs. Distortion curve
   - Compression vs. Recall curve
   - Which configs are Pareto-optimal

   **Plot Options:**
   ```bash
   # Plot all sweeps combined
   vq-benchmark plot

   # Plot only a specific sweep
   vq-benchmark plot --sweep-id sweep_20251003_143052_a3b9f2c1

   # Create separate plots for each method (better for mixed PQ/SQ sweeps)
   vq-benchmark plot --separate-methods

   # Combine filters
   vq-benchmark plot --sweep-id sweep_20251003_143052_a3b9f2c1 --separate-methods
   ```

   **Output:** Plots are saved in timestamped folders like `plots/20251003_143530/`

3. **Trade-off Analysis**:
   - Identify "elbow" in the curve (diminishing returns)
   - Choose config that meets your requirements
   - Example: "Need 50x compression with >0.8 recall" → pick matching config

---

## Visual Example

```
Compression-Recall Trade-off:

Recall@10
   1.0 |    Config 1 (4 chunks, high quality)
       |     ●
   0.9 |
       |        ● Config 3 (8 chunks, 256 clusters)
   0.8 |          ● Config 2 (8 chunks, 128 clusters)
       |
   0.7 |
       |                ● Config 4 (16 chunks, lower quality)
   0.6 |
       +-----|-----|-----|-----|-----
             32x   64x   128x  256x  Compression Ratio

Sweet spot: Config 3 offers good balance (64x compression, 0.85 recall)
```