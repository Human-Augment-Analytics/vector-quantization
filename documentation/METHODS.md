# Quantization Methods Guide

This guide provides an overview of all vector quantization methods implemented in the HAAG Vector Quantization project.

---

## Overview

The project implements **5 quantization methods** spanning different approaches to vector compression:

| Method | Type | Paper/Reference | Implementation |
|--------|------|----------------|----------------|
| **PQ** | Subspace Quantization | [Jégou et al., 2011](https://hal.inria.fr/inria-00514462v2/document) | `product_quantization.py` |
| **OPQ** | Optimized Subspace | [Ge et al., 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf) | `optimized_product_quantization.py` |
| **SQ** | Scalar Quantization | Standard | `scalar_quantization.py` |
| **SAQ** | Segmented CAQ | [Zhou et al., 2024](https://arxiv.org/abs/2410.06482) | `saq.py` |
| **RaBitQ** | Bit-level Quantization | [Gao & Long, 2024](https://dl.acm.org/doi/pdf/10.1145/3654970) | `rabit_quantization.py` |

---

## 1. Product Quantization (PQ)

**Paper:** [Product Quantization for Nearest Neighbor Search](https://hal.inria.fr/inria-00514462v2/document) (Jégou et al., TPAMI 2011)

### Description
Product Quantization decomposes vectors into **M subvectors** and quantizes each independently using **k-means clustering**. This creates a Cartesian product of codebooks.

### Key Idea
- Split D-dimensional vector into M subvectors of dimension D/M
- Learn k-means codebook (typically 256 clusters = 8 bits) for each subspace
- Encode vector as M indices (one per subspace)

### Parameters
- **M** (num_subquantizers): Number of subquantizers/chunks
  - Common values: 4, 8, 16, 32
  - Must divide the vector dimension evenly
- **B** (bits): Bits per subvector index
  - Common: 8 bits = 256 clusters per subspace
  - Can use 4, 6, 8 bits

### Compression Ratio
For D=1024, M=8, B=8:
- Original: 1024 × 4 bytes = 4096 bytes
- Compressed: 8 × 1 byte = 8 bytes
- **Ratio: 512:1**

### CLI Usage
```bash
# Single run
vq-benchmark run --method pq --num-chunks 8 --num-clusters 256

# Parameter sweep
vq-benchmark sweep --method pq \
  --pq-subquantizers "4,8,16" \
  --pq-bits "8"
```

### Python API
```python
from haag_vq.methods.product_quantization import ProductQuantizer

pq = ProductQuantizer(num_chunks=8, num_clusters=256)
pq.fit(train_vectors)
codes = pq.compress(vectors)
reconstructed = pq.decompress(codes)
```

### Pros/Cons
✅ Simple and effective baseline
✅ Fast compression/decompression
✅ Widely used in industry (FAISS, Pinecone, etc.)
❌ Assumes subspaces are independent (not optimal)
❌ Sensitive to vector dimension ordering

---

## 2. Optimized Product Quantization (OPQ)

**Paper:** [Optimized Product Quantization](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf) (Ge et al., TPAMI 2013)

### Description
OPQ improves upon PQ by learning an **optimal rotation matrix** before applying product quantization. This decorrelates dimensions and balances variance across subspaces.

### Key Idea
- Learn rotation matrix R that minimizes quantization error
- Apply R before PQ encoding: `codes = PQ(R × vector)`
- Apply R^T after PQ decoding: `vector ≈ R^T × PQ_decode(codes)`

### Parameters
- **M** (num_quantizers): Number of subquantizers
  - Same as PQ
- **B** (bits): Bits per subvector
  - Typically 8 bits

### Compression Ratio
Same as PQ (rotation matrix is shared across all vectors, not stored per-vector)

### CLI Usage
```bash
# Single run
vq-benchmark run --method opq

# Parameter sweep
vq-benchmark sweep --method opq \
  --opq-quantizers "8,16,32" \
  --opq-bits "8"
```

### Python API
```python
from haag_vq.methods.optimized_product_quantization import OptimizedProductQuantizer

opq = OptimizedProductQuantizer(M=8, B=8)
opq.fit(train_vectors)
codes = opq.compress(vectors)
reconstructed = opq.decompress(codes)
```

### Pros/Cons
✅ Better accuracy than PQ (learns optimal rotation)
✅ Same compression ratio as PQ
✅ Industry standard (FAISS OPQ)
❌ More complex training (iterative optimization)
❌ Slightly slower encoding/decoding than PQ

---

## 3. Scalar Quantization (SQ)

**Type:** Standard uniform quantization

### Description
Scalar Quantization uniformly quantizes each dimension independently to a fixed number of bits (typically 8 bits = 1 byte per dimension).

### Key Idea
- Map each float32 value to discrete levels (e.g., 256 levels for 8 bits)
- Learn min/max or mean/std per dimension
- Uniform binning within range

### Parameters
- **bits**: Bits per dimension
  - Common: 8 bits (uint8)
  - Can use 4, 8, 16 bits

### Compression Ratio
For D=1024, 8 bits:
- Original: 1024 × 4 bytes = 4096 bytes
- Compressed: 1024 × 1 byte = 1024 bytes
- **Ratio: 4:1**

### CLI Usage
```bash
# Single run
vq-benchmark run --method sq

# Parameter sweep
vq-benchmark sweep --method sq --sq-bits "4,8,16"
```

### Python API
```python
from haag_vq.methods.scalar_quantization import ScalarQuantizer

sq = ScalarQuantizer()
sq.fit(train_vectors)
codes = sq.compress(vectors)
reconstructed = sq.decompress(codes)
```

### Pros/Cons
✅ Simple and fast
✅ Per-dimension quantization preserves structure
✅ Easy to implement and debug
❌ Lower compression ratio than PQ/OPQ
❌ Doesn't exploit correlations between dimensions

---

## 4. Segmented CAQ (SAQ)

**Paper:** [Learned Compression for Compressed Learning](https://arxiv.org/abs/2410.06482) (Zhou et al., 2024)

### Description
SAQ applies **PCA rotation** followed by **segment-wise bit allocation** with uniform scalar quantization. It intelligently allocates more bits to high-variance dimensions and fewer bits (or drops) low-variance dimensions.

### Key Idea
1. Apply PCA to decorrelate and order dimensions by variance
2. Segment dimensions into groups
3. Allocate bits per segment to minimize distortion under a global bit budget
4. Apply uniform scalar quantization within each segment

### Parameters
- **num_bits**: Default per-dimension bitwidth (if not using total_bits)
  - Range: 0-8, where 0 means dimension is dropped
- **total_bits**: Global per-vector bit budget
  - If set, overrides num_bits and uses greedy allocation
- **allowed_bits**: Discrete bitwidths permitted (e.g., [0,2,4,6,8])
- **n_segments**: Number of segments (auto-determined if None)

### Compression Ratio
Variable based on bit allocation. Example:
- D=1024, total_bits=4096 (4 bits/dim average)
- Original: 4096 bytes
- Compressed: 512 bytes
- **Ratio: 8:1**

### CLI Usage
```bash
# Fixed bits per dimension
vq-benchmark sweep --method saq --saq-num-bits "4,6,8"

# Global bit budget
vq-benchmark sweep --method saq \
  --saq-total-bits "2048,4096,8192" \
  --saq-allowed-bits "0,2,4,6,8"
```

### Python API
```python
from haag_vq.methods.saq import SAQ

# Fixed bits
saq = SAQ(num_bits=4)

# Or bit budget
saq = SAQ(total_bits=4096, allowed_bits=[0,2,4,6,8])

saq.fit(train_vectors)
codes = saq.compress(vectors)
reconstructed = saq.decompress(codes)
```

### Pros/Cons
✅ Adaptive bit allocation based on importance
✅ Can drop unimportant dimensions (0 bits)
✅ Theoretical grounding in rate-distortion theory
❌ Requires careful tuning of segments and bit budget
❌ More complex than PQ/SQ

---

## 5. RaBitQ (RaBit Quantization)

**Paper:** [RabitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound](https://dl.acm.org/doi/pdf/10.1145/3654970) (Gao & Long, SIGMOD 2024)

### Description
RaBitQ is a **bit-level quantization** method with theoretical error bounds. It uses FAISS's implementation for efficient compression.

### Key Idea
- Quantize vectors to very low bit representations (often < 1 bit/dimension)
- Provides theoretical guarantees on approximation error
- Optimized for approximate nearest neighbor search

### Parameters
- **metric_type**: Distance metric (L2 or inner product)
  - Options: "L2", "IP" (inner product)

### Compression Ratio
Varies based on FAISS's internal algorithm. Typically very high (> 100:1)

### CLI Usage
```bash
# Single run
vq-benchmark run --method rabitq

# Parameter sweep with different metrics
vq-benchmark sweep --method rabitq \
  --rabitq-metric-type "L2,IP"
```

### Python API
```python
from haag_vq.methods.rabit_quantization import RaBitQuantizer
from haag_vq.utils.faiss_utils import MetricType

rabitq = RaBitQuantizer(metric_type=MetricType.L2)
rabitq.fit(train_vectors)
codes = rabitq.compress(vectors)
reconstructed = rabitq.decompress(codes)
```

### Pros/Cons
✅ Extremely high compression ratios
✅ Theoretical error bounds
✅ Optimized FAISS implementation
❌ Newest method (less battle-tested)
❌ Limited parameter control (mostly automatic)

---

## Method Comparison

### Compression Ratio vs Accuracy Trade-off

| Method | Typical Compression | Accuracy | Complexity | Use Case |
|--------|---------------------|----------|------------|----------|
| **SQ** | 4:1 | High | Low | When accuracy is critical |
| **PQ** | 32-512:1 | Medium | Low | General-purpose baseline |
| **OPQ** | 32-512:1 | Medium-High | Medium | Better than PQ, worth the cost |
| **SAQ** | 8-32:1 | Medium-High | Medium | Adaptive importance-based |
| **RaBitQ** | 100:1+ | Medium | Low* | Extreme compression needs |

*Low complexity for users (FAISS handles internals)

### When to Use Each Method

**Use PQ when:**
- You need a simple, proven baseline
- Fast encoding/decoding is critical
- You're okay with moderate accuracy loss

**Use OPQ when:**
- You want better accuracy than PQ
- You can afford slightly slower training
- Your data has correlated dimensions

**Use SQ when:**
- Accuracy is more important than compression
- You want simple, interpretable quantization
- You have storage/bandwidth to spare

**Use SAQ when:**
- You want adaptive bit allocation
- You can identify unimportant dimensions to drop
- You need fine-grained compression control

**Use RaBitQ when:**
- You need extreme compression
- You're using FAISS already
- You prioritize approximate NN search speed

---

## Parameter Tuning Guidelines

### PQ/OPQ: Choosing M (subquantizers)
- **More M** → Better compression, but lower accuracy
  - M=4: Conservative, high accuracy
  - M=8: Balanced (common default)
  - M=16-32: Aggressive compression

### PQ/OPQ: Choosing B (bits)
- **B=8** (256 clusters): Standard, good trade-off
- **B=6** (64 clusters): Higher compression, lower accuracy
- **B=4** (16 clusters): Very aggressive

### SAQ: Bit Budget Strategy
- Start with total_bits = D × 4 (4 bits/dim average)
- Use allowed_bits=[0,2,4,6,8] to allow dropping dimensions
- Increase n_segments for finer-grained control

### General Tips
1. **Always benchmark on your specific dataset** - results vary!
2. **Start with PQ as baseline** - easy to understand and interpret
3. **Use parameter sweeps** to explore trade-offs systematically
4. **Precompute ground truth** for large datasets before sweeps

---

## Running Method Comparisons

### Compare All Methods
```bash
# Run sweeps for each method
vq-benchmark sweep --method pq --dataset dbpedia-1536
vq-benchmark sweep --method opq --dataset dbpedia-1536
vq-benchmark sweep --method sq --dataset dbpedia-1536
vq-benchmark sweep --method saq --dataset dbpedia-1536
vq-benchmark sweep --method rabitq --dataset dbpedia-1536

# Generate comparison plots
vq-benchmark plot
```

### Multi-Method Benchmark Script
```python
from haag_vq.data import load_dbpedia_openai_1536
from haag_vq.methods import *

# Load data
ds = load_dbpedia_openai_1536(limit=10000)

# Test each method
methods = {
    "PQ": ProductQuantizer(num_chunks=8, num_clusters=256),
    "OPQ": OptimizedProductQuantizer(M=8, B=8),
    "SQ": ScalarQuantizer(),
    "SAQ": SAQ(num_bits=4),
}

for name, method in methods.items():
    method.fit(ds.vectors)
    codes = method.compress(ds.vectors)
    reconstructed = method.decompress(codes)

    # Compute metrics
    distortion = compute_distortion(ds.vectors, reconstructed)
    ratio = method.get_compression_ratio(ds.vectors)

    print(f"{name}: Compression={ratio:.1f}x, Distortion={distortion:.6f}")
```

---

## Adding New Methods

See [ADDING_NEW_METHODS.md](ADDING_NEW_METHODS.md) for a detailed guide on implementing new quantization methods.

---

## References

1. **PQ:** Jégou et al. "Product Quantization for Nearest Neighbor Search." TPAMI 2011.
2. **OPQ:** Ge et al. "Optimized Product Quantization." TPAMI 2013.
3. **SAQ:** Zhou et al. "Learned Compression for Compressed Learning." 2024.
4. **RaBitQ:** Gao & Long. "RabitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound." SIGMOD 2024.

---

## Questions?

Post in the `#vector-quantization` Slack channel or see:
- [USAGE.md](USAGE.md) - CLI reference
- [ADDING_NEW_METHODS.md](ADDING_NEW_METHODS.md) - Developer guide
- [METRICS_GUIDE.md](METRICS_GUIDE.md) - Understanding metrics
