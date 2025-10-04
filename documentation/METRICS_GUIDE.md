# Vector Quantization Metrics Guide

## Overview

This document explains the metrics used in the HAAG Vector Quantization benchmarking framework. Understanding these metrics is essential for evaluating compression quality and making informed trade-offs.

---

## 1. Reconstruction Distortion (MSE)

**What it measures:** How much the compressed vectors differ from the original vectors after decompression.

**Formula:**
```
distortion = mean(||original - decompress(compress(original))||²)
```

**Interpretation:**
- Lower is better
- Measures average squared error per vector
- Good for understanding information loss from compression

**When to use:**
- Comparing compression quality at similar compression ratios
- Applications where exact reconstruction matters

**Location:** `src/haag_vq/metrics/distortion.py`

---

## 2. Pairwise Distance Distortion

**What it measures:** How well the compression preserves distances between pairs of vectors.

**Formula:**
```
For each pair (v1, v2):
    distortion = |distance(compressed(v1), compressed(v2)) / distance(v1, v2) - 1|
```

**Interpretation:**
- 0 = perfect distance preservation
- 0.1 = distances preserved within 10% error
- 1.0 = 100% relative error (very bad)

**Why it matters:**
1. **Asymmetric Distance Computation:** Many vector databases (e.g., FAISS with PQ) compute distances directly in compressed space without decompression
2. **Ranking Quality:** Preserving relative distances is crucial for nearest neighbor search
3. **Different than Reconstruction Error:** Low MSE doesn't guarantee good distance preservation

**Example Use Case:**
If you're building a similarity search engine, you care more about whether `distance(a, b) < distance(a, c)` is preserved than whether you can perfectly reconstruct `a`.

**Location:** `src/haag_vq/metrics/pairwise_distortion.py`

---

## 3. Rank Distortion

**What it measures:** How much the compression changes the ranking of nearest neighbors.

**Formula:**
```
For a query q and k nearest neighbors:
    rank_distortion@k = hamming_distance(
        top_k_neighbors(compressed_space, q),
        top_k_neighbors(original_space, q)
    ) / k
```

**Interpretation:**
- 0 = perfect ranking preservation (all top-k are the same)
- 0.3 = 30% of top-k neighbors are wrong
- 1.0 = completely different top-k sets

**Relationship to Recall:**
```
rank_distortion@k ≈ 1 - recall@k
```

**Why it matters:**
- Directly measures nearest neighbor search quality
- More interpretable than recall for some use cases
- Critical for approximate nearest neighbor (ANN) systems

**Example:**
If `rank_distortion@10 = 0.2`, it means 2 out of 10 top results are wrong when using compressed representations.

**Location:** `src/haag_vq/metrics/rank_distortion.py`

---

## 4. Recall@k

**What it measures:** What fraction of true top-k neighbors were retrieved using compressed representations.

**Formula:**
```
For a query q:
    recall@k = |true_top_k ∩ retrieved_top_k| / k
```

**Interpretation:**
- 1.0 = all true neighbors found
- 0.5 = only half of true neighbors found
- 0 = none of the true neighbors found

**Location:** `src/haag_vq/metrics/recall.py`

---

## 5. Compression Ratio

**What it measures:** How much smaller the compressed representation is.

**Formula:**
```
compression_ratio = original_size_bytes / compressed_size_bytes
```

**Example:**
- Float32 vector of dimension 128: 128 × 4 bytes = 512 bytes
- PQ with 8 chunks, 256 clusters: 8 × 1 byte = 8 bytes
- Compression ratio: 512 / 8 = 64×

**Location:** Implemented in each quantizer (`get_compression_ratio()` method)

---

## Choosing the Right Metrics

### For Similarity Search Applications
**Primary metrics:**
1. Pairwise Distance Distortion (most important for ranking)
2. Rank Distortion or Recall@k
3. Compression Ratio (for storage/cost trade-offs)

**Secondary:**
- Reconstruction Distortion (less important)

### For Exact Reconstruction Applications
**Primary metrics:**
1. Reconstruction Distortion
2. Compression Ratio

**Secondary:**
- Pairwise metrics (less important)

### For Research/Comparison
**Use all metrics** to understand:
- Trade-offs between compression and quality
- How different methods preserve different properties
- Where each method excels

---

## Example Trade-off Analysis

```
Method A (High Compression):
- Compression: 128×
- Reconstruction MSE: 50.0
- Pairwise Distortion: 0.15
- Rank Distortion@10: 0.25
→ Good for large-scale search where some error is acceptable

Method B (High Quality):
- Compression: 4×
- Reconstruction MSE: 0.01
- Pairwise Distortion: 0.02
- Rank Distortion@10: 0.05
→ Good for applications requiring high accuracy

Method C (Balanced):
- Compression: 16×
- Reconstruction MSE: 5.0
- Pairwise Distortion: 0.08
- Rank Distortion@10: 0.12
→ Sweet spot for most applications
```

---

## Running Benchmarks with All Metrics

```bash
# Single run with all metrics
vq-benchmark run --method pq --with-recall

# Parameter sweep with all metrics
vq-benchmark sweep --method pq --chunks "8,16,32" --clusters "128,256,512"

# Visualize trade-offs
vq-benchmark plot
```

---

## References

1. **Product Quantization:** Jégou et al. (2011) - "Product Quantization for Nearest Neighbor Search"
2. **FAISS Documentation:** https://github.com/facebookresearch/faiss/wiki
3. **Vector Database Survey:** Qdrant, Weaviate, Pinecone technical blogs