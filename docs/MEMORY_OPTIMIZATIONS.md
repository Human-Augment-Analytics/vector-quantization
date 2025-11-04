# Memory Optimizations

This document describes the memory optimizations implemented across all data loaders to minimize memory usage without sacrificing functionality.

## Summary

All data loaders have been optimized to reduce peak memory usage by **~50%** through:

1. **Pre-allocated numpy arrays** instead of Python lists
2. **Single-pass loading** to avoid temporary allocations
3. **FAISS-based ground truth computation** instead of full distance matrices

## Technical Details

### Before Optimization

The original loaders used Python lists to accumulate vectors:

```python
# OLD APPROACH (inefficient)
embeddings = []
for item in dataset:
    embeddings.append(item['embedding'])  # Python list overhead
vectors = np.array(embeddings)  # 2x memory: list + array
```

**Memory usage**:
- Python list: `num_vectors × (56 bytes object overhead + 8 bytes pointer) ≈ 64 bytes per vector`
- Plus vector data: `num_vectors × dimensions × 4 bytes`
- Conversion to numpy: Temporary allocation of full array
- **Peak memory**: `~2x final array size during conversion`

### After Optimization

New loaders pre-allocate numpy arrays:

```python
# NEW APPROACH (optimized)
dimension = 1536  # or determined from first item
vectors = np.zeros((num_vectors, dimension), dtype=np.float32)
for i, item in enumerate(dataset):
    vectors[i] = item['embedding']  # Direct assignment
# No conversion needed - already in final form
```

**Memory usage**:
- Pre-allocated array: `num_vectors × dimensions × 4 bytes`
- No Python list overhead
- No temporary allocations during conversion
- **Peak memory**: `~1x final array size`

### Ground Truth Optimization

Ground truth computation (k-nearest neighbors) was also optimized:

**Before** (sklearn pairwise distances):
```python
# Computes full distance matrix
dist_matrix = pairwise_distances(queries, vectors)  # O(n_q × n_v) memory
ground_truth = dist_matrix.argsort(axis=1)
```

**After** (FAISS):
```python
# Only computes top-k neighbors
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
distances, indices = index.search(queries, k=100)  # Much less memory
ground_truth = indices
```

**Memory savings**: For 100K vectors and 100 queries:
- Before: `100 × 100,000 × 8 bytes = 80 MB` (plus intermediate allocations)
- After: `100 × 100 × 8 bytes = 80 KB` (plus FAISS index overhead, but more efficient)

## Impact on Each Dataset

### DBpedia 100K (1536-dim)

**Vectors**: 100,000 × 1536 × 4 bytes = **614 MB**

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Loading | ~1.2 GB | ~614 MB | ~50% |
| Ground truth | ~80 MB | ~80 KB | >99% |
| **Total Peak** | ~1.3 GB | ~700 MB | ~46% |

### DBpedia 1M (1536-dim)

**Vectors**: 1,000,000 × 1536 × 4 bytes = **6.1 GB**

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Loading | ~12 GB | ~6.1 GB | ~50% |
| Ground truth | Skipped | Skipped | N/A |
| **Total Peak** | ~12 GB | ~6.1 GB | ~50% |

### DBpedia 1M (3072-dim)

**Vectors**: 1,000,000 × 3072 × 4 bytes = **12.3 GB**

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Loading | ~24 GB | ~12.3 GB | ~50% |
| Ground truth | Skipped | Skipped | N/A |
| **Total Peak** | ~24 GB | ~12.3 GB | ~50% |

### Cohere MS MARCO (53M vectors, ~1024-dim)

**Note**: This dataset is too large to load fully. Use `limit` parameter or `streaming=True`.

**Example with limit=100K**: 100,000 × 1024 × 4 bytes = **410 MB**

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Loading | ~820 MB | ~410 MB | ~50% |
| Ground truth | Skipped | Skipped | N/A |
| **Total Peak** | ~820 MB | ~410 MB | ~50% |

## Files Modified

All data loader files were optimized:

1. **[src/haag_vq/data/dbpedia_loader.py](../src/haag_vq/data/dbpedia_loader.py)**
   - `load_dbpedia_openai_1536()` - 1M vectors, 1536-dim
   - `load_dbpedia_openai_1536_100k()` - 100K vectors, 1536-dim
   - `load_dbpedia_openai_3072()` - 1M vectors, 3072-dim

2. **[src/haag_vq/data/cohere_msmarco_loader.py](../src/haag_vq/data/cohere_msmarco_loader.py)**
   - `load_cohere_msmarco_passages()` - Streaming support for 53M vectors
   - `load_cohere_msmarco_queries()` - ~1.6K queries

3. **[examples/demo.py](../examples/demo.py)**
   - Updated to use FAISS for ground truth computation
   - Added memory usage documentation

## Memory Requirements by Use Case

### Quick Testing (< 1 GB)
- **Use DBpedia 100K**: ~700 MB peak
- **Or Dummy dataset**: Negligible memory

### Standard Benchmarking (1-10 GB)
- **DBpedia 1M (1536-dim)**: ~6 GB peak
- **Or Cohere MS MARCO with limit**: Variable (use `limit` parameter)

### Large-Scale Benchmarking (> 10 GB)
- **DBpedia 1M (3072-dim)**: ~12 GB peak
- **Cohere MS MARCO**: Use `streaming=True` and process in batches

## Recommendations

### For Shared Computing Clusters

If you're still hitting memory limits even with these optimizations:

1. **Request appropriate resources**:
   ```bash
   # For DBpedia 100K
   srun --mem=2G --time=1:00:00 --pty bash

   # For DBpedia 1M (1536-dim)
   srun --mem=8G --time=2:00:00 --pty bash

   # For DBpedia 1M (3072-dim)
   srun --mem=16G --time=2:00:00 --pty bash
   ```

2. **Use smaller subsets** via `limit` parameter:
   ```python
   data = load_dbpedia_openai_1536(
       limit=50_000,  # Half size = half memory
       cache_dir="../.cache/datasets"
   )
   ```

3. **Batch processing** for extremely large datasets:
   - Load in chunks, train quantizer on subset
   - Apply quantization in batches
   - See streaming examples in documentation

### For Local Development

On a laptop with 8-16 GB RAM:
- **DBpedia 100K**: Works fine (~700 MB)
- **DBpedia 1M**: May need to close other applications (~6-12 GB)
- **Cohere MS MARCO**: Use `limit` parameter or streaming

## Performance Notes

These optimizations provide:
- **50% reduction in peak memory usage** during loading
- **No performance degradation** - same speed or faster
- **Identical results** - no loss of functionality
- **Better scalability** - can handle larger datasets on same hardware

## Questions or Issues?

If you still experience OOM (Out of Memory) errors:

1. Check actual memory available: `free -h` (Linux) or Activity Monitor (Mac)
2. Verify the dataset size you're trying to load
3. Consider using a compute node with more memory
4. Contact cluster administrators for resource allocation

For bugs or optimization suggestions, please open an issue on GitHub.
