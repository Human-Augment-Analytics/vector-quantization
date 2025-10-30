# Dataset Guide

This guide covers how to load and use datasets in the HAAG Vector Quantization project.

## Overview

The project supports both **pre-embedded** datasets (vectors already computed) and **text datasets** that need embedding. Pre-embedded datasets are recommended for large-scale benchmarking as they avoid expensive embedding computation.

---

## Available Datasets

### 1. Dummy Dataset (Built-in)
Simple random vectors for testing.

```python
from haag_vq.data import load_dummy_dataset

ds = load_dummy_dataset(num_samples=10000, dim=1024, seed=42)
```

**CLI Usage:**
```bash
vq-benchmark run --dataset dummy --num-samples 10000
```

---

### 2. Cohere MS MARCO v2.1 (Pre-embedded)

**Source:** [Cohere/msmarco-v2.1-embed-english-v3](https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3)

**Details:**
- **Size:** ~53.2M passages, ~1.6K queries
- **Embedding model:** Cohere Embed English v3
- **Dimensions:** ~1024
- **Use case:** Large-scale document retrieval benchmarking

**Python Usage:**
```python
from haag_vq.data import load_cohere_msmarco_passages, load_cohere_msmarco_queries

# Load passages (with streaming for large dataset)
ds = load_cohere_msmarco_passages(
    limit=100_000,              # Load 100k passages
    num_queries=100,            # Use 100 query vectors
    cache_dir="../datasets",    # Cache location
    streaming=True,             # Stream to avoid memory issues
)

# Load queries separately
queries = load_cohere_msmarco_queries(
    cache_dir="../datasets"
)
```

**Standalone Script:**
```bash
python -m haag_vq.data.cohere_msmarco_loader \
    --limit 100000 \
    --cache-dir ../datasets \
    --out data/cohere-msmarco-100k.npz
```

---

### 3. DBpedia OpenAI 1536-dim 100K (Pre-embedded)

**Source:** [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K)

**Details:**
- **Size:** 100K DBpedia entities (fast for testing)
- **Embedding model:** OpenAI text-embedding-3-large
- **Dimensions:** 1536
- **Use case:** Quick experiments before scaling to 1M dataset

**Python Usage:**
```python
from haag_vq.data import load_dbpedia_openai_1536_100k

# Load full 100K dataset
ds = load_dbpedia_openai_1536_100k(
    cache_dir="../datasets",
)

# Or use convenience function
from haag_vq.data import load_dbpedia_openai
ds = load_dbpedia_openai(
    embedding_dim=1536,
    use_100k_subset=True,
    cache_dir="../datasets",
)
```

**Standalone Script:**
```bash
python -m haag_vq.data.dbpedia_loader \
    --dim 1536 \
    --use-100k \
    --cache-dir ../datasets \
    --out data/dbpedia-1536-100k.npz
```

---

### 4. DBpedia OpenAI 1536-dim 1M (Pre-embedded)

**Source:** [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M)

**Details:**
- **Size:** 1M DBpedia entities
- **Embedding model:** OpenAI text-embedding-3-large
- **Dimensions:** 1536
- **Use case:** Large-scale knowledge graph entity embeddings

**Python Usage:**
```python
from haag_vq.data import load_dbpedia_openai_1536

ds = load_dbpedia_openai_1536(
    limit=100_000,
    num_queries=100,
    cache_dir="../datasets",
)
```

**Standalone Script:**
```bash
python -m haag_vq.data.dbpedia_loader \
    --dim 1536 \
    --limit 100000 \
    --cache-dir ../datasets \
    --out data/dbpedia-1536-100k.npz
```

---

### 5. DBpedia OpenAI 3072-dim (Pre-embedded)

**Source:** [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M)

**Details:**
- **Size:** 1M DBpedia entities
- **Embedding model:** OpenAI text-embedding-3-large
- **Dimensions:** 3072
- **Use case:** High-dimensional entity embeddings

**Python Usage:**
```python
from haag_vq.data import load_dbpedia_openai_3072

ds = load_dbpedia_openai_3072(
    limit=100_000,
    num_queries=100,
    cache_dir="../datasets",
)

# Or use convenience function
from haag_vq.data import load_dbpedia_openai

ds = load_dbpedia_openai(
    embedding_dim=3072,  # or 1536
    limit=100_000,
    cache_dir="../datasets",
)
```

**Standalone Script:**
```bash
python -m haag_vq.data.dbpedia_loader \
    --dim 3072 \
    --limit 100000 \
    --cache-dir ../datasets \
    --out data/dbpedia-3072-100k.npz
```

---

### 6. MS MARCO (Text-based, requires embedding)

**Details:**
- Load from local TSV or Hugging Face
- Requires embedding computation (time/compute intensive)

**Python Usage:**
```python
from haag_vq.data import load_msmarco_passages_from_hf

ds = load_msmarco_passages_from_hf(
    hf_dataset="ms_marco",
    limit=100_000,
    model_name="all-MiniLM-L6-v2",
)
```

**Standalone Script:**
```bash
python -m haag_vq.data.msmarco_loader \
    --hf \
    --limit 100000 \
    --model all-MiniLM-L6-v2 \
    --out data/msmarco-mini.npz
```

---

### 7. Hugging Face Text Datasets (Generic loader)

**Details:**
- Load any text dataset from Hugging Face
- Computes embeddings using Sentence-BERT

**Python Usage:**
```python
from haag_vq.data import load_huggingface_dataset

ds = load_huggingface_dataset(
    dataset_name="stsb_multi_mt",
    config_name="en",
    model_name="all-MiniLM-L6-v2",
    split="train"
)
```

---

## Dataset Directory Structure

For large pre-embedded datasets, we recommend storing them outside the repo to save space:

```
/path/to/parent/
├── datasets/                    # Hugging Face cache (auto-downloaded)
│   └── ...                      # Large cached datasets here
├── results/                     # Benchmark results
│   └── ...
└── vector-quantization/         # This repo
    ├── data/                    # Small processed datasets only
    │   ├── cohere-msmarco-100k.npz
    │   └── dbpedia-1536-100k.npz
    └── ...
```

**Set cache directory when loading:**
```python
ds = load_cohere_msmarco_passages(
    cache_dir="/path/to/parent/datasets"  # or "../datasets" if relative
)
```

---

## Common Parameters

All dataset loaders support these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `Optional[int]` | 100,000 | Max vectors to load. Use `None` for all. |
| `num_queries` | `int` | 100 | Number of query vectors (from start) |
| `cache_dir` | `Optional[str]` | `None` | Hugging Face cache directory |
| `streaming` | `bool` | `False`/`True`* | Stream dataset to avoid loading all into memory |

*Default depends on dataset size

---

## Working with Large Datasets

### Memory Considerations

| Dataset Size | Recommended Approach | Memory Required |
|--------------|---------------------|-----------------|
| < 100K vectors | Load directly | < 1GB |
| 100K - 1M vectors | Use `limit` parameter | 1-10GB |
| > 1M vectors | Use `streaming=True` + precompute ground truth | Variable |

### Example: Loading 1M vectors

```python
# Option 1: Load subset
ds = load_cohere_msmarco_passages(
    limit=100_000,  # Only load 100k
    cache_dir="../datasets",
)

# Option 2: Stream full dataset
ds = load_cohere_msmarco_passages(
    limit=None,  # Load all
    streaming=True,
    cache_dir="../datasets",
)
```

### Precomputing Ground Truth

For large datasets, ground truth computation requires too much memory. Precompute once:

```bash
# 1. Save vectors to .npy file
python -m haag_vq.data.cohere_msmarco_loader \
    --limit 1000000 \
    --cache-dir ../datasets \
    --out data/cohere-msmarco-1M.npz

# 2. Extract vectors array
python -c "import numpy as np; d=np.load('data/cohere-msmarco-1M.npz'); np.save('data/cohere-vectors.npy', d['vectors'])"

# 3. Precompute ground truth
vq-benchmark precompute-gt \
    --vectors-path data/cohere-vectors.npy \
    --output-path data/cohere-gt.npy \
    --num-queries 1000 \
    --k 100

# 4. Use in benchmarks
vq-benchmark run \
    --dataset cohere-msmarco \
    --ground-truth-path data/cohere-gt.npy \
    --with-recall
```

---

## Dataset Format

All loaders return a `Dataset` object with:

```python
class Dataset:
    vectors: np.ndarray        # Main vectors (N × D)
    queries: np.ndarray        # Query vectors (num_queries × D)
    ground_truth: np.ndarray   # Ground truth k-NN (num_queries × k)
    distance_metric: Callable  # Distance function
```

---

## Saving/Loading Processed Datasets

Save processed datasets for reuse:

```python
import numpy as np
from haag_vq.data import load_cohere_msmarco_passages

# Load and save
ds = load_cohere_msmarco_passages(limit=100_000, cache_dir="../datasets")
np.savez_compressed(
    "data/my-dataset.npz",
    vectors=ds.vectors,
    queries=ds.queries,
)

# Load later
data = np.load("data/my-dataset.npz")
vectors = data['vectors']
queries = data['queries']
```

---

## Downloading Datasets Manually

Datasets are auto-downloaded via Hugging Face, but you can pre-download:

```bash
# Set cache directory
export HF_HOME=../datasets

# Download (will cache in $HF_HOME)
python -c "
from datasets import load_dataset
load_dataset('Cohere/msmarco-v2.1-embed-english-v3', 'passages', cache_dir='../datasets')
"
```

---

## Adding New Datasets

See the existing loaders as templates:
- [cohere_msmarco_loader.py](../src/haag_vq/data/cohere_msmarco_loader.py)
- [dbpedia_loader.py](../src/haag_vq/data/dbpedia_loader.py)
- [msmarco_loader.py](../src/haag_vq/data/msmarco_loader.py)

Basic template:

```python
def load_my_dataset(limit=100_000, cache_dir=None):
    ds = load_dataset("org/dataset-name", cache_dir=cache_dir)

    embeddings = [item['embedding_column'] for item in ds]
    vectors = np.array(embeddings, dtype=np.float32)

    return Dataset(
        vectors=vectors,
        queries=vectors[:100],
        skip_ground_truth=True,
    )
```

---

## Troubleshooting

### "Out of Memory" errors

```python
# Use streaming
ds = load_cohere_msmarco_passages(streaming=True)

# Or reduce limit
ds = load_cohere_msmarco_passages(limit=10_000)
```

### Slow downloads

```python
# Set cache directory to fast storage
ds = load_cohere_msmarco_passages(cache_dir="/scratch/$USER/datasets")
```

### Dataset not found

```bash
# Install datasets library
pip install datasets

# Check Hugging Face access
python -c "from datasets import load_dataset; print(load_dataset.__file__)"
```

---

## Performance Tips

1. **Use pre-embedded datasets** when possible (faster than computing embeddings)
2. **Cache datasets** in a shared location (e.g., `../datasets/`)
3. **Use streaming** for datasets > 1M vectors
4. **Precompute ground truth** once, reuse many times
5. **Save processed datasets** as `.npz` files for quick reloading

---

## Questions?

Post in the `#vector-quantization` Slack channel or see [USAGE.md](USAGE.md) for CLI usage.
