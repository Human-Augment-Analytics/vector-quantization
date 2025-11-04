# PACE/ICE Setup Guide

One-time setup to optimize dataset downloads and caching.

## The Caching Strategy

**Problem**: Each Slurm job gets a fresh `$TMPDIR`, so datasets re-download every job.

**Solution**: Download datasets **once** to shared persistent storage, then all jobs use that cache.

## One-Time Setup (5 minutes)

### Step 1: SSH to PACE/ICE
```bash
ssh <your-username>@login-ice.pace.gatech.edu
```

### Step 2: Navigate to Project
```bash
cd /storage/ice-shared/cs8903onl/vector_quantization/vector-quantization
```

### Step 3: Create Shared Cache Directory
```bash
mkdir -p /storage/ice-shared/cs8903onl/.cache/huggingface/datasets
```

### Step 4: Download Datasets to Shared Cache

```bash
module load python/3.12
source .venv312/bin/activate

# Download DBpedia 100K (~500 MB, ~2 min)
python -c "
from haag_vq.data import load_dbpedia_openai_1536_100k
print('Downloading DBpedia 100K...')
load_dbpedia_openai_1536_100k(
    cache_dir='/storage/ice-shared/cs8903onl/.cache/huggingface/datasets',
    limit=1  # Just download, don't process
)
print('✓ DBpedia 100K cached')
"

# Download DBpedia 1M 1536-dim (~6 GB, ~10 min)
python -c "
from haag_vq.data import load_dbpedia_openai_1536
print('Downloading DBpedia 1M (1536-dim)...')
load_dbpedia_openai_1536(
    cache_dir='/storage/ice-shared/cs8903onl/.cache/huggingface/datasets',
    limit=1
)
print('✓ DBpedia 1M (1536) cached')
"

# Optional: DBpedia 1M 3072-dim (~12 GB, ~20 min)
python -c "
from haag_vq.data import load_dbpedia_openai_3072
print('Downloading DBpedia 1M (3072-dim)...')
load_dbpedia_openai_3072(
    cache_dir='/storage/ice-shared/cs8903onl/.cache/huggingface/datasets',
    limit=1
)
print('✓ DBpedia 1M (3072) cached')
"

# Optional: Cohere MS MARCO subset (~1 GB, ~5 min)
python -c "
from haag_vq.data import load_cohere_msmarco_passages
print('Downloading Cohere MS MARCO subset...')
load_cohere_msmarco_passages(
    cache_dir='/storage/ice-shared/cs8903onl/.cache/huggingface/datasets',
    limit=100_000,
    streaming=True
)
print('✓ MS MARCO subset cached')
"
```

### Step 5: Verify Cache
```bash
ls -lh /storage/ice-shared/cs8903onl/.cache/huggingface/datasets/

# You should see:
# Qdrant___dbpedia-entities-openai3-text-embedding-3-large-1536-100_k/
# Qdrant___dbpedia-entities-openai3-text-embedding-3-large-1536-1_m/
# (and others if you downloaded them)
```

## How Scripts Use the Cache

**The scripts now automatically check for shared cache first:**

```python
# In scripts/run_sweep.py and examples/*.py
SHARED_CACHE = "/storage/ice-shared/cs8903onl/.cache/huggingface"

if os.path.exists(SHARED_CACHE):
    # Use shared cache - NO re-download!
    hf_cache_base = SHARED_CACHE
    print("[INFO] Using shared persistent cache")
elif "TMPDIR" in os.environ:
    # Fall back to $TMPDIR - will re-download
    hf_cache_base = os.path.join(os.environ["TMPDIR"], "hf_cache")
    print("[INFO] Using $TMPDIR - will re-download")
```

## Benefits

**Before setup**:
```
Job 1: Downloads DBpedia 100K (~2 min)
Job 2: Downloads DBpedia 100K again (~2 min)
Job 3: Downloads DBpedia 100K again (~2 min)
...
```

**After setup**:
```
Job 1: Uses shared cache (instant)
Job 2: Uses shared cache (instant)
Job 3: Uses shared cache (instant)
...
```

**Savings**: ~2-10 minutes per job + reduced network load

## Disk Usage

| Dataset | Download Size | Disk Space |
|---------|---------------|------------|
| DBpedia 100K | ~500 MB | ~500 MB |
| DBpedia 1M (1536) | ~6 GB | ~6 GB |
| DBpedia 1M (3072) | ~12 GB | ~12 GB |
| MS MARCO (100K) | ~1 GB | ~1 GB |
| **Total (all)** | **~19 GB** | **~19 GB** |

**Your quota**: Check with `quota -s` on PACE/ICE

## Cleaning Up

If you need to free space:

```bash
# Remove specific dataset
rm -rf /storage/ice-shared/cs8903onl/.cache/huggingface/datasets/Qdrant___dbpedia-entities-openai3-text-embedding-3-large-1536-100_k

# Remove all cached datasets
rm -rf /storage/ice-shared/cs8903onl/.cache/huggingface/datasets/*

# Next job will re-download
```

## Running Jobs After Setup

Just run normally - scripts will automatically use shared cache:

```bash
./sweep dbpedia-100k
# [INFO] Using shared persistent cache: /storage/ice-shared/cs8903onl/.cache/huggingface
# ✓ No re-download needed!
```

## Troubleshooting

### "Permission denied" on shared cache
```bash
# Make sure cache directory has correct permissions
chmod -R 775 /storage/ice-shared/cs8903onl/.cache
```

### "Dataset not found in cache"
```bash
# Re-run the download script for that dataset
python -c "from haag_vq.data import load_dbpedia_openai_1536_100k; load_dbpedia_openai_1536_100k(cache_dir='/storage/ice-shared/cs8903onl/.cache/huggingface/datasets', limit=1)"
```

### "Out of quota"
```bash
# Check usage
quota -s

# Remove less-used datasets
rm -rf /storage/ice-shared/cs8903onl/.cache/huggingface/datasets/Qdrant___dbpedia-entities-openai3-text-embedding-3-large-3072-1_m
```

## Summary

✅ **One-time setup**: Download datasets to shared storage
✅ **All jobs**: Automatically use shared cache
✅ **No re-downloads**: Save time and network bandwidth
✅ **Production-ready**: All 4 datasets cached and ready

Next: [Run your first sweep](QUICKSTART_PACE.md)
