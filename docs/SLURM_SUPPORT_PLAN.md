# SLURM Support Plan - Remaining Work

This document tracks what needs to be completed before running large-scale benchmarks on ICE.

---

## ✅ Completed (Ready to Use)

### 1. Path Configuration
- ✅ Configurable codebooks directory (`--codebooks-dir`, `$CODEBOOKS_DIR`)
- ✅ Configurable database path (`--db-path`, `$DB_PATH`)
- ✅ Priority: CLI arg > env var > default

### 2. Ground Truth System
- ✅ `Dataset` class supports precomputed ground truth
- ✅ `Dataset` class supports skipping ground truth for large datasets
- ✅ New CLI command: `vq-benchmark precompute-gt`
- ✅ Uses FAISS for efficient k-NN computation
- ✅ Batch processing to avoid OOM
- ✅ GPU support option

### 3. Production Safeguards
- ✅ Dataset parameter now required (no accidental dummy runs)
- ✅ Clear warnings when ground truth is skipped
- ✅ Database path supports `$SLURM_JOB_ID` for unique files

### 4. Documentation
- ✅ USAGE.md - Complete CLI reference
- ✅ SLURM_GUIDE.md - ICE cluster instructions with job scripts
- ✅ This document - Tracking remaining work

---

## 🚧 In Progress / TODO

### 1. Update `sweep.py` Command ⚠️ HIGH PRIORITY

**Status:** Not started

**What needs to be done:**

Add the same configuration options to `sweep.py` that were added to `run_benchmarks.py`:

```python
def sweep(
    method: str = typer.Option("pq", ...),
    dataset: str = typer.Option(..., help="Dataset (REQUIRED)"),  # Make required
    # ... existing params ...
    ground_truth_path: str = typer.Option(None, ...),  # Add this
    codebooks_dir: str = typer.Option(None, ...),      # Add this
    db_path: str = typer.Option(None, ...),            # Add this
):
```

**Why it matters:**
- Sweeps are the primary use case for ICE cluster
- Without these options, can't run sweeps on large datasets with custom paths

**Estimated time:** 30 minutes

---

### 2. Add MSMARCO Dataset Support ⏸️ BLOCKED

**Status:** Waiting for data

**Blocker:** Real MSMARCO data format not yet available

**What needs to be done:**

1. Understand actual MSMARCO data format (TSV? Binary? Pre-embedded?)
2. Implement `load_msmarco_dataset()` function in `datasets.py`
3. Add to both `run` and `sweep` commands
4. Support both:
   - Loading pre-computed embeddings (`.npy` files)
   - Computing embeddings from text (if TSV with passages)

**Example implementation:**
```python
def load_msmarco_dataset(
    data_path: str,
    max_samples: Optional[int] = None,
    precomputed_embeddings: bool = False,
) -> Dataset:
    if precomputed_embeddings:
        vectors = np.load(data_path)
    else:
        # Load TSV, compute embeddings
        ...
    return Dataset(vectors=vectors, skip_ground_truth=True)
```

**Why it matters:**
- Can't run on real data without this
- Primary research goal depends on MSMARCO benchmarks

**Estimated time:** 2-4 hours (once data format is known)

---

### 3. Add Batch Processing Support ⚠️ MEDIUM PRIORITY

**Status:** Not started

**Why this is necessary:**

When datasets are too large to fit in memory, we need to:
1. Load vectors in chunks
2. Process each chunk separately
3. Aggregate results

**Current problem:**
```python
# This loads entire dataset into RAM at once
X = data.vectors  # OOM if dataset > available memory!
model.fit(X)
```

**Example scenario:**
- MSMARCO: 8.8M passages × 768D × 4 bytes = 27GB just for vectors
- Add queries, codebooks, intermediate arrays → 40-50GB total
- ICE nodes: 128GB max → can fit, but tight
- **Larger datasets** (10M+, 1024D) → won't fit at all

**What needs to be done:**

Add chunk-based processing:

```python
def run(
    # ... existing params ...
    data_chunk_start: int = typer.Option(None, help="Start index for data chunk"),
    data_chunk_end: int = typer.Option(None, help="End index for data chunk"),
    batch_size: int = typer.Option(None, help="Process in batches of this size"),
):
    # Load only subset of data
    if data_chunk_start is not None:
        vectors = load_vectors_chunk(path, data_chunk_start, data_chunk_end)

    # Or process in batches
    if batch_size is not None:
        for batch in load_vectors_batches(path, batch_size):
            process_batch(batch)
```

**SLURM job array use case:**
```bash
#!/bin/bash
#SBATCH --array=0-9  # 10 tasks

# Each task processes 1/10 of dataset
START=$(($SLURM_ARRAY_TASK_ID * 1000000))
END=$((($SLURM_ARRAY_TASK_ID + 1) * 1000000))

vq-benchmark run \
    --data-chunk-start $START \
    --data-chunk-end $END \
    ...
```

**Benefits:**
- Can process datasets larger than node memory
- Parallel processing across multiple nodes
- Fault tolerance (re-run failed chunks)

**Estimated time:** 4-6 hours

**Alternative (simpler):** Just require datasets to fit in memory, use 256GB nodes on ICE if needed

---

### 4. Add Memory Validation 🔵 LOW PRIORITY

**Status:** Not started

**What needs to be done:**

Check available memory before loading data:

```python
import psutil

def validate_memory(dataset_size_bytes):
    available = psutil.virtual_memory().available
    required = dataset_size_bytes * 2  # 2x for safety

    if available < required:
        print(f"WARNING: Insufficient memory!")
        print(f"  Available: {available / 1e9:.1f} GB")
        print(f"  Required: {required / 1e9:.1f} GB")
        raise MemoryError("Not enough RAM")
```

**Why it's low priority:**
- SLURM kills jobs that exceed memory anyway
- User can check `seff <job_id>` after job completes
- Not critical for functionality

**Estimated time:** 1 hour

---

### 5. Add Checkpointing 🔵 LOW PRIORITY

**Status:** Not started

**What needs to be done:**

Save intermediate results so jobs can resume if killed:

```python
checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{config_hash}.pkl"

if checkpoint_path.exists():
    model, metrics = load_checkpoint(checkpoint_path)
else:
    model.fit(X)
    metrics = compute_metrics(...)
    save_checkpoint(checkpoint_path, model, metrics)
```

**Why it's low priority:**
- Most benchmarks run in < 4 hours
- ICE time limits are generous (up to 7 days)
- Can just re-run if needed

**Estimated time:** 2-3 hours

---

### 6. Test with Large Dataset 🧪 TESTING

**Status:** Cannot start until MSMARCO data available

**What needs to be tested:**

1. **Memory usage:** Run with 100K vectors, monitor with `htop`
2. **Performance:** Benchmark compression time vs dataset size
3. **Ground truth:** Verify precompute-gt works with large data
4. **SLURM integration:** Submit actual job on ICE
5. **Database merging:** Test merging multiple DBs from job arrays

**Test script:**
```bash
# Generate synthetic large dataset
python scripts/generate_test_data.py \
    --num-vectors 1000000 \
    --dim 768 \
    --output /scratch/$USER/test_vectors.npy

# Precompute GT
vq-benchmark precompute-gt \
    --vectors-path /scratch/$USER/test_vectors.npy \
    --output-path /scratch/$USER/test_gt.npy

# Run benchmark
vq-benchmark run \
    --dataset dummy \
    --num-samples 1000000 \
    --ground-truth-path /scratch/$USER/test_gt.npy \
    --with-recall
```

**Estimated time:** 2-4 hours

---

## Priority Summary

| Priority | Task | Blocker | Time |
|----------|------|---------|------|
| 🔴 HIGH | Update sweep.py | None | 30 min |
| 🟡 MEDIUM | Batch processing | None | 4-6 hrs |
| 🔵 LOW | Memory validation | None | 1 hr |
| 🔵 LOW | Checkpointing | None | 2-3 hrs |
| ⏸️ BLOCKED | MSMARCO support | Need data | 2-4 hrs |
| 🧪 TEST | Large dataset test | Need MSMARCO | 2-4 hrs |

---

## Minimum Viable for ICE Cluster

To run benchmarks on ICE RIGHT NOW, you need:

### ✅ Already Done
- ✅ Configurable paths
- ✅ Ground truth precomputation
- ✅ SLURM job scripts
- ✅ Documentation

### ⚠️ Must Complete First
1. **Update sweep.py** (30 min) - Most important!

### 🤷 Nice to Have (Optional)
2. Batch processing (if datasets > 128GB)
3. Memory validation (convenience)
4. Checkpointing (safety net)

---

## Next Steps

**Immediate (this session):**
1. ✅ Create template SLURM scripts → See `slurm/` directory
2. ✅ Update documentation → USAGE.md, SLURM_GUIDE.md
3. 🚧 Update sweep.py → IN PROGRESS
4. 🚧 Explain why batch processing is needed → See section above

**Short term (before ICE runs):**
1. Update sweep.py with configuration options
2. Test on local machine with synthetic data
3. Create `slurm/` directory with template scripts

**Long term (as needed):**
1. Wait for MSMARCO data format
2. Implement MSMARCO loader
3. Add batch processing if datasets don't fit in memory
4. Run full benchmarks on ICE

---

## Questions to Answer

Before starting large-scale runs:

1. **What is the actual MSMARCO data format?**
   - TSV with text? Pre-computed embeddings? HDF5?

2. **How large will datasets be?**
   - If < 128GB → current code works
   - If > 128GB → need batch processing

3. **Do you have ICE cluster access?**
   - Can test SLURM scripts?
   - What are quota limits on `/scratch/$USER/`?

4. **What metrics are most important?**
   - Recall (needs ground truth)
   - Compression ratio (always computed)
   - QPS (needs FAISS export)
   - Distortion (always computed)

---

## Contact

Questions about this plan? Discuss in `#vector-quantization` Slack or weekly meetings.
