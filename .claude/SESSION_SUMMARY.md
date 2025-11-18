# Session Summary: Repository Cleanup & Consolidation

## Current Task

**User's explicit requirement:**
> "Why are there 2 readmes? and what's with the unnecessary data, docs, notebooks folders? is there a way to consolidate the information in the docs .md files and include all of that in the main README? It just doesn't feel like this repo really follows engineering best practices. And I feel like there's not a clear and CONCISE enough 1, 2, 3 of here's what we're doing in this project, here are the methods currently implemented and the configurable parameters, here's how we run on PACE vs locally if we want, here's what the results should look like and where logs go and how you find results and blah blah. You know? Like it seems scattered and I'm not even sure if everything here is coherent."

## What Needs to Be Done

1. **Create ONE comprehensive README.md** that includes:
   - Clear 1-2-3 of what the project does
   - All method details (from docs/METHODS.md)
   - All dataset details (from docs/DATASETS.md)
   - How to run locally vs PACE/Slurm
   - Where results go and how to query them
   - Configurable parameters for each method

2. **Delete unnecessary folders and files:**
   - `docs/` directory (consolidate into main README)
   - `documentation/` directory (same)
   - `notebooks/` (empty)
   - `examples/` (already deleted earlier)
   - `scripts/README.md` (redundant with main README)
   - `data/msmarco/README.md` (not needed)

3. **Keep only what's essential:**
   - `scripts/benchmark.py` - THE single script to run
   - Main README.md - THE single source of documentation
   - Source code in `src/haag_vq/`
   - `logs/` for results

## Key Information to Include in README

### Methods (5 total)
- **PQ**: M=8,16,32 subquantizers, 32-512:1 compression
- **OPQ**: Same as PQ + rotation matrix, better accuracy
- **SQ**: 8-bit per dimension, 4:1 compression, highest accuracy
- **SAQ**: num_bits=4,6, variable bits per dimension, 8-32:1
- **RaBitQ**: FAISS automatic, 100:1+, extreme compression

### Datasets (4 total)
- `dbpedia-100k`: 100K vectors, 1536-dim, ~1GB memory, quick testing
- `dbpedia-1m`: 1M vectors, 1536-dim, ~6GB memory, production
- `dbpedia-3072`: 1M vectors, 3072-dim, ~12GB memory, high-dimensional
- `msmarco`: 53M vectors, 1024-dim, use --limit for subsets

### Running
```bash
# Local
python scripts/benchmark.py --dataset dbpedia-100k

# PACE/Slurm
sbatch --mem=8G --time=4:00:00 --wrap="python scripts/benchmark.py --dataset dbpedia-1m"
```

### Results
- Saved to `logs/benchmark_runs.db` (SQLite)
- Metrics: compression_ratio, mse, pairwise_distortion, recall@10
- Query with: `sqlite3 logs/benchmark_runs.db`

## Files to Delete

```bash
# Directories
rm -rf docs/
rm -rf documentation/
rm -rf notebooks/
rm -rf examples/  # already deleted

# Individual files
rm scripts/README.md
rm data/msmarco/README.md  # if exists
```

## Current State

- Created `scripts/benchmark.py` as single entry point ✅
- Started updating README but user wants MORE consolidation
- Need to actually delete the scattered docs and create ONE clean README

## Next Steps

1. Read current README.md
2. Create comprehensive new README with ALL information from docs/ consolidated
3. Delete docs/, documentation/, notebooks/, scripts/README.md
4. Verify no important information is lost
5. Result: ONE script, ONE README, clean repo structure

## Important Context

- This is for HAAG research (Georgia Tech)
- Used on PACE/ICE cluster with Slurm
- Memory optimizations already done (pre-allocated arrays, FAISS ground truth)
- Automatic cache detection (shared storage vs $TMPDIR)
- User previously frustrated with over-organization (too many scripts, wrappers, docs)
- NOW frustrated with scattered documentation and unclear structure
- User wants: SIMPLE, CLEAR, CONCISE, PROFESSIONAL

## File Structure After Cleanup

```
vector-quantization/
├── README.md                 # ONE comprehensive doc
├── scripts/
│   └── benchmark.py          # ONE script
├── src/haag_vq/              # Source code
├── logs/                     # Results
└── tests/                    # Tests
```

That's it. Nothing else.
