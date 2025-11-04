# Repository Organization Summary

The repository has been reorganized for clarity and ease of use.

## 🎯 Quick Reference

### To run a sweep:
```bash
./sweep dbpedia-100k          # Quick 100K test
./sweep dbpedia-1m-subset     # Medium 500K
./sweep dbpedia-1m-full       # Full 1M
./sweep custom <args>         # Custom config
```

### To find something:
- **Production scripts** → `scripts/`
- **Documentation** → `docs/`
- **Examples** → `examples/`
- **Core library** → `src/haag_vq/`

## 📁 New Structure

```
vector-quantization/
├── scripts/               # ALL production scripts here
│   ├── run_sweep.py      # Main sweep script (was in root)
│   ├── sweeps/           # Quick-launch scripts
│   ├── jobs/             # Slurm templates
│   └── utils/            # Helper scripts
│
├── docs/                  # ALL documentation here
│   ├── QUICKSTART_PACE.md
│   ├── RUNNING_SWEEPS.md
│   ├── RUNNING_ON_PACE.md
│   ├── REPOSITORY_STRUCTURE.md  # Full structure details
│   └── archive/          # Old/deprecated docs
│
├── examples/              # Demo scripts
│   ├── demo.py
│   └── demo_msmarco_streaming.py
│
├── logs/                  # Generated (gitignored)
├── results/               # Generated (gitignored)
├── .cache/                # Generated (gitignored)
│
└── sweep                  # Convenience wrapper (NEW!)
```

## 🆕 What's New

### 1. Convenience Wrapper
**NEW: `./sweep` command** - Easy interface for running sweeps:
```bash
./sweep dbpedia-100k       # Instead of: ./scripts/sweeps/sweep_dbpedia_100k.sh
./sweep dbpedia-1m-subset  # Instead of: ./scripts/sweeps/sweep_dbpedia_1m_subset.sh
./sweep custom --dataset dbpedia-1536 --methods pq opq
```

### 2. Organized Scripts
All production scripts moved to `scripts/`:
- **Before**: Root directory cluttered with `sweep_*.sh`, `slurm_*.sh`, `run_sweep.py`
- **After**: Clean root, everything in `scripts/`

### 3. Unified Documentation
All docs in `docs/`:
- **Before**: `documentation/` folder + docs in root
- **After**: Everything in `docs/`

### 4. Archived Old Files
Old/deprecated files in `docs/archive/`:
- Old Slurm templates (superseded by new scripts)
- Old quick start guides (superseded by QUICKSTART_PACE.md)
- Temporary fix instructions (no longer needed)

## 📝 Path Changes

### Running Sweeps
```bash
# OLD
./sweep_dbpedia_100k.sh
python run_sweep.py --dataset dbpedia-100k

# NEW
./sweep dbpedia-100k
python scripts/run_sweep.py --dataset dbpedia-100k
```

### Documentation
```bash
# OLD
cat QUICKSTART_PACE.md
cat documentation/RUNNING_ON_PACE.md

# NEW
cat docs/QUICKSTART_PACE.md
cat docs/RUNNING_ON_PACE.md
```

### Scripts
```bash
# OLD
./slurm_demo.sh
./submit_sweep.sh

# NEW
./scripts/jobs/slurm_demo.sh
./scripts/jobs/submit_sweep.sh
```

## ✅ What Stayed the Same

- **Core library**: `src/haag_vq/` (unchanged)
- **Tests**: `tests/` (unchanged)
- **Examples**: `examples/` (unchanged)
- **Database**: `logs/benchmark_runs.db` (same location)
- **Results**: `results/` (same location)

## 🚀 Migration Guide

### If you had scripts that referenced old paths:

1. **Update sweep commands**:
   ```bash
   # Replace
   ./sweep_dbpedia_100k.sh
   # With
   ./sweep dbpedia-100k
   ```

2. **Update Python calls**:
   ```bash
   # Replace
   python run_sweep.py
   # With
   python scripts/run_sweep.py
   ```

3. **Update documentation links**:
   ```bash
   # Replace
   QUICKSTART_PACE.md
   # With
   docs/QUICKSTART_PACE.md
   ```

### If you had custom Slurm scripts:
- Move them to `scripts/jobs/` or `scripts/sweeps/`
- Update any paths they reference

## 📚 Key Documentation

All in `docs/`:
- **QUICKSTART_PACE.md** - Get running in 5 minutes
- **RUNNING_SWEEPS.md** - Complete production guide
- **RUNNING_ON_PACE.md** - PACE/ICE details
- **REPOSITORY_STRUCTURE.md** - Full structure reference
- **DATASETS.md** - Dataset information
- **METHODS.md** - Quantization methods
- **MEMORY_OPTIMIZATIONS.md** - Memory efficiency details

## 🎓 Benefits

✅ **Cleaner root** - No script clutter
✅ **Easier navigation** - Everything has a place
✅ **Better organization** - Scripts/docs/examples separated
✅ **Simpler commands** - Use `./sweep` wrapper
✅ **Consistent structure** - Follows Python project conventions

## 💡 Tips

1. **Use the wrapper**: `./sweep <type>` is the easiest way
2. **Check docs first**: Everything documented in `docs/`
3. **Follow patterns**: Put new scripts in appropriate subdirectories
4. **Keep it clean**: Generated files go in `logs/`, `results/`, `.cache/`

## ❓ Questions?

- Structure details → [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)
- Running sweeps → [docs/RUNNING_SWEEPS.md](docs/RUNNING_SWEEPS.md)
- Quick start → [docs/QUICKSTART_PACE.md](docs/QUICKSTART_PACE.md)
- Main README → [README.md](README.md)
