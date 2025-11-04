# Repository Structure

Clean, organized structure for production use.

## Directory Layout

```
vector-quantization/
├── src/haag_vq/              # Core library code
│   ├── data/                # Data loaders
│   ├── methods/             # Quantization implementations
│   ├── metrics/             # Evaluation metrics
│   ├── utils/               # Utilities
│   └── visualization/       # Plotting tools
│
├── scripts/                  # Production execution scripts
│   ├── run_sweep.py         # Main configurable sweep script
│   ├── sweeps/              # Pre-configured sweep scripts
│   │   ├── sweep_dbpedia_100k.sh
│   │   ├── sweep_dbpedia_1m_subset.sh
│   │   └── sweep_dbpedia_1m_full.sh
│   ├── jobs/                # Slurm job templates
│   │   ├── slurm_demo.sh
│   │   ├── slurm_sweep.sh
│   │   ├── slurm_msmarco.sh
│   │   └── submit_sweep.sh
│   └── utils/               # Helper scripts
│       └── msmarco_setup.sh
│
├── docs/                     # All documentation
│   ├── QUICKSTART_PACE.md   # 5-minute quickstart
│   ├── RUNNING_SWEEPS.md    # Production sweep guide
│   ├── RUNNING_ON_PACE.md   # PACE/ICE details
│   ├── DATASETS.md          # Dataset information
│   ├── METHODS.md           # Quantization methods
│   ├── MEMORY_OPTIMIZATIONS.md
│   ├── MS_MARCO_STREAMING.md
│   └── archive/             # Archived docs
│
├── examples/                 # Example/demo scripts
│   ├── demo.py              # DBpedia 100K demo
│   └── demo_msmarco_streaming.py
│
├── tests/                    # Test files
│   ├── test_methods.py
│   └── test_metrics.py
│
├── logs/                     # Generated logs (gitignored)
│   ├── benchmark_runs.db    # SQLite results database
│   └── *.out, *.err         # Job output logs
│
├── results/                  # Generated results (gitignored)
│   └── <sweep_id>_summary.txt
│
├── .cache/                   # HuggingFace cache (gitignored)
│   └── datasets/
│
├── notebooks/                # Jupyter notebooks
│
├── sweep                     # Convenience wrapper script
├── README.md                 # Main documentation
├── pyproject.toml           # Package configuration
└── requirements.txt         # Python dependencies
```

## Key Files

### Root Level
- **`sweep`** - Convenience wrapper for running sweeps
- **`README.md`** - Main project documentation
- **`pyproject.toml`** - Python package configuration

### Scripts
- **`scripts/run_sweep.py`** - Main configurable sweep script
- **`scripts/sweeps/*.sh`** - Pre-configured Slurm submission scripts
- **`scripts/jobs/*.sh`** - Slurm job templates

### Documentation
- **`docs/QUICKSTART_PACE.md`** - Get started in 5 minutes
- **`docs/RUNNING_SWEEPS.md`** - Complete production guide
- **`docs/RUNNING_ON_PACE.md`** - PACE/ICE cluster details
- **`docs/DATASETS.md`** - Dataset information
- **`docs/METHODS.md`** - Quantization method details

### Examples
- **`examples/demo.py`** - Complete demo on DBpedia 100K
- **`examples/demo_msmarco_streaming.py`** - Streaming demo for large datasets

## Usage Patterns

### Run a Sweep
```bash
# Using convenience wrapper
./sweep dbpedia-100k

# Using script directly
./scripts/sweeps/sweep_dbpedia_100k.sh

# Custom sweep
python scripts/run_sweep.py --dataset dbpedia-1536 --methods pq opq
```

### View Results
```bash
# Latest results
sqlite3 logs/benchmark_runs.db "SELECT * FROM runs ORDER BY timestamp DESC LIMIT 10;"

# Summary files
cat results/<sweep_id>_summary.txt
```

### Generate Plots
```bash
vq-benchmark plot --output-dir plots/
```

## File Organization Principles

1. **Production scripts** → `scripts/`
2. **Documentation** → `docs/`
3. **Examples/demos** → `examples/`
4. **Generated files** → `logs/`, `results/`, `.cache/` (gitignored)
5. **Library code** → `src/haag_vq/`
6. **Tests** → `tests/`

## Gitignored Directories

The following are generated during runtime and not committed:
- `logs/` - Job outputs and database
- `results/` - Sweep summaries
- `.cache/` - HuggingFace datasets
- `.venv/`, `.venv312/` - Virtual environments
- `build/`, `*.egg-info/` - Build artifacts

## Migration from Old Structure

Old structure → New location:
- Root-level `*.sh` scripts → `scripts/sweeps/` or `scripts/jobs/`
- Root-level `documentation/` → `docs/`
- Root-level `slurm/` → `docs/archive/` (old templates)
- Root-level `QUICKSTART_PACE.md` etc → `docs/`

## Adding New Files

- **New sweep script** → Add to `scripts/sweeps/`
- **New documentation** → Add to `docs/`
- **New example** → Add to `examples/`
- **New utility** → Add to `scripts/utils/`
- **New method** → Add to `src/haag_vq/methods/`

## See Also

- [QUICKSTART_PACE.md](QUICKSTART_PACE.md) - Get started quickly
- [RUNNING_SWEEPS.md](RUNNING_SWEEPS.md) - Production sweep guide
- [Main README](../README.md) - Project overview
