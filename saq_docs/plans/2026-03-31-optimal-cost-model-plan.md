# Implementation Plan: Optimal Cost Model

Spec: `docs/superpowers/specs/2026-03-31-optimal-cost-model-design.md`
Branch: `feat/optimal-cost-model` (off `main`)

## Tasks

### 1. Python: `preprocessing/compute_costs.py`
- Reuse `compute_codebook_dp()` from `experiments/compute_codebook.py`
- CLI: `--data-dir`, outputs `optimal_costs.fvecs`
- D × 9 matrix (float32), row = dimension, cols = bits 0..8
- Use 2000 histogram bins, 5000 subsamples per dim
- Run on dbpedia_100k data, verify output shape

### 2. C++: `SaqDataMaker` cost model
- `quantization_plan.h`: Add `FloatRowMat optimal_costs_` member, `set_optimal_costs()`, `has_optimal_costs()`
- `quantization_plan.cpp`: In `dynamic_programming()`:
  - When `optimal_costs_` is set: sum `optimal_costs_(d, min(b, max_col))` over segment dims
  - When not set: existing `var_sum / (1 << b)` (backward compatible)
  - Also handle 0-bit tail cost with optimal costs

### 3. C++: IVF plumbing
- `ivf_index.h`: Add `void set_optimal_costs(FloatRowMat costs)` public method
- Forward to internal `SaqDataMaker` (store costs, set before `construct()`)
- `saq_data_maker_` is created in `construct()` — need to store costs as member, pass during construct

### 4. C++: Benchmark sample `saq_codebook_compare.cpp`
- Load data + optimal_costs.fvecs
- Build two IVF indexes: baseline (A) and optimal-cost (B)
- Search both at multiple nprobe values
- Compute and print:
  - Recall@1/10/100 side-by-side
  - Quantization plan comparison
  - Cosine distortion: actual cosine sim of returned results vs ground truth top-K
- Register in `samples/CMakeLists.txt`

### 5. Build and run
- `cmake -B build -G Ninja -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`
- First run `python -m preprocessing.compute_costs` to generate optimal_costs.fvecs
- Then run the comparison benchmark
- Report results
