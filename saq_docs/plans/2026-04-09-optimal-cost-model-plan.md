# Optimal Cost Model Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SAQ's DP bit-allocation cost (`variance / 2^bits`) with precomputed DP-optimal codebook MSE costs, then measure the recall impact via a controlled A/B benchmark.

**Architecture:** A new Python preprocessing script (`compute_costs.py`) generates an `optimal_costs.fvecs` file (D x 9 float32 matrix). `SaqDataMaker::dynamic_programming()` is extended with a second cost path that sums pre-computed MSE values over segment dimensions instead of using the closed-form variance heuristic. `IVF` exposes `set_optimal_costs()` so callers can inject the table before `construct()`. A new benchmark sample (`saq_codebook_compare.cpp`) builds two indexes back-to-back and prints a side-by-side recall and plan comparison.

**Tech Stack:** Python 3 + NumPy (compute_costs.py), C++20 / MSVC + Ninja (SaqDataMaker, IVF, benchmark), Eigen3 (FloatRowMat), existing `io_utils` (fvecs I/O).

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `python/preprocessing/compute_costs.py` | Create | Compute per-dimension optimal codebook MSE at bits 0..8, save as D x 9 fvecs |
| `python/preprocessing/__init__.py` | Read (no change expected) | Confirm module is importable as `preprocessing.compute_costs` |
| `include/saq/quantization_plan.h` | Modify | Add `optimal_costs_` member + `set_optimal_costs` / `has_optimal_costs` to `SaqDataMaker` |
| `src/quantization_plan.cpp` | Modify | Branch on `has_optimal_costs()` inside `dynamic_programming()` for both per-bit cost and 0-bit tail |
| `include/index/ivf_index.h` | Modify | Add `set_optimal_costs(FloatRowMat)` public method; add `pending_optimal_costs_` storage member |
| `src/ivf_index.cpp` | Modify | In `construct()`, call `saq_data_maker_->set_optimal_costs()` if pending costs exist |
| `samples/saq_codebook_compare.cpp` | Create | A/B benchmark: build baseline + optimal-cost IVF, print recall tables + plan diff |
| `samples/CMakeLists.txt` | Modify | Register `saq_codebook_compare` executable |

---

## Task 1: Python — `compute_costs.py`

**Files:**
- Create: `python/preprocessing/compute_costs.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# python/tests/test_compute_costs.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from preprocessing.compute_costs import compute_codebook_dp, compute_all_costs

def test_compute_codebook_dp_shape():
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 500).astype(np.float32)
    costs = compute_codebook_dp(values, max_bits=8, n_bins=200)
    assert costs.shape == (9,), f"expected (9,), got {costs.shape}"

def test_costs_decrease_with_bits():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 500).astype(np.float32)
    costs = compute_codebook_dp(values, max_bits=8, n_bins=200)
    for b in range(1, 9):
        assert costs[b] <= costs[b - 1] + 1e-9, \
            f"cost not monotone: costs[{b}]={costs[b]} > costs[{b-1}]={costs[b-1]}"

def test_compute_all_costs_shape():
    rng = np.random.default_rng(7)
    data = rng.normal(0, 1, (200, 16)).astype(np.float32)
    result = compute_all_costs(data, max_bits=8, n_bins=100, n_samples=200)
    assert result.shape == (16, 9), f"expected (16, 9), got {result.shape}"
    assert result.dtype == np.float32
```

- [ ] **Step 2: Run to confirm failure**

```
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ
python -m pytest python/tests/test_compute_costs.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'preprocessing.compute_costs'`

- [ ] **Step 3: Implement `compute_costs.py`**

```python
"""
Optimal codebook MSE cost computation for SAQ bit allocation.

For each PCA-rotated dimension, runs DP-optimal 1D k-means at
bits 0..max_bits and records the resulting MSE. These costs replace
the variance/2^b heuristic in SaqDataMaker::dynamic_programming().

Usage:
    python -m preprocessing.compute_costs --data-dir data/datasets/dbpedia_100k

Output:
    <data_dir>/optimal_costs.fvecs   -- shape (D, max_bits+1), float32
                                        row d = [MSE_0, MSE_1, ..., MSE_8] for dim d
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

from preprocessing.utils.io import read_somefiles, write_fvecs


def compute_codebook_dp(
    values: np.ndarray,
    max_bits: int = 8,
    n_bins: int = 2000,
) -> np.ndarray:
    """
    Compute DP-optimal codebook MSE for a single dimension at bits 0..max_bits.

    Uses a histogram approximation (n_bins bins) to reduce the O(N^2 * 2^b)
    DP to O(B^2 * 2^b) where B = min(n_bins, N).

    Args:
        values: 1D float32 array of dimension values (already subsampled).
        max_bits: Maximum bit rate to evaluate (inclusive). Default 8.
        n_bins: Histogram bin count. Default 2000.

    Returns:
        costs: float32 array of shape (max_bits + 1,).
                costs[b] = MSE when using 2^b centroids.
    """
    values = np.asarray(values, dtype=np.float64)
    N = len(values)
    if N == 0:
        return np.zeros(max_bits + 1, dtype=np.float32)

    values_sorted = np.sort(values)
    vmin, vmax = values_sorted[0], values_sorted[-1]

    # --- Build histogram ---
    B_req = min(n_bins, N)
    if vmin == vmax:
        # Constant dimension: MSE is 0 for all bit rates
        return np.zeros(max_bits + 1, dtype=np.float32)

    bin_edges = np.linspace(vmin, vmax, B_req + 1)
    bin_indices = np.searchsorted(bin_edges[1:-1], values_sorted)  # shape (N,)

    # Per-bin sufficient statistics
    counts = np.bincount(bin_indices, minlength=B_req).astype(np.float64)
    sums   = np.bincount(bin_indices, weights=values_sorted, minlength=B_req).astype(np.float64)
    sumsqs = np.bincount(bin_indices, weights=values_sorted ** 2, minlength=B_req).astype(np.float64)

    # Remove empty bins
    mask = counts > 0
    counts = counts[mask]
    sums   = sums[mask]
    sumsqs = sumsqs[mask]
    B = len(counts)

    # Prefix sums for O(1) SSE queries
    pc = np.concatenate([[0.0], np.cumsum(counts)])   # shape (B+1,)
    ps = np.concatenate([[0.0], np.cumsum(sums)])     # shape (B+1,)
    pq = np.concatenate([[0.0], np.cumsum(sumsqs)])   # shape (B+1,)

    def sse(a: int, b: int) -> float:
        """SSE of bins[a..b] (inclusive) represented by their weighted mean."""
        c  = pc[b + 1] - pc[a]
        s  = ps[b + 1] - ps[a]
        sq = pq[b + 1] - pq[a]
        if c <= 0:
            return 0.0
        return sq - s * s / c

    costs_out = np.empty(max_bits + 1, dtype=np.float64)

    # 0 bits: single cluster = global mean
    costs_out[0] = sse(0, B - 1) / N

    total_sumsq = pq[B]
    # Per-bin variance sum (for the k >= B case)
    per_bin_var_sum = float(np.sum(sumsqs - sums ** 2 / counts))

    for bits in range(1, max_bits + 1):
        k = 1 << bits  # 2^bits

        if k >= B:
            # More clusters than bins: assign each bin its own centroid
            costs_out[bits] = per_bin_var_sum / N
            continue

        # DP: dp[j, i] = min SSE for bins[0..i] using j clusters
        INF = float('inf')
        dp_prev = np.full(B, INF, dtype=np.float64)  # j-1 clusters
        dp_curr = np.full(B, INF, dtype=np.float64)  # j clusters

        # Base case: j=1
        for i in range(B):
            dp_prev[i] = sse(0, i)

        split = np.zeros((k, B), dtype=np.int32)  # for backtrack (not used here)

        for j in range(2, k + 1):
            dp_curr[:] = INF
            for i in range(j - 1, B):
                best = INF
                for m in range(j - 1, i + 1):
                    prev_val = dp_prev[m - 1] if m > 0 else 0.0
                    candidate = prev_val + sse(m, i)
                    if candidate < best:
                        best = candidate
                dp_curr[i] = best
            dp_prev, dp_curr = dp_curr, dp_prev  # swap buffers

        costs_out[bits] = dp_prev[B - 1] / N

    return costs_out.astype(np.float32)


def compute_all_costs(
    data: np.ndarray,
    max_bits: int = 8,
    n_bins: int = 2000,
    n_samples: int = 5000,
) -> np.ndarray:
    """
    Compute optimal codebook MSE for every dimension independently.

    Args:
        data: Float array of shape (N, D).
        max_bits: Max bits to evaluate per dimension (default 8).
        n_bins: Histogram bins for the DP approximation (default 2000).
        n_samples: Number of values to subsample per dimension (default 5000).

    Returns:
        costs: float32 array of shape (D, max_bits + 1).
    """
    N, D = data.shape
    rng = np.random.default_rng(seed=42)
    costs = np.empty((D, max_bits + 1), dtype=np.float32)

    idx = rng.choice(N, size=min(n_samples, N), replace=False) if N > n_samples else np.arange(N)

    for d in range(D):
        if d % 100 == 0:
            print(f"  dim {d}/{D}...", flush=True)
        costs[d] = compute_codebook_dp(data[idx, d], max_bits=max_bits, n_bins=n_bins)

    return costs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-dimension optimal codebook MSE costs for SAQ DP allocation.",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing vectors_pca.fvecs")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as --data-dir)")
    parser.add_argument("--base-file", default="vectors_pca.fvecs",
                        help="PCA-rotated base vectors filename (default: vectors_pca.fvecs)")
    parser.add_argument("--max-bits", type=int, default=8,
                        help="Maximum bit rate to evaluate (default: 8)")
    parser.add_argument("--n-bins", type=int, default=2000,
                        help="Histogram bins for DP approximation (default: 2000)")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Values subsampled per dimension (default: 5000)")
    args = parser.parse_args()

    data_dir   = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(data_dir, args.base_file)
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run pca.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {data_path}...")
    data = read_somefiles(data_path)
    print(f"Data shape: {data.shape}")

    print(f"Computing optimal costs (max_bits={args.max_bits}, "
          f"n_bins={args.n_bins}, n_samples={args.n_samples})...")
    t0 = time.time()
    costs = compute_all_costs(data, max_bits=args.max_bits,
                              n_bins=args.n_bins, n_samples=args.n_samples)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. costs shape: {costs.shape}")

    out_path = os.path.join(output_dir, "optimal_costs.fvecs")
    write_fvecs(out_path, costs)
    print(f"Saved to {out_path}")

    # Sanity check: costs should be non-increasing with bits
    violations = int(np.sum(np.diff(costs, axis=1) > 1e-6))
    if violations > 0:
        print(f"WARNING: {violations} (dim, bit) pairs where cost increased with more bits.")
    else:
        print("Monotonicity check passed: costs decrease with more bits for all dimensions.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to confirm they pass**

```
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ
python -m pytest python/tests/test_compute_costs.py -v
```

Expected: All 3 tests PASS. (The pure-Python DP is slow but correct for small test inputs.)

---

## Task 2: C++ — Extend `SaqDataMaker` with optional cost table

**Files:**
- Modify: `include/saq/quantization_plan.h`
- Modify: `src/quantization_plan.cpp`

- [ ] **Step 1: Add `optimal_costs_` member and accessors to `SaqDataMaker`**

In `include/saq/quantization_plan.h`, inside the `SaqDataMaker` class (after the existing `data_` member, around line 141):

```cpp
    FloatRowMat optimal_costs_; ///< D x (max_bits+1) table; empty when not set

  public:
    // ... existing public methods ...

    /// @brief Inject precomputed per-dimension MSE costs (D x max_bits+1 matrix).
    ///        Must be called before set_variance() or compute_variance() so that
    ///        analyze_plan() picks up the table.
    void set_optimal_costs(FloatRowMat costs) {
        optimal_costs_ = std::move(costs);
    }

    bool has_optimal_costs() const {
        return optimal_costs_.rows() > 0;
    }
```

- [ ] **Step 2: Replace the per-bit cost in `dynamic_programming()`**

In `src/quantization_plan.cpp`, inside `SaqDataMaker::dynamic_programming()`, replace the inner loop body (around line 160-169).

Current inner loop:
```cpp
for (size_t b = 1; b <= kMaxQuantBit; ++b) {
    auto B_new = used_bits + b * j * kDimPaddingSize + num_bit_factors;
    if (B_new > tot_bits)
        break;
    auto v = var_sum / (1 << b);
    auto &f_to = f[ns + 1][i + j][B_new];
    if (f_to.first > f[ns][i][used_bits].first + v) {
        f_to.first = f[ns][i][used_bits].first + v;
        f_to.second = (i << 4) + b;
    }
}
```

Replace with:
```cpp
for (size_t b = 1; b <= kMaxQuantBit; ++b) {
    auto B_new = used_bits + b * j * kDimPaddingSize + num_bit_factors;
    if (B_new > tot_bits)
        break;

    double v;
    if (has_optimal_costs()) {
        v = 0.0;
        size_t b_clamped = std::min(b, static_cast<size_t>(optimal_costs_.cols() - 1));
        for (size_t blk = i; blk < i + j; ++blk) {
            for (size_t d = blk * kDimPaddingSize;
                 d < (blk + 1) * kDimPaddingSize; ++d) {
                if (d < static_cast<size_t>(optimal_costs_.rows())) {
                    v += static_cast<double>(optimal_costs_(static_cast<Eigen::Index>(d),
                                                            static_cast<Eigen::Index>(b_clamped)));
                }
            }
        }
    } else {
        v = var_sum / (1 << b);
    }

    auto &f_to = f[ns + 1][i + j][B_new];
    if (f_to.first > f[ns][i][used_bits].first + v) {
        f_to.first = f[ns][i][used_bits].first + v;
        f_to.second = (i << 4) + b;
    }
}
```

- [ ] **Step 3: Replace the 0-bit tail cost in `dynamic_programming()`**

Still in `dynamic_programming()`, locate the 0-bit tail block (around line 172-177):

Current:
```cpp
// Also try assigning 0 bits to remaining dimensions (unquantized tail)
auto err0 = var_sum;
if (f[ns][i][used_bits].first + err0 < f[1 + ns][i_end][used_bits].first) {
    f[1 + ns][i_end][used_bits].first = f[ns][i][used_bits].first + err0;
    f[1 + ns][i_end][used_bits].second = (i << 4) + 0;
}
```

Replace with:
```cpp
// Also try assigning 0 bits to remaining dimensions (unquantized tail)
double err0;
if (has_optimal_costs()) {
    err0 = 0.0;
    constexpr size_t b_zero = 0;
    for (size_t blk = i; blk < i_end; ++blk) {
        for (size_t d = blk * kDimPaddingSize;
             d < (blk + 1) * kDimPaddingSize; ++d) {
            if (d < static_cast<size_t>(optimal_costs_.rows())) {
                err0 += static_cast<double>(optimal_costs_(
                    static_cast<Eigen::Index>(d),
                    static_cast<Eigen::Index>(b_zero)));
            }
        }
    }
} else {
    err0 = var_sum;
}
if (f[ns][i][used_bits].first + err0 < f[1 + ns][i_end][used_bits].first) {
    f[1 + ns][i_end][used_bits].first = f[ns][i][used_bits].first + err0;
    f[1 + ns][i_end][used_bits].second = (i << 4) + 0;
}
```

- [ ] **Step 4: Verify the project still builds (baseline sanity)**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && cmake -B build -G Ninja -DSAQ_BUILD_SAMPLES=ON -DSAQ_USE_OPENMP=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl && cmake --build build --target saq 2>&1"
```

Expected: build succeeds with no errors.

---

## Task 3: C++ — IVF plumbing (`set_optimal_costs`)

**Files:**
- Modify: `include/index/ivf_index.h`
- Modify: `src/ivf_index.cpp`

- [ ] **Step 1: Add storage member and public method to `IVF`**

In `include/index/ivf_index.h`, add to the `protected:` section (after `saq_data_maker_` member, around line 53):

```cpp
    FloatRowMat pending_optimal_costs_; ///< Held until construct() calls SaqDataMaker
```

In the `public:` section, add after the `set_variance` method (around line 108):

```cpp
    /// @brief Inject optimal codebook MSE costs before construct().
    ///        The matrix must be shape (num_dim_padded, max_bits+1).
    ///        No-op if called after construct().
    void set_optimal_costs(FloatRowMat costs) {
        pending_optimal_costs_ = std::move(costs);
    }
```

- [ ] **Step 2: Forward costs to `SaqDataMaker` inside `construct()`**

In `src/ivf_index.cpp`, in `IVF::construct()`, inside the "2. prepare SAQ data" block, add the forwarding call before `return_data()`:

```cpp
    // 2. prepare SAQ data
    {
        if (pending_optimal_costs_.rows() > 0) {
            saq_data_maker_->set_optimal_costs(std::move(pending_optimal_costs_));
        }
        if (!saq_data_maker_->is_variance_set()) {
            saq_data_maker_->compute_variance(data);
        }
        saq_data_ = saq_data_maker_->return_data();
        printQPlan(saq_data_.get());
    }
```

- [ ] **Step 3: Build to verify IVF compiles cleanly**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && cmake --build build --target saq 2>&1"
```

Expected: no errors, no warnings about unused members.

---

## Task 4: C++ — Benchmark sample `saq_codebook_compare.cpp`

**Files:**
- Create: `samples/saq_codebook_compare.cpp`
- Modify: `samples/CMakeLists.txt`

- [ ] **Step 1: Create the benchmark source file**

```cpp
/// @file saq_codebook_compare.cpp
/// @brief A/B comparison: variance-cost DP vs. optimal-cost DP for SAQ bit allocation.
///
/// Builds two IVF indexes from identical data:
///   A (baseline) — standard variance / 2^bits cost model
///   B (optimal)  — precomputed DP-optimal codebook MSE costs
///
/// Prints side-by-side recall@1/10/100, build times, and quantization plans
/// so the bit-allocation difference can be inspected directly.
///
/// Usage:
///   saq_codebook_compare <data_dir> [bpd] [num_clusters] [nprobe] [num_threads]
///     data_dir:     path to preprocessed dataset (must contain optimal_costs.fvecs)
///     bpd:          bits per dimension (default 2.0)
///     num_clusters: K (default 4096)
///     nprobe:       nprobe for search (default 200)
///     num_threads:  index construction threads (default 8)

#include "index/ivf_index.h"
#include "saq/config.h"
#include "saq/defines.h"
#include "saq/io_utils.h"
#include "saq/stopw.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef SAQ_USE_OPENMP
#include <omp.h>
#endif

using namespace saq;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float recall_at_k(const std::vector<std::vector<PID>>& results,
                          const UintRowMat& gt, size_t k) {
    size_t nq = results.size();
    size_t correct = 0, total = 0;
    for (size_t q = 0; q < nq; ++q) {
        size_t gt_k  = std::min(k, static_cast<size_t>(gt.cols()));
        size_t res_k = std::min(k, results[q].size());
        std::unordered_set<PID> gt_set;
        for (size_t i = 0; i < gt_k; ++i) gt_set.insert(gt(q, i));
        for (size_t i = 0; i < res_k; ++i) {
            if (gt_set.count(results[q][i])) ++correct;
        }
        total += gt_k;
    }
    return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 0.0f;
}

static std::string plan_string(const SaqData* sd) {
    std::string s;
    size_t d = 0;
    for (auto& [dim, bits] : sd->quant_plan) {
        s += "[" + std::to_string(d) + ".."
           + std::to_string(d + dim) + ")@" + std::to_string(bits) + "b ";
        d += dim;
    }
    return s;
}

struct RunResult {
    std::string label;
    float build_time_s;
    float r1, r10, r100;
    std::string plan;
};

static RunResult run_benchmark(
        const std::string& label,
        const FloatRowMat& data,
        const FloatRowMat& centroids,
        const UintRowMat&  cluster_ids,
        const UintRowMat&  gt,
        const FloatVec&    var_vec,
        const FloatRowMat* optimal_costs,   // nullptr for baseline
        float bpd,
        size_t num_clusters,
        size_t nprobe,
        int    num_threads) {

    size_t num_vecs = static_cast<size_t>(data.rows());
    size_t num_dim  = static_cast<size_t>(data.cols());
    size_t num_q    = static_cast<size_t>(gt.rows());
    constexpr size_t TOPK = 100;

    QuantizeConfig cfg;
    cfg.avg_bits               = bpd;
    cfg.single.quant_type      = BaseQuantType::CAQ;
    cfg.single.random_rotation = true;
    cfg.single.use_fastscan    = true;
    cfg.single.caq_adj_rd_lmt  = 6;
    cfg.enable_segmentation    = true;

    IVF ivf(num_vecs, num_dim, num_clusters, cfg);
    ivf.set_variance(FloatVec(var_vec));
    if (optimal_costs) {
        ivf.set_optimal_costs(FloatRowMat(*optimal_costs));
    }

    StopW build_timer;
    ivf.construct(data, centroids, cluster_ids.data(), num_threads);
    float build_time_s = build_timer.getElapsedTimeMili() / 1000.0f;

    SearcherConfig searcher_cfg;
    searcher_cfg.dist_type = DistType::L2Sqr;

    std::vector<std::vector<PID>> results(num_q, std::vector<PID>(TOPK));
    for (size_t q = 0; q < num_q; ++q) {
        ivf.search<DistType::L2Sqr>(queries.row(q), TOPK, nprobe, searcher_cfg,
                                     results[q].data());
    }

    RunResult r;
    r.label        = label;
    r.build_time_s = build_time_s;
    r.r1           = recall_at_k(results, gt, 1);
    r.r10          = recall_at_k(results, gt, 10);
    r.r100         = recall_at_k(results, gt, 100);
    r.plan         = plan_string(ivf.get_saq_data());
    return r;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    std::string data_dir     = "data/datasets/dbpedia_100k";
    float       bpd          = 2.0f;
    size_t      num_clusters = 4096;
    size_t      nprobe       = 200;
    int         num_threads  = 8;

    if (argc > 1) data_dir     = argv[1];
    if (argc > 2) bpd          = std::stof(argv[2]);
    if (argc > 3) num_clusters = std::stoul(argv[3]);
    if (argc > 4) nprobe       = std::stoul(argv[4]);
    if (argc > 5) num_threads  = std::stoi(argv[5]);

    std::string k_str = std::to_string(num_clusters);

    std::cout << "================================================================\n";
    std::cout << "SAQ Cost Model A/B Comparison\n";
    std::cout << "  data_dir=" << data_dir << "  bpd=" << bpd
              << "  K=" << num_clusters << "  nprobe=" << nprobe << "\n";
    std::cout << "================================================================\n\n";

    // -----------------------------------------------------------------------
    // Load data
    // -----------------------------------------------------------------------
    auto data_file     = data_dir + "/vectors_pca.fvecs";
    auto query_file    = data_dir + "/queries_pca.fvecs";
    auto centroid_file = data_dir + "/centroids_" + k_str + "_pca.fvecs";
    auto cids_file     = data_dir + "/cluster_ids_" + k_str + ".ivecs";
    auto gt_file       = data_dir + "/groundtruth.ivecs";
    auto var_file      = data_dir + "/variances_pca.fvecs";
    auto costs_file    = data_dir + "/optimal_costs.fvecs";

    for (const auto& f : {data_file, query_file, centroid_file, cids_file, gt_file, var_file}) {
        if (!file_exists(f.c_str())) {
            std::cerr << "ERROR: required file missing: " << f << "\n";
            return 1;
        }
    }
    if (!file_exists(costs_file.c_str())) {
        std::cerr << "ERROR: optimal_costs.fvecs missing. Run:\n"
                  << "  python -m preprocessing.compute_costs --data-dir " << data_dir << "\n";
        return 1;
    }

    FloatRowMat data, queries, centroids, variances, optimal_costs_mat;
    UintRowMat  cluster_ids, gt;

    load_something<float,    FloatRowMat>(data_file.c_str(),     data);
    load_something<float,    FloatRowMat>(query_file.c_str(),    queries);
    load_something<float,    FloatRowMat>(centroid_file.c_str(), centroids);
    load_something<uint32_t, UintRowMat >(cids_file.c_str(),     cluster_ids);
    load_something<uint32_t, UintRowMat >(gt_file.c_str(),       gt);
    load_something<float,    FloatRowMat>(var_file.c_str(),      variances);
    load_something<float,    FloatRowMat>(costs_file.c_str(),    optimal_costs_mat);

    FloatVec var_vec = variances.row(0);

    std::cout << "Data:    " << data.rows() << " x " << data.cols() << "\n";
    std::cout << "Queries: " << queries.rows() << " x " << queries.cols() << "\n";
    std::cout << "Costs:   " << optimal_costs_mat.rows()
              << " x " << optimal_costs_mat.cols() << "\n\n";

    // -----------------------------------------------------------------------
    // Run A: baseline (variance cost)
    // -----------------------------------------------------------------------
    std::cout << "--- Run A: variance cost (baseline) ---\n";
    auto result_a = run_benchmark("A-baseline", data, centroids, cluster_ids, gt,
                                  var_vec, nullptr, bpd, num_clusters, nprobe, num_threads);

    // -----------------------------------------------------------------------
    // Run B: optimal costs
    // -----------------------------------------------------------------------
    std::cout << "\n--- Run B: optimal codebook MSE costs ---\n";
    auto result_b = run_benchmark("B-optimal", data, centroids, cluster_ids, gt,
                                  var_vec, &optimal_costs_mat, bpd, num_clusters,
                                  nprobe, num_threads);

    // -----------------------------------------------------------------------
    // Print comparison table
    // -----------------------------------------------------------------------
    std::cout << "\n================================================================\n";
    std::cout << "Results at nprobe=" << nprobe << "  bpd=" << bpd << "\n";
    std::cout << "================================================================\n";

    auto pct = [](float v) { return std::fixed << std::setprecision(2) << (v * 100.0f) << "%"; };

    std::cout << std::setw(14) << "Run"
              << std::setw(12) << "Build(s)"
              << std::setw(12) << "R@1"
              << std::setw(12) << "R@10"
              << std::setw(12) << "R@100"
              << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (const auto& r : {result_a, result_b}) {
        std::cout << std::setw(14) << r.label
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.build_time_s
                  << std::setw(11) << std::fixed << std::setprecision(2) << (r.r1   * 100) << "%"
                  << std::setw(11) << std::fixed << std::setprecision(2) << (r.r10  * 100) << "%"
                  << std::setw(11) << std::fixed << std::setprecision(2) << (r.r100 * 100) << "%"
                  << "\n";
    }
    std::cout << std::string(62, '-') << "\n";

    std::cout << "\nDelta R@1:   "
              << std::fixed << std::setprecision(2)
              << ((result_b.r1   - result_a.r1)   * 100) << " pp\n";
    std::cout << "Delta R@10:  "
              << std::fixed << std::setprecision(2)
              << ((result_b.r10  - result_a.r10)  * 100) << " pp\n";
    std::cout << "Delta R@100: "
              << std::fixed << std::setprecision(2)
              << ((result_b.r100 - result_a.r100) * 100) << " pp\n";

    std::cout << "\nPlan A: " << result_a.plan << "\n";
    std::cout << "Plan B: " << result_b.plan << "\n";

    if (result_a.plan == result_b.plan) {
        std::cout << "\nWARNING: Plans are identical — cost model may not be active.\n";
    } else {
        std::cout << "\nPlans differ — optimal cost model is changing bit allocation.\n";
    }

    return 0;
}
```

Note: the `run_benchmark` lambda captures `queries` from the enclosing `main()` scope. Move the lambda body into `main()` or pass `queries` explicitly. The code above has a bug: `queries` is not a parameter of `run_benchmark`. Fix by passing `const FloatRowMat& queries` as an additional argument and threading it through.

- [ ] **Step 2: Fix the `queries` parameter bug (corrected signature)**

The `run_benchmark` function signature must include `queries`:

```cpp
static RunResult run_benchmark(
        const std::string& label,
        const FloatRowMat& data,
        const FloatRowMat& queries,        // <-- add this
        const FloatRowMat& centroids,
        const UintRowMat&  cluster_ids,
        const UintRowMat&  gt,
        const FloatVec&    var_vec,
        const FloatRowMat* optimal_costs,
        float bpd,
        size_t num_clusters,
        size_t nprobe,
        int    num_threads)
```

And the two call sites in `main()` become:

```cpp
auto result_a = run_benchmark("A-baseline", data, queries, centroids, cluster_ids, gt,
                              var_vec, nullptr, bpd, num_clusters, nprobe, num_threads);

auto result_b = run_benchmark("B-optimal",  data, queries, centroids, cluster_ids, gt,
                              var_vec, &optimal_costs_mat, bpd, num_clusters, nprobe, num_threads);
```

- [ ] **Step 3: Register in `samples/CMakeLists.txt`**

Append after the existing `saq_dbpedia_sample` block:

```cmake
add_executable(saq_codebook_compare
  ${CMAKE_CURRENT_SOURCE_DIR}/saq_codebook_compare.cpp
)
target_link_libraries(saq_codebook_compare PRIVATE saq)
target_include_directories(saq_codebook_compare PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(saq_codebook_compare PROPERTIES
  VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
if(MSVC)
  target_compile_options(saq_codebook_compare PRIVATE /arch:AVX512)
else()
  target_compile_options(saq_codebook_compare PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vl)
endif()
```

- [ ] **Step 4: Build the new sample**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && cmake --build build --target saq_codebook_compare 2>&1"
```

Expected: compiles and links with zero errors.

---

## Task 5: End-to-end run and validation

**Files:** (none created — this is the validation pass)

- [ ] **Step 1: Generate `optimal_costs.fvecs`**

```
cd /mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ
python -m preprocessing.compute_costs --data-dir data/datasets/dbpedia_100k
```

Expected output ends with:
```
Saved to data/datasets/dbpedia_100k/optimal_costs.fvecs
Monotonicity check passed: costs decrease with more bits for all dimensions.
```

Verify shape:
```python
import numpy as np, sys
sys.path.insert(0,'python')
from preprocessing.utils.io import read_somefiles
c = read_somefiles('data/datasets/dbpedia_100k/optimal_costs.fvecs')
print(c.shape)  # should be (1536, 9) for dbpedia_100k (D=1536, bits 0..8)
```

- [ ] **Step 2: Verify baseline is unchanged (regression check)**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && build\samples\saq_dbpedia_sample.exe data\datasets\dbpedia_100k results\saq 2.0 4096 200 8 2>&1"
```

Expected: R@1 ~92.8%, R@10 ~92.6%, R@100 ~90.0% — matching prior benchmark results.

- [ ] **Step 3: Run the A/B comparison at 2 bpd**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && build\samples\saq_codebook_compare.exe data\datasets\dbpedia_100k 2.0 4096 200 8 2>&1"
```

Expected:
- Plan A and Plan B differ in bit allocation
- Delta R@1/10/100 values (positive = optimal costs help, negative = they hurt)
- Build times are similar between A and B

- [ ] **Step 4: Run at 4 bpd**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && build\samples\saq_codebook_compare.exe data\datasets\dbpedia_100k 4.0 4096 200 8 2>&1"
```

Expected: similar structure; at higher bpd the allocation effect may be smaller.

- [ ] **Step 5: Run at 1 bpd (expected degradation)**

```
cmd.exe /c "cd /d E:\Documents\OMSCS\07_2026_Spring\CS6999\SAQ && build\samples\saq_codebook_compare.exe data\datasets\dbpedia_100k 1.0 4096 200 8 2>&1"
```

Based on Python experiment, optimal costs at 1 bpd increase MSE (harmful reallocation). Expect negative delta — this validates the cost model is active and the experiment is reproducing the expected behavior.

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|-----------|
| `compute_costs.py` — D x 9 fvecs, 2000 bins, 5000 samples | Task 1 |
| `SaqDataMaker::set_optimal_costs` / `has_optimal_costs` | Task 2, Step 1 |
| DP cost branch: sum `optimal_costs_(d, b)` over segment | Task 2, Step 2 |
| 0-bit tail cost also uses optimal_costs | Task 2, Step 3 |
| `IVF::set_optimal_costs` forwarding | Task 3 |
| `saq_codebook_compare` A/B benchmark | Task 4 |
| Side-by-side recall + plan comparison printout | Task 4 |
| Baseline `saq_dbpedia_sample` unaffected | Task 5, Step 2 |
| Python unit tests (shape + monotonicity) | Task 1, Steps 1-4 |

**Placeholder scan:** No TBDs, no "implement later" phrases. All code blocks are complete.

**Type consistency:**
- `FloatRowMat` used for `optimal_costs_` everywhere (matches `set_variance` convention in `SaqDataMaker`).
- `FloatVec var_vec = variances.row(0)` matches existing usage in `saq_dbpedia_sample.cpp`.
- `optimal_costs_.cols()` is `Eigen::Index`; cast to `size_t` via `static_cast` before `std::min`.

**Known issue flagged inline:** `run_benchmark` in Task 4 Step 1 references `queries` from outer scope — corrected immediately in Step 2 with explicit parameter addition. The final file written should incorporate both steps as a single consistent source.
