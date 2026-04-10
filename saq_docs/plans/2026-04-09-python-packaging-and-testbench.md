# SAQ Python Packaging and Test Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a proper `pyproject.toml`-based installable Python package for SAQ alongside a pytest-based test bench that covers preprocessing utilities, the C++ bindings interface, and end-to-end recall correctness.

**Architecture:** The `python/` tree already contains a `saq/` package and `preprocessing/` module but has no packaging metadata. We add `pyproject.toml` (scikit-build-core backend) so that `pip install -e .` builds the C++ extension in-place, and a `tests/python/` directory with pytest suites split into three layers: pure-Python unit tests (no C++ required), binding smoke tests (require built `.pyd`/`.so`), and an integration test that builds a tiny IVF index and checks recall against a synthetic ground truth.

**Tech Stack:** Python 3.10+, pytest, scikit-build-core, pybind11, numpy, faiss-cpu (optional for preprocessing tests)

---

## Requirements Analysis

### What exists
- `python/saq/__init__.py` — package entry point, DLL path setup, re-exports from `_saq_core` and optionally `_saq_gpu`
- `python/bindings/saq_bindings.cpp` — pybind11 module `_saq_core` exposing `IVF`, configs, enums, `load_fvecs`/`load_ivecs`
- `python/bindings/saq_gpu_bindings.cpp` — `_saq_gpu` module exposing `GpuIVF`
- `python/bindings/CMakeLists.txt` — builds both modules, copies output to `python/saq/`
- `python/preprocessing/{pca,ivf,compute_gt}.py` + `utils/io.py` — preprocessing CLI scripts
- Root `CMakeLists.txt` — controls `SAQ_BUILD_PYTHON` flag

### What is missing
- No `pyproject.toml` or `setup.py` — no `pip install` path
- No test infrastructure — zero pytest files exist
- `preprocessing/` has no `__init__.py` beyond being a module (works as `-m` target but not importable as library)
- No pytest fixtures for synthetic data generation
- No recall-correctness test with a reproducible small index

### Gaps and assumptions
1. **Build backend choice**: scikit-build-core is the modern standard for pybind11 projects. It delegates to CMake, so we reuse the existing `CMakeLists.txt` without modification.
2. **Editable install**: `pip install -e . --no-build-isolation` is the developer workflow. scikit-build-core supports editable installs via `--editable`.
3. **GPU tests are optional**: Binding smoke tests for `GpuIVF` are skipped when `_saq_gpu` is not importable — consistent with the existing try/except in `__init__.py`.
4. **Synthetic dataset**: Integration tests use a programmatically generated 1000-vector, 64-dim dataset with K=32, nprobe=10, bpd=2.0. This exercises the full pipeline in ~5 seconds without needing the DBpedia files.
5. **Preprocessing tests**: `pca.py` and `ivf.py` require faiss. Tests are skipped (`pytest.importorskip`) when faiss is absent to keep CI lightweight.
6. **Windows path handling**: The DLL directory setup in `__init__.py` is already correct; no changes needed there.

---

## File Structure

```
python/
  pyproject.toml                    # NEW — package metadata + scikit-build-core config
  python/saq/__init__.py            # existing — no changes needed
  python/saq/benchmark.py           # NEW — Python-level recall computation utility (used by tests + users)

tests/
  python/
    conftest.py                     # NEW — shared fixtures (synthetic data, built index)
    test_io_utils.py                # NEW — unit tests for fvecs/ivecs read/write round-trips
    test_preprocessing_utils.py     # NEW — unit tests for io.py helpers (no faiss required)
    test_bindings_smoke.py          # NEW — smoke tests for _saq_core bindings (requires built .pyd)
    test_ivf_integration.py         # NEW — end-to-end IVF build + search + recall correctness
```

---

## Task 1: Add `pyproject.toml` for `pip install`

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["scikit-build-core>=0.9", "pybind11>=2.12"]
build-backend = "scikit_build_core.build"

[project]
name = "saq"
version = "0.1.0"
description = "Scalar Additive Quantization for approximate nearest neighbor search"
requires-python = ">=3.10"
dependencies = ["numpy>=1.20"]

[project.optional-dependencies]
preprocessing = ["faiss-cpu>=1.7", "numpy>=1.20"]
dev = ["pytest>=7.0", "numpy>=1.20"]

[tool.scikit-build]
cmake.args = [
  "-DSAQ_BUILD_PYTHON=ON",
  "-DSAQ_USE_OPENMP=ON",
  "-DSAQ_BUILD_SAMPLES=OFF",
  "-DSAQ_BUILD_TESTS=OFF",
]
cmake.build-type = "Release"
# Output the extension modules into the source tree for editable installs
wheel.packages = ["python/saq"]
# scikit-build-core places extension modules next to the package
install.components = ["python_modules"]

[tool.scikit-build.cmake.define]
SAQ_BUILD_PYTHON = "ON"
SAQ_USE_OPENMP = "ON"

[tool.pytest.ini_options]
testpaths = ["tests/python"]
markers = [
  "requires_bindings: test requires compiled _saq_core extension",
  "requires_faiss: test requires faiss-cpu installation",
  "slow: test takes >10 seconds",
]
```

- [ ] **Step 2: Verify the file is syntactically valid**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: no output (no exception)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml for pip-installable SAQ package (scikit-build-core)"
```

---

## Task 2: Add `python/saq/benchmark.py` — recall utility

This module is useful both for tests and for users running experiments in Python notebooks. It computes Recall@K given result IDs and ground truth IDs.

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/python/saq/benchmark.py`

- [ ] **Step 1: Write the module**

```python
"""Recall and benchmark utilities for SAQ search evaluation."""

from __future__ import annotations

import numpy as np


def recall_at_k(
    results: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Compute Recall@k over a batch of queries.

    Parameters
    ----------
    results:
        Integer array of shape (nq, topk) containing returned neighbor IDs.
    ground_truth:
        Integer array of shape (nq, gt_k) containing ground truth neighbor IDs.
        gt_k must be >= k.
    k:
        Recall cutoff. Counts how many of the top-k ground truth neighbors
        appear anywhere in the returned results[:k].

    Returns
    -------
    float
        Mean recall across all queries, in [0, 1].
    """
    nq = results.shape[0]
    k_res = min(k, results.shape[1])
    k_gt = min(k, ground_truth.shape[1])

    hits = 0
    total = 0
    for q in range(nq):
        gt_set = set(ground_truth[q, :k_gt].tolist())
        for idx in results[q, :k_res].tolist():
            if idx in gt_set:
                hits += 1
        total += k_gt

    return hits / total if total > 0 else 0.0


def compute_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    top_k: int = 100,
) -> np.ndarray:
    """Brute-force ground truth via exact L2 distances.

    Parameters
    ----------
    base:
        Float32 array of shape (n, d).
    queries:
        Float32 array of shape (nq, d).
    top_k:
        Number of nearest neighbors to return per query.

    Returns
    -------
    np.ndarray
        Integer array of shape (nq, top_k) containing 0-indexed neighbor IDs.
    """
    nq = queries.shape[0]
    gt = np.empty((nq, top_k), dtype=np.int32)
    for q in range(nq):
        diffs = base - queries[q]
        dists = np.einsum("nd,nd->n", diffs, diffs)  # squared L2
        gt[q] = np.argsort(dists)[:top_k]
    return gt
```

- [ ] **Step 2: Export from `__init__.py`**

Edit `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/python/saq/__init__.py` — add after the `__all__` block:

```python
from .benchmark import recall_at_k, compute_ground_truth

__all__ += ["recall_at_k", "compute_ground_truth"]
```

- [ ] **Step 3: Commit**

```bash
git add python/saq/benchmark.py python/saq/__init__.py
git commit -m "feat(python): add benchmark.py with recall_at_k and compute_ground_truth"
```

---

## Task 3: Create test infrastructure — `conftest.py` and fixtures

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/tests/python/conftest.py`

- [ ] **Step 1: Write `conftest.py`**

```python
"""Shared pytest fixtures for SAQ Python tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic dataset parameters
# ---------------------------------------------------------------------------
SYNTH_N = 1000       # number of base vectors
SYNTH_DIM = 64       # vector dimensionality
SYNTH_NQ = 50        # number of query vectors
SYNTH_K = 32         # IVF clusters
SYNTH_SEED = 42


@pytest.fixture(scope="session")
def synth_rng() -> np.random.Generator:
    return np.random.default_rng(SYNTH_SEED)


@pytest.fixture(scope="session")
def synth_base(synth_rng: np.random.Generator) -> np.ndarray:
    """Random float32 base vectors, shape (1000, 64)."""
    return synth_rng.standard_normal((SYNTH_N, SYNTH_DIM)).astype(np.float32)


@pytest.fixture(scope="session")
def synth_queries(synth_rng: np.random.Generator) -> np.ndarray:
    """Random float32 query vectors, shape (50, 64)."""
    return synth_rng.standard_normal((SYNTH_NQ, SYNTH_DIM)).astype(np.float32)


@pytest.fixture(scope="session")
def synth_ground_truth(synth_base: np.ndarray, synth_queries: np.ndarray) -> np.ndarray:
    """Exact brute-force L2 ground truth, shape (50, 100)."""
    from saq.benchmark import compute_ground_truth
    return compute_ground_truth(synth_base, synth_queries, top_k=100)


@pytest.fixture(scope="session")
def tmp_dir() -> Path:
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_bindings: requires compiled _saq_core extension"
    )
    config.addinivalue_line(
        "markers", "requires_faiss: requires faiss-cpu to be installed"
    )
    config.addinivalue_line("markers", "slow: test takes >10 seconds")
```

- [ ] **Step 2: Run pytest collection to verify conftest loads cleanly**

Run: `python -m pytest tests/python/ --collect-only -q`
Expected: `0 errors` in collection output (tests not yet written, 0 items is fine)

- [ ] **Step 3: Commit**

```bash
git add tests/python/conftest.py
git commit -m "test: add pytest conftest.py with synthetic data fixtures"
```

---

## Task 4: I/O utility unit tests (no C++ required)

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/tests/python/test_io_utils.py`

- [ ] **Step 1: Write the tests**

```python
"""Unit tests for python/preprocessing/utils/io.py — no C++ or faiss required."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from preprocessing.utils.io import (
    read_fvecs,
    read_ivecs,
    write_fvecs,
    write_ivecs,
)


@pytest.fixture
def tmp_path_local() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestFvecsRoundTrip:
    def test_write_then_read_float32(self, tmp_path_local: Path) -> None:
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = str(tmp_path_local / "test.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        np.testing.assert_array_equal(loaded, data)

    def test_single_row(self, tmp_path_local: Path) -> None:
        data = np.array([[0.1, 0.2]], dtype=np.float32)
        path = str(tmp_path_local / "single.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        np.testing.assert_allclose(loaded, data, rtol=1e-6)

    def test_large_matrix(self, tmp_path_local: Path) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 128)).astype(np.float32)
        path = str(tmp_path_local / "large.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        assert loaded.shape == (500, 128)
        np.testing.assert_array_equal(loaded, data)

    def test_output_dtype_is_float32(self, tmp_path_local: Path) -> None:
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        path = str(tmp_path_local / "dtype.fvecs")
        write_fvecs(path, data)
        loaded = read_fvecs(path)
        assert loaded.dtype == np.float32


class TestIvecsRoundTrip:
    def test_write_then_read_int32(self, tmp_path_local: Path) -> None:
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        path = str(tmp_path_local / "test.ivecs")
        write_ivecs(path, data)
        loaded = read_ivecs(path)
        np.testing.assert_array_equal(loaded, data)

    def test_single_column(self, tmp_path_local: Path) -> None:
        data = np.arange(100, dtype=np.int32).reshape(100, 1)
        path = str(tmp_path_local / "single_col.ivecs")
        write_ivecs(path, data)
        loaded = read_ivecs(path)
        np.testing.assert_array_equal(loaded, data)

    def test_output_dtype(self, tmp_path_local: Path) -> None:
        data = np.array([[10, 20]], dtype=np.int32)
        path = str(tmp_path_local / "dtype.ivecs")
        write_ivecs(path, data)
        loaded = read_ivecs(path)
        assert loaded.dtype == np.int32
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/python/test_io_utils.py -v`
Expected: all 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_io_utils.py
git commit -m "test: add fvecs/ivecs round-trip unit tests"
```

---

## Task 5: `benchmark.py` unit tests (pure Python, no C++)

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/tests/python/test_benchmark_utils.py`

- [ ] **Step 1: Write the tests**

```python
"""Unit tests for saq.benchmark module — no C++ required."""

from __future__ import annotations

import numpy as np
import pytest

from saq.benchmark import compute_ground_truth, recall_at_k


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        # results == ground truth
        results = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        gt = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        assert recall_at_k(results, gt, k=3) == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        results = np.array([[10, 11, 12]], dtype=np.int32)
        gt = np.array([[0, 1, 2]], dtype=np.int32)
        assert recall_at_k(results, gt, k=3) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        # 1 of 2 ground truth neighbors found
        results = np.array([[0, 99, 98]], dtype=np.int32)
        gt = np.array([[0, 1, 2]], dtype=np.int32)
        # k=2: gt top-2 are [0, 1], results top-2 are [0, 99] → 1 hit / 2 total
        assert recall_at_k(results, gt, k=2) == pytest.approx(0.5)

    def test_k_larger_than_results(self) -> None:
        # k > results.shape[1] should not crash; clips to available results
        results = np.array([[0, 1]], dtype=np.int32)
        gt = np.array([[0, 1, 2, 3]], dtype=np.int32)
        r = recall_at_k(results, gt, k=4)
        assert 0.0 <= r <= 1.0

    def test_batch_averaging(self) -> None:
        # query 0: perfect, query 1: zero → average = 0.5
        results = np.array([[0, 1], [9, 8]], dtype=np.int32)
        gt = np.array([[0, 1], [0, 1]], dtype=np.int32)
        assert recall_at_k(results, gt, k=2) == pytest.approx(0.5)


class TestComputeGroundTruth:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.standard_normal((100, 8)).astype(np.float32)
        queries = rng.standard_normal((10, 8)).astype(np.float32)
        gt = compute_ground_truth(base, queries, top_k=5)
        assert gt.shape == (10, 5)

    def test_nearest_neighbor_correctness(self) -> None:
        # Put a query right on top of base vector 7
        base = np.eye(10, 10, dtype=np.float32)
        query = np.zeros((1, 10), dtype=np.float32)
        query[0, 7] = 1.0
        gt = compute_ground_truth(base, query, top_k=1)
        assert gt[0, 0] == 7

    def test_no_duplicate_ids(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.standard_normal((50, 16)).astype(np.float32)
        queries = rng.standard_normal((5, 16)).astype(np.float32)
        gt = compute_ground_truth(base, queries, top_k=10)
        for row in gt:
            assert len(set(row.tolist())) == len(row), "duplicate IDs in ground truth"
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/python/test_benchmark_utils.py -v`
Expected: all 8 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_benchmark_utils.py
git commit -m "test: add benchmark utility unit tests (recall_at_k, compute_ground_truth)"
```

---

## Task 6: Binding smoke tests (require compiled `_saq_core`)

These tests verify the pybind11 module is importable, types are correct, and all exposed attributes exist. They do not build an index — that is covered in Task 7.

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/tests/python/test_bindings_smoke.py`

- [ ] **Step 1: Write the tests**

```python
"""Smoke tests for _saq_core Python bindings.

Marked requires_bindings — skip if the extension module is not built.
Run after: cmake --build build --target _saq_core
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_bindings

# Skip entire module if _saq_core is not available
_saq_core = pytest.importorskip("saq._saq_core", reason="_saq_core not built")


class TestEnums:
    def test_dist_type_values(self) -> None:
        from saq import DistType
        assert hasattr(DistType, "L2Sqr")
        assert hasattr(DistType, "IP")
        assert DistType.L2Sqr != DistType.IP

    def test_base_quant_type_values(self) -> None:
        from saq import BaseQuantType
        assert hasattr(BaseQuantType, "CAQ")
        assert hasattr(BaseQuantType, "RBQ")
        assert hasattr(BaseQuantType, "LVQ")


class TestConfigConstruction:
    def test_quantize_config_defaults(self) -> None:
        from saq import QuantizeConfig
        cfg = QuantizeConfig()
        assert cfg.avg_bits == pytest.approx(0.0)
        assert cfg.enable_segmentation is True

    def test_quant_single_config_defaults(self) -> None:
        from saq import QuantSingleConfig, BaseQuantType
        cfg = QuantSingleConfig()
        assert cfg.quant_type == BaseQuantType.CAQ
        assert cfg.random_rotation is True
        assert cfg.use_fastscan is True

    def test_searcher_config_defaults(self) -> None:
        from saq import SearcherConfig, DistType
        cfg = SearcherConfig()
        assert cfg.dist_type == DistType.L2Sqr
        assert cfg.searcher_vars_bound_m == pytest.approx(4.0)

    def test_config_mutation(self) -> None:
        from saq import QuantizeConfig, DistType, BaseQuantType, QuantSingleConfig
        cfg = QuantizeConfig()
        cfg.avg_bits = 2.0
        cfg.enable_segmentation = False
        assert cfg.avg_bits == pytest.approx(2.0)
        assert cfg.enable_segmentation is False


class TestIVFInterface:
    def test_ivf_default_constructor(self) -> None:
        from saq import IVF
        ivf = IVF()
        assert ivf is not None

    def test_ivf_parameterized_constructor(self) -> None:
        from saq import IVF, QuantizeConfig
        cfg = QuantizeConfig()
        cfg.avg_bits = 2.0
        ivf = IVF(100, 64, 8, cfg)
        assert ivf.num_data == 100
        assert ivf.num_dim == 64
        assert ivf.k == 8

    def test_ivf_has_expected_methods(self) -> None:
        from saq import IVF
        for method in ("set_variance", "construct", "search", "search_batch", "save", "load"):
            assert hasattr(IVF, method), f"IVF missing method: {method}"


class TestLoadUtilities:
    def test_load_fvecs_is_callable(self) -> None:
        from saq import load_fvecs
        assert callable(load_fvecs)

    def test_load_ivecs_is_callable(self) -> None:
        from saq import load_ivecs
        assert callable(load_ivecs)
```

- [ ] **Step 2: Run smoke tests**

Run: `python -m pytest tests/python/test_bindings_smoke.py -v`
Expected: all tests PASS (or SKIP if `_saq_core` is not built yet)

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_bindings_smoke.py
git commit -m "test: add _saq_core binding smoke tests (enums, config, IVF interface)"
```

---

## Task 7: End-to-end IVF integration test (build index + search + recall)

This is the critical correctness test. It builds a small IVF index on a 1000-vector synthetic dataset and asserts that Recall@10 at nprobe=10 exceeds a minimum threshold of 0.60 at 2.0 bpd. The threshold is conservative — the algorithm routinely achieves >0.85 on real data, but synthetic random data with K=32 clusters at 2 bpd can be lower.

**Files:**
- Create: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/tests/python/test_ivf_integration.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end integration tests for IVF index build + search + recall.

Requires compiled _saq_core. Uses synthetic session-scoped data from conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_bindings

pytest.importorskip("saq._saq_core", reason="_saq_core not built")

from saq import (
    DistType,
    BaseQuantType,
    IVF,
    QuantizeConfig,
    QuantSingleConfig,
    SearcherConfig,
)
from saq.benchmark import recall_at_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(bpd: float) -> QuantizeConfig:
    cfg = QuantizeConfig()
    cfg.avg_bits = bpd
    cfg.enable_segmentation = True
    single = QuantSingleConfig()
    single.quant_type = BaseQuantType.CAQ
    single.random_rotation = True
    single.use_fastscan = True
    single.caq_adj_rd_lmt = 6
    cfg.single = single
    return cfg


def _build_index(
    base: np.ndarray,
    k: int,
    bpd: float,
    num_threads: int = 1,
) -> tuple[IVF, np.ndarray, np.ndarray]:
    """K-means cluster + build IVF. Returns (ivf, centroids, cluster_ids)."""
    from sklearn.cluster import MiniBatchKMeans  # type: ignore[import]
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(base)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    cluster_ids = kmeans.labels_.astype(np.uint32)

    cfg = _make_config(bpd)
    ivf = IVF(len(base), base.shape[1], k, cfg)
    ivf.construct(base, centroids, cluster_ids, num_threads)
    return ivf, centroids, cluster_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIVFBuildAndSearch:
    @pytest.mark.slow
    def test_build_completes_without_error(
        self,
        synth_base: np.ndarray,
    ) -> None:
        """Index construction must not raise."""
        pytest.importorskip("sklearn", reason="sklearn needed for K-means in test")
        ivf, _, _ = _build_index(synth_base, k=32, bpd=2.0)
        assert ivf.num_data == len(synth_base)
        assert ivf.num_dim == synth_base.shape[1]
        assert ivf.k == 32

    @pytest.mark.slow
    def test_search_returns_correct_shape(
        self,
        synth_base: np.ndarray,
        synth_queries: np.ndarray,
    ) -> None:
        """search_batch must return (nq, topk) array of uint32."""
        pytest.importorskip("sklearn", reason="sklearn needed for K-means in test")
        ivf, _, _ = _build_index(synth_base, k=32, bpd=2.0)
        cfg = SearcherConfig()
        cfg.dist_type = DistType.L2Sqr
        results = ivf.search_batch(synth_queries, topk=10, nprobe=10, config=cfg)
        assert results.shape == (len(synth_queries), 10)
        assert results.dtype == np.uint32

    @pytest.mark.slow
    def test_recall_at_10_exceeds_threshold(
        self,
        synth_base: np.ndarray,
        synth_queries: np.ndarray,
        synth_ground_truth: np.ndarray,
    ) -> None:
        """Recall@10 at nprobe=10, bpd=2.0 must exceed 0.60 on synthetic data."""
        pytest.importorskip("sklearn", reason="sklearn needed for K-means in test")
        ivf, _, _ = _build_index(synth_base, k=32, bpd=2.0)
        cfg = SearcherConfig()
        cfg.dist_type = DistType.L2Sqr
        results = ivf.search_batch(synth_queries, topk=10, nprobe=10, config=cfg)
        r10 = recall_at_k(results.astype(np.int32), synth_ground_truth, k=10)
        assert r10 >= 0.60, f"Recall@10={r10:.3f} below threshold 0.60"

    @pytest.mark.slow
    def test_recall_increases_with_nprobe(
        self,
        synth_base: np.ndarray,
        synth_queries: np.ndarray,
        synth_ground_truth: np.ndarray,
    ) -> None:
        """Recall@10 at nprobe=20 must be >= recall at nprobe=5."""
        pytest.importorskip("sklearn", reason="sklearn needed for K-means in test")
        ivf, _, _ = _build_index(synth_base, k=32, bpd=2.0)
        cfg = SearcherConfig()
        cfg.dist_type = DistType.L2Sqr

        results_low = ivf.search_batch(synth_queries, topk=10, nprobe=5, config=cfg)
        results_high = ivf.search_batch(synth_queries, topk=10, nprobe=20, config=cfg)

        r_low = recall_at_k(results_low.astype(np.int32), synth_ground_truth, k=10)
        r_high = recall_at_k(results_high.astype(np.int32), synth_ground_truth, k=10)

        assert r_high >= r_low - 0.05, (
            f"Recall should not decrease with more nprobe: "
            f"nprobe=5 → {r_low:.3f}, nprobe=20 → {r_high:.3f}"
        )

    @pytest.mark.slow
    def test_save_and_load_roundtrip(
        self,
        synth_base: np.ndarray,
        synth_queries: np.ndarray,
        tmp_dir,
    ) -> None:
        """Save + load must produce identical search results."""
        pytest.importorskip("sklearn", reason="sklearn needed for K-means in test")
        ivf, _, _ = _build_index(synth_base, k=32, bpd=2.0)
        cfg = SearcherConfig()
        cfg.dist_type = DistType.L2Sqr

        # Search before save
        results_before = ivf.search_batch(synth_queries, topk=5, nprobe=10, config=cfg)

        # Save and reload
        index_path = str(tmp_dir / "test_index.index")
        ivf.save(index_path)

        ivf2 = IVF()
        ivf2.load(index_path)
        results_after = ivf2.search_batch(synth_queries, topk=5, nprobe=10, config=cfg)

        np.testing.assert_array_equal(
            results_before, results_after,
            err_msg="Results differ after save/load roundtrip"
        )
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/python/test_ivf_integration.py -v -m slow`
Expected: all 5 tests PASS (requires sklearn + built `_saq_core`)

- [ ] **Step 3: Commit**

```bash
git add tests/python/test_ivf_integration.py
git commit -m "test: add end-to-end IVF integration tests (build, search, recall, save/load)"
```

---

## Task 8: Wire up `pytest.ini_options` path and verify full suite

The `pyproject.toml` already sets `testpaths = ["tests/python"]`. This task verifies the complete test suite runs cleanly and records the expected output pattern.

**Files:**
- Modify: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/tests/python/conftest.py` (add `sys.path` setup)

- [ ] **Step 1: Add `sys.path` injection for `preprocessing` module**

Edit `tests/python/conftest.py` — add at the top after imports:

```python
import sys
from pathlib import Path

# Make python/ directory importable so `from preprocessing.utils.io import ...` works
_python_dir = Path(__file__).parent.parent.parent / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
```

- [ ] **Step 2: Run the full suite (pure-Python tests only, no bindings)**

Run: `python -m pytest tests/python/ -v -k "not requires_bindings and not slow"`
Expected: `test_io_utils.py` (7 tests) + `test_benchmark_utils.py` (8 tests) = 15 PASS

- [ ] **Step 3: Run full suite with bindings (after building `_saq_core`)**

Run: `python -m pytest tests/python/ -v`
Expected: ~25+ tests, 0 failures. Binding tests skip gracefully if extension is absent.

- [ ] **Step 4: Commit**

```bash
git add tests/python/conftest.py
git commit -m "test: fix sys.path for preprocessing imports; verify full test suite"
```

---

## Task 9: Document the dev workflow in CLAUDE.md

**Files:**
- Modify: `/mnt/e/Documents/OMSCS/07_2026_Spring/CS6999/SAQ/CLAUDE.md`

- [ ] **Step 1: Add Python Testing section**

After the existing `## Build Commands` section, add:

```markdown
## Python Testing

### Install package in editable mode (builds C++ bindings)
```bash
pip install -e . --no-build-isolation
```

### Run pure-Python tests (no C++ build required)
```bash
python -m pytest tests/python/ -k "not requires_bindings and not slow" -v
```

### Run full test suite (requires built `_saq_core`)
```bash
python -m pytest tests/python/ -v
```

### Run only slow integration tests
```bash
python -m pytest tests/python/ -m slow -v
```
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add Python testing workflow to CLAUDE.md"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `pyproject.toml` packaging metadata — Task 1
- [x] `pip install -e .` editable install path — Task 1 (scikit-build-core editable)
- [x] Pure-Python unit tests for I/O utilities — Task 4
- [x] Unit tests for recall computation utility — Task 5
- [x] Smoke tests confirming bindings interface is intact — Task 6
- [x] End-to-end index build + search + recall correctness — Task 7
- [x] Save/load roundtrip correctness — Task 7
- [x] Recall monotonically increases with nprobe — Task 7
- [x] Test suite runnable without C++ build (pure-Python subset) — Task 8

**Placeholder scan:** No TBD, TODO, or "implement later" in any step.

**Type consistency:**
- `synth_base` fixture returns `np.float32` — matches `IVF.construct()` which takes `Eigen::Ref<const FloatRowMat>` (float32)
- `cluster_ids` passed as `np.uint32` — matches `cluster_ids` binding parameter type
- `results` from `search_batch` are `np.uint32`; cast to `np.int32` before `recall_at_k` to match `ground_truth` dtype
- `SearcherConfig` attribute access in tests matches exactly what is bound in `saq_bindings.cpp`

**Known dependency note:** Integration tests use `sklearn.cluster.MiniBatchKMeans` to avoid requiring faiss in the test environment. If sklearn is also unavailable, `pytest.importorskip("sklearn")` skips the test gracefully. Alternatively, a pure-numpy K-means implementation can replace this — but adding sklearn as a dev dependency is the simpler path for a research codebase.
