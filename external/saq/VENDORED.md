# Vendored SAQ engine

This directory is a **vendored snapshot** of the SAQ quantization engine that the
`saq` method in this repository loads at runtime (`import saq`, see
`src/haag_vq/methods/search/saq_index.py`).

## Provenance

- **Source:** https://github.com/rohilrs/SAQ
- **Branch:** `feat/gpu-caq-sequential`
- **Commit:** `73e730e`
- **Snapshot taken:** 2026-07-30 (via `git archive`, tracked files only — no build artifacts)

This branch is the unified engine used for every reported result: it contains the
greedy bit-allocation work (`exp/greedy-allocation`), the GPU codebook/kernels
(`gpu-codebook`), and the GPU CAQ + search-distance additions
(sequential Gauss-Seidel CAQ, `search_batch(return_dists=True)`). It is 72 commits
ahead of SAQ `main`; the superseded experiment branches (`optimal-codebook`,
`gaussian_tail_codebook`, `codebook-fidelity`, …) are **not** included — they are
not needed to reproduce results.

This is a frozen copy. It does not track upstream SAQ; to refresh it, re-run
`git archive` from the source repo at the desired commit and update this file.

## Prebuilt CPU wheel (convenience)

`dist/saq-0.2.0-cp310-cp310-manylinux_2_34_x86_64.whl` is a **self-contained CPU
wheel** for running the `saq` method without compiling anything:

```bash
pip install external/saq/dist/saq-0.2.0-cp310-cp310-manylinux_2_34_x86_64.whl
```

It exposes the codebook builders (`build_codebook_lloyd`, `build_codebook_exact`,
`build_codebook_dp`) and allocators used by the `saq` method — **no GPU** (no
`GpuIVF`). It was built `SAQ_USE_FAISS=OFF` and repaired with `auditwheel`, so it
bundles its `libglog`/`libfmt`/`libgomp` deps and needs no external libraries.

**It works only where the tags match:** Linux **x86_64**, **CPython 3.10**,
**glibc ≥ 2.34**, and a CPU with **AVX-512** (the extension `SIGILL`s otherwise).
For any other Python version, CPU, or a GPU build, build from source below.

## Building from source

Builds a Python extension (`_saq_core`, plus `_saq_gpu` when CUDA is enabled). See
this directory's `README.md`/`CMakeLists.txt` for the authoritative details. The
key flags (matching how the project builds it):

- **CPU wheel:**
  ```bash
  cmake -B build -DSAQ_BUILD_PYTHON=ON -DSAQ_USE_FAISS=OFF -DSAQ_BUILD_SAMPLES=OFF
  cmake --build build --target _saq_core -j
  cd python && pip wheel . --no-deps -w dist/     # then: auditwheel repair for portability
  ```
  `SAQ_USE_FAISS=OFF` uses the Eigen BDCSVD fallback for PCA/k-means — **do not**
  link a conda `libfaiss` (ABI mismatch → `SIGABRT`). Requires an **AVX-512**
  compiler (`-mfma`); the result `SIGILL`s on non-AVX-512 (e.g. AMD) CPUs.
- **GPU wheel:** add `-DSAQ_BUILD_CUDA=ON` with a CUDA toolchain
  (`module load cuda/12.6.1` on PACE) and `CMAKE_CUDA_ARCHITECTURES`
  (e.g. `"80;86;89;90-real"` for A100/L40S/H100). Exposes `GpuIVF`.

After building, install the wheel (or put the built `python/` on `PYTHONPATH`) so
`import saq` resolves.
