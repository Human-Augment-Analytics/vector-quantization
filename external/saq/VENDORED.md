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

## Building

The engine builds a Python extension (`saq` / `_saq_core`, plus `_saq_gpu` when CUDA
is enabled). See `README.md` and `CMakeLists.txt` in this directory for the
authoritative instructions. In brief:

- **CPU wheel:** requires an **AVX-512** toolchain (`-mavx512f -mfma`); the build
  `SIGILL`s on non-AVX-512 (e.g. AMD) CPUs.
- **GPU wheel:** configure with `SAQ_BUILD_CUDA=ON` and a CUDA toolchain
  (`module load cuda/12.6.1` on PACE), setting `CMAKE_CUDA_ARCHITECTURES`
  (e.g. `"80;86;89;90-real"` for A100/L40S/H100). Exposes the `GpuIVF` class.

After building, make the module importable (install the wheel, or put the built
`python/` on `PYTHONPATH`) so `import saq` resolves.
