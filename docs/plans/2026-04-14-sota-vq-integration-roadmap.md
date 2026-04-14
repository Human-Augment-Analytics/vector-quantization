# SotA VQ Integration Roadmap

**Status:** planning / WIP
**Owner:** Rohil Shah
**Last touched:** 2026-04-14
**Goal:** Benchmark this repo's VQ methods against the full SotA, fairly and reproducibly.

---

## 1. Why this doc exists

The current `ivf_benchmark` / `run_benchmarks` harness has good bones (BaseSearchIndex contract, timestamped results) but the method roster is incomplete vs. the current VQ SotA. Before drawing any "SAQ is X× better than Y" conclusion from our CSVs, the comparison set needs to cover the methods that actually move the Pareto front as of 2025–2026.

This doc captures:
- The target method list and why each is in scope.
- Integration cost and risk per method (some are 30 lines; some are multi-day C++ spikes).
- A priority order so we can land increments, not one big PR.
- The open design questions that should be answered before a full re-benchmark.

Resume here on a new session. Each section has "Next action" that's picking-up-friendly.

---

## 2. Current method roster (post 2026-04-14)

| Method | Status | Runner(s) | Notes |
|---|---|---|---|
| PQ (`pq_flat`) | ✅ | `run_benchmarks`, `ivf_benchmark` | Classic baseline. Jégou 2011. |
| OPQ (`opq_flat`) | 🟡 partial | `sweep.py` only; needs promotion | Quantizer exists (`OptimizedProductQuantizer`) — just needs branch registrations. Ge 2013. |
| SQ (`sq_flat`) | ✅ | `run_benchmarks`, `ivf_benchmark` | Weak baseline. |
| SAQ (`saq`) | ✅ | Both, via C++ `SaqIndex` wrapper | Post-Apr-14 fix — previously silently using a Python scaffold. |
| Faiss IVFPQ (`faiss_ivfpq`) | ✅ | Both | Reference for indexed PQ. |
| RaBitQ 1-bit flat (`rabitq`) | ✅ native estimator | Both, via `RaBitQIndex` (faiss.IndexRaBitQ) | Post-Apr-14 upgrade — recall lifted 0.64 → ~0.70 vs the old decode+flat path. |
| IVF+RaBitQ (`rabitq_ivf`) | ❌ not in | — | Cheap follow-up using `faiss.IndexIVFRaBitQ`. Track A. |
| **Extended-RaBitQ** (multi-bit) | ❌ not in | — | SIGMOD 2025, VectorDB-NTU. **SotA for random-projection VQ.** Priority 1. |
| ScaNN (anisotropic PQ + rerank) | ❌ not in | — | Guo et al. 2020. Python-native (`pip install scann`). Priority 2. |
| Additive Quantization (AQ) | ❌ not in | — | Babenko & Lempitsky 2014. Check if faiss's `IndexResidualQuantizer` / `LocalSearchQuantizer` can be adapted. Priority 3. |
| LVQ (Locally-adaptive VQ) | ❌ not in | — | Tepper et al. 2023 (Intel Labs). Needs source identification; not in faiss. Priority 4. |

Legend: ✅ fully wired, 🟡 exists in some runner but not the primary comparison harness, ❌ not integrated.

---

## 3. Priority 1: Extended-RaBitQ

**Paper:** "Practical and Asymptotically Optimal Quantization" — SIGMOD 2025. VectorDB-NTU.
**Repos:**
- Original: https://github.com/VectorDB-NTU/Extended-RaBitQ (archived 2026-03-30)
- Successor: https://github.com/VectorDB-NTU/RaBitQ-Library (active)

**Why it matters:** faiss's `IndexRaBitQ` is the *original* 1-bit RaBitQ. Extended-RaBitQ generalises the estimator to multi-bit codes (3/4/5/7/8/9 bpd) with tight theoretical error bounds, directly addressing the recall ceiling we hit on 1-bit RaBitQ (~0.70 on MSMarco 100K). Without it, "RaBitQ" in our table understates the RaBitQ family by a large margin.

### Integration reality

- **Language:** 99% C++, CMake build, AVX512 required.
- **Python interface:** thin wrapper scripts that subprocess into C++ CLI binaries (`create_index`, `test_search`) reading/writing `.fvecs` files. **Not pip-installable. No pybind11 bindings in tree.**
- **Install path:** `git clone`, `cmake .. && make`, produces binaries in `./bin/`.

### Two integration paths

**Path A — Subprocess adapter (faithful, paper-matched)**
1. Verify AVX512 availability on the `ice-cpu` partition (NOT guaranteed on all HPC nodes). `sbatch` a hello-world that runs `lscpu | grep avx512` on a representative node. If avx512 is absent, abort Path A.
2. Clone RaBitQ-Library into `/storage/ice-shared/cs8903onl/vector_quantization/third_party/rabitq/`.
3. Build the C++ binaries. Resolve third-party deps (see `inc/third/README.md`).
4. Write `src/haag_vq/methods/search/extended_rabitq_index.py`: a `ExtendedRaBitQIndex(BaseSearchIndex)` that:
   - On `fit`: materializes `train` to a tmp `.fvecs`, invokes `create_index [.fvecs] [clusters] [bits]` via subprocess, captures the index dir.
   - On `search`: materializes `queries` to a tmp `.fvecs`, invokes `test_search [.fvecs] [bits]`, parses top-k IDs from stdout or result files.
   - On `save`/`load`: tarball the index dir; faithful roundtrip.
5. Add `ext_rabitq` to `AVAILABLE_METHODS` in both runners.
6. Tests: unit test that guards the subprocess contract (binary exists, writes & parses output) plus an integration test on a small synthetic `.fvecs`.

Estimated effort: 1 working session (~4–6h including build pain), potentially +1 session if AVX512 is unavailable or build breaks.

**Path B — Reimplement in pure Python (Plan B if Path A fails)**
1. Read the paper carefully (the quantiser is a linear rotation + scale-bias factors + per-code estimator with closed-form correction).
2. Implement `ExtendedRaBitQuantizer(BaseQuantizer)` in ~100 lines of numpy (ideally JIT'd with numba or jax).
3. Use the existing `FlatQuantizedIndex` wrapper. Search is estimator-based; reconstruct for `reconstruction_mse`.

Estimated effort: 1 session (~4h) if the paper is clear. Risk: we re-derive, our numbers don't match the paper's published benchmarks, blame is ambiguous.

### Decision criterion

If the faiss 1-bit RaBitQ number (flat or IVF) is already Pareto-dominated by PQ or SAQ on our target use case (embedding retrieval at 8–32× compression), Extended-RaBitQ is the highest-value integration. Otherwise it's a nice-to-have.

From today's numbers: 1-bit RaBitQ@30× gives recall 0.64 (flat) / 0.74 (IVF). SAQ@7.8× gives recall 0.915. At matched compression (30× bucket), PQ and SAQ can't compete (they'd need bpd < 1 which PQ/SAQ don't support), so **RaBitQ family is uncontested at >15× compression** — and Extended-RaBitQ at say 3 bpd (compression ~10.7×) vs SAQ at bpd=2 (15.4×, 0.841) is the interesting head-to-head.

**Verdict:** worth the spike. Priority 1.

### Next action (resume here)

1. `ssh` to an ice-cpu node or `sbatch` `lscpu` → confirm AVX512F/AVX512VNNI flags.
2. If yes: proceed with Path A build.
3. If no: ask whether to try coc-gpu (may have AVX512 on newer SKUs) or fall back to Path B.

---

## 4. Priority 2: ScaNN

**Paper:** Guo et al. 2020, "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (Google).
**Repo:** https://github.com/google-research/google-research/tree/master/scann
**Install:** `pip install scann` — glibc/linux wheels available, usually smooth on HPC.

### Why

ScaNN's anisotropic PQ uses a loss function tuned for inner-product retrieval with optional reranking. It's the method routinely cited as SotA for ANN benchmarks like ann-benchmarks.com at recall >0.90. Without it our "SotA comparison" has a visible hole.

### Integration

- `src/haag_vq/methods/search/scann_index.py`: `ScannIndex(BaseSearchIndex)`.
  - `fit(X, metric)`: construct `scann.scann_ops_pybind.builder(X, k, metric)`, `.tree(...).score_ah(...).reorder(...).build()`.
  - `search(Q, k)`: `.search_batched(Q, final_num_neighbors=k)`.
- Register `scann` in both runners. Memory footprint: ScaNN's `searcher.config()` returns string; parse or compute approximately from code params.
- Tests: fit/search/memory positive; skip if `scann` not installable.

Estimated effort: ~2h (single wrapper class + tests). Lowest-risk SotA add.

### Next action

`pip install scann` in `saq-dev`. Verify basic `searcher.search_batched` works on MSMarco 100K. Then implement the wrapper.

---

## 5. Priority 3: Additive Quantization (AQ) / LSQ

**Paper:** Babenko & Lempitsky 2014 (AQ); Martinez et al. 2016 (LSQ — Local Search Quantizer).
**Availability:** faiss exposes `faiss.ResidualQuantizer`, `faiss.LocalSearchQuantizer`, `faiss.ProductAdditiveQuantizer`. Probably already enough — no external dep.

### Why

AQ and its derivatives (LSQ) consistently outperform PQ at matched bpd on natural embedding distributions (they don't assume subspace independence). Having AQ in the table closes the "is our PQ baseline fair?" question.

### Integration

- Much like PQ: `FlatQuantizedIndex(ResidualQuantizer(...))` via a new `additive_quantization.py` `BaseQuantizer` subclass OR directly instantiate the faiss types if they're already BaseQuantizer-compatible (needs inspection).
- Register `aq_flat`, maybe `lsq_flat`, in both runners.

Estimated effort: ~3h including sanity checks on faiss quantizer API.

### Next action

Probe `faiss.ResidualQuantizer.compute_codes` / `.decode` shape; determine if it fits BaseQuantizer directly.

---

## 6. Priority 4: LVQ (Locally-adaptive Vector Quantization)

**Paper:** Tepper et al. 2023, "LeanVec: Search your vectors faster by making them fit" (Intel). Also LVQ in a line of Intel Labs work.
**Availability:** no faiss implementation; Intel's SVS (Scalable Vector Search) library has a C++ reference (https://github.com/intel/ScalableVectorSearch). Python bindings experimental.

### Why

LVQ's selling point is adaptive per-vector scaling that lifts recall at very low bpd. Included for completeness; may be low-signal if we already have SAQ (which has similar local-scale adaptation).

### Integration

Same shape as Extended-RaBitQ Path A: subprocess adapter against Intel SVS binaries. High integration cost, uncertain pay-off.

### Next action

Read a current SVS release note first; decide whether LVQ@8bpd materially beats PQ+OPQ at the same bpd. If not, cut.

---

## 7. Cross-cutting design questions

These should be settled before the "big compare" re-run so the CSV schema doesn't need a second rewrite.

### 7.1. Flat vs IVF as a column, not a method-name axis

Currently we have `pq_flat`, `faiss_ivfpq` etc. — the index structure is baked into the method name. But for a fair comparison, method × index-structure should be orthogonal:

- `rabitq` + `flat` → current `rabitq`
- `rabitq` + `ivf` → `rabitq_ivf`
- `pq` + `flat` → `pq_flat`
- `pq` + `ivf` → `faiss_ivfpq`
- `saq` + `ivf` → current `saq`
- Extended-RaBitQ × IVF / flat / HNSW (library supports all three)

**Proposal:** add an `index_structure` column to result CSVs (values: `flat`, `ivf`, `hnsw`), keep `method` as the quantizer name only. Downstream analysis filters by both.

**Migration:** one-time rewrite of `ivf_benchmark` / `run_benchmarks` method registries from the current name-mangled form to `(method, index_structure)` tuples. Tests updated. Existing result files have the old schema — treat them as legacy and document the cutover date.

### 7.2. bpd budget fairness

Methods have different "natural" bits:
- PQ: multiples of `nbits_per_sub` (usually 8) across M subquantizers; bpd = M×8/D.
- SQ: fixed bit depths (4, 8, 16).
- SAQ: any avg_bits in [2, 8] (continuous).
- RaBitQ: 1 bpd fixed (original); 3/4/5/7/8/9 (extended).
- ScaNN: any bpd given code params.
- AQ: M×B bits per vector, any M×B.

A single "bpd=4" benchmark isn't apples-to-apples — some methods must round to a different effective bpd. **Proposal:** record `effective_bpd` in the CSV (computed as `code_size_bits / D`) alongside the requested bpd. Plot against effective bpd when reporting.

### 7.3. Fairness on query-side cost

RaBitQ's `qb` (query bits) is an orthogonal axis. ScaNN has `reorder_num_neighbors`. Faiss has `nprobe`. These are search-time knobs that trade recall for QPS. Report recall @ fixed QPS AND QPS @ fixed recall (two different sweep strategies) to avoid hiding behind a single operating point.

### 7.4. Dataset diversity

MSMarco 100K + DBPedia are both Cohere/OpenAI embeddings with similar structure. For a claim about "SotA across embeddings" we probably also want:
- SIFT1M (SIFT descriptors — different distribution)
- GIST1M (natural image features)
- GloVe (text, different dim)

**Proposal:** add these to `datasets/` once the method roster is stable. Not blocking.

---

## 8. 53M scale: get comparable recall/QPS (blocking)

**Status:** currently broken — MSMarco 53M numbers are not directly comparable to the MSMarco 100K table.

**Root cause:** two divergent runners.
- `slurm/bench_msmarco_100k.sh` → `haag_vq.benchmarks.ivf_benchmark` → builds a real search index, measures recall / QPS / memory / MSE.
- `msmarco.sh` (53M) → `haag_vq.benchmarks.streaming_sweep` → never builds an index, only computes streaming reconstruction MSE + compression.

`streaming_sweep` exists because 53M × 1024d × float32 ≈ 217 GB, which doesn't fit in a standard SLURM allocation if you try to load the whole corpus. `ivf_benchmark` doesn't stream — it does `np.load('train.npy')` and expects the full corpus in RAM.

Result: the "SAQ beats Faiss-IVFPQ" story from the 100K run has no 53M analog because the 53M path only produces compression/MSE, not recall/QPS.

### Three paths to fix

**Path 1 (cheapest, ~2h): materialize 53M as `.npy`, memmap it, keep `ivf_benchmark`**
1. One-time: convert the HF parquet at `datasets/msmarco-v2.1-embed-english-v3/` → `train.npy` on shared storage. **Prerequisite:** verify ~220 GB quota headroom. Script: iterate HF `load_dataset(..., streaming=True)`, write a pre-allocated `np.memmap` to `.npy`.
2. Subsample 100–1000 query vectors into `queries.npy`.
3. Patch `ivf_benchmark._load_npy_dataset`: `np.load(train_path, mmap_mode='r')` instead of a full in-RAM load. faiss `index.add()` copies batch-by-batch from the memmap; only the index's working set lives in RAM.
4. Copy `bench_msmarco_100k.sh` → `bench_msmarco_53m.sh`, swap dataset path, bump `--mem` (probably 48–64 G for IVF-based methods; flat indexes would need ~220 G).
5. Run. Results comparable to 100K table.

**Path 2 (unifying, ~1 day): teach `ivf_benchmark` to stream-build**
1. Train coarse quantizer on a 1M sample (already done in streaming pipeline).
2. Stream HF iterator, call `index.add()` batch-by-batch. faiss IVF/IVFRaBitQ/IVFPQ support incremental add. Flat indexes don't — either skip them at 53M or fall back to Path 1 for them only.
3. After stream completes, evaluate queries against the on-disk index. Save via `faiss.write_index`.
4. Deprecate `streaming_sweep.py` (MSE-only) — replaced by the unified harness.

Path 2 is the better long-term shape but not needed to unblock the current Slack claim; Path 1 is sufficient.

**Path 3 (not recommended): keep `streaming_sweep` but add faiss index construction alongside.** Creates a third partially-overlapping runner. Skip.

### Prerequisites for Path 1

- Verify disk quota: `df -h /storage/ice-shared/cs8903onl/` → need ≥ 220 GB free; if tight, consider writing the `.npy` under scratch (`/storage/ice1/0/3/rshah95/scratch/` if that has quota) and symlinking.
- Verify the HF parquet download is complete in the shared datasets dir (earlier `ls` showed `msmarco-v2.1-embed-english-v3/` present; confirm row count matches 53M).
- Decide query provenance. The 100K script synthesises queries from the tail of train. For 53M, either (a) hold out the last 1000 rows from train, or (b) use the actual MS MARCO query set if it's in the HF dataset.

### Known caveats once we run it

- `streaming_sweep.py` itself has two correctness bugs that carry over if we keep using it as a fallback for methods that don't fit Path 1: (i) SAQ branch references the deleted Python scaffold and crashes, (ii) RaBitQ branch uses the old decode-era path, not the native estimator. Both fixed by either deprecating `streaming_sweep` (Path 2) or by routing through `SaqIndex` / `RaBitQIndex` in-place (smaller patch, but keeps the parallel runner alive).
- SLURM `--tmp=100G` is still probably correct for HF cache residue even under Path 1, since HF metadata still writes to cache during any `datasets` preprocessing.

### Next action (resume here)

1. `df -h /storage/ice-shared/cs8903onl/` — confirm quota.
2. Write a small `scripts/materialize_msmarco_53m.py` that streams the HF parquet into a single `.npy` memmap under `datasets/msmarco_53m/train.npy`.
3. Run it (one-time, 1–2h depending on I/O).
4. Patch `ivf_benchmark._load_npy_dataset` for memmap support.
5. Submit `bench_msmarco_53m.sh`.

This is blocking for any headline claim about SAQ/RaBitQ beating SotA at scale — without comparable 53M recall numbers, every number in the Slack update is a 100K claim.

## 9. Execution order (recommended)

1. **Today (Apr 14, already in flight):** Track A — rewrite `RaBitQIndex` to native estimator, add `rabitq_ivf`, promote OPQ.
2. **Blocking follow-up (§8):** materialize 53M as `.npy` + memmap patch + re-run — get real recall/QPS at scale. Until this lands, "SotA comparison" = "100K SotA comparison."
3. **Next session:** ScaNN (cheap, high signal). AQ via faiss residual quantizers (cheap, closes a fairness gap).
4. **Session after:** Extended-RaBitQ Path A (AVX512 probe → build → wrapper). Fall back to Path B if AVX512 missing.
5. **Eventually:** CSV schema refactor (§7.1), effective-bpd reporting (§7.2), dataset diversity (§7.4).
6. **Conditional:** LVQ if the roster still has a gap after (1)–(5).

---

## 10. Deferred questions for later

- What's the target "headline" comparison table? (methods × datasets × compression buckets — how many cells?)
- Who's the audience for these numbers — a paper, an internal doc, an ablation for a talk? Affects reporting rigor.
- Do we want a reproducibility harness (pinned deps, hashes) in-repo, or is the current slurm+commit-sha lineage enough?

---

## 11. Known tech-debt items (tangential but worth noting)

- `_bpd_to_pq_M` in `ivf_benchmark` clamps to divisors of D; bpd=4 and bpd=6 collapse to the same M at D=1024. Doesn't affect SotA integration but confuses the Faiss sweep.
- `SaqIndex.memory_footprint` was fixed on Apr 14 (D-factor restored); prior CSVs pre-fix are invalid for SAQ.
- `run_benchmarks.py` and `ivf_benchmark.py` have overlapping responsibilities. Consider merging or clearly delineating their scopes in a future refactor.
