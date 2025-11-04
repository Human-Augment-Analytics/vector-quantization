# HAAG Vector Quantization Project

This repository supports **HAAG research efforts** in developing new **vector quantization and compression methods**, and benchmarking them against existing open-source and private-industry methods.

---

## 📌 Project Overview

This work is part of the HAAG "Methods Unit", with a focus on:

- **Literature review** of quantization/compression techniques
- **Implementation** of selected quantization methods
- **Benchmarking** using open-source frameworks and commercial comparators
- **Weekly presentations and discussions** as part of collaborative research

---

## 🔗 Quick Links

### HAAG Resources

- 👥 **[Enrollment Roster](https://gtvault.sharepoint.com/:x:/s/HAAG/EbRWUBbmh3pPpGuh9HF34DgBPnJQEdtMQoBTtANXCxOg9Q?e=B8ykCV)**
- 📄 **[Weekly Report Template](https://gtvault.sharepoint.com/:w:/s/HAAG/EcKDOtAbNKZEr3KrZDfRlZ4BD_IMA-4hTSc7ll52J6_79A)**
- 🧠 **[Explain Paper](https://www.explainpaper.com/)** – great tool for understanding complex papers.

### Methods Unit Resources

- 🎤 **[Presentation Sign-Up Sheet](https://gtvault-my.sharepoint.com/:x:/g/personal/byu321_gatech_edu/EUB3IKLuDwdLkG5dlPwJoccByYUJ9XJgcngZMbOa8pwq0A)**

---

## 📚 Literature & Learning

### Vector Quantization Project Resources

- 📝 [**Project Doc**](https://gtvault-my.sharepoint.com/:w:/r/personal/smussmann3_gatech_edu/_layouts/15/Doc.aspx?sourcedoc=%7B805CAAA2-48BB-42CD-A20D-C04F2DA3CA41%7D&file=Vector_Quantization_project.docx&action=default&mobileredirect=true&DefaultItemOpen=1)
- 📚 **Literature Review** – _[link to be added]_  
- **Key PQ Variants (Faiss)**:  
  ![PQ Variants Chart](https://raw.githubusercontent.com/wiki/facebookresearch/faiss/PQ_variants_Faiss_annotated.png)
- **8-bit Rotational Quantization (Weaviate Blog)**  
  https://weaviate.io/blog/8-bit-rotational-quantization

### General Learning Resources

- 📺 [Vector Quantization – YouTube Intro](https://www.youtube.com/watch?v=c36lUUr864M)

---

## 🧰 Open Source Ecosystem

- **[pgvector (Postgres extension)](https://github.com/pgvector/pgvector)**
- **[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss/wiki)**

---

## 💼 Private Industry References

- [Qdrant – What is Vector Quantization?](https://qdrant.tech/articles/what-is-vector-quantization/)
- [Weaviate Blog](https://weaviate.io/blog)
- [Pinecone Research](https://www.pinecone.io/research/)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd vector-quantization

# Install in editable mode
pip install -e .
```

### Load a Dataset

```python
from haag_vq.data import load_cohere_msmarco_passages

# Load pre-embedded MS MARCO dataset
ds = load_cohere_msmarco_passages(
    limit=100_000,
    cache_dir="../datasets",
)
```

See [DATASETS.md](documentation/DATASETS.md) for all available datasets.

### Run a Benchmark

```bash
# Simple benchmark
vq-benchmark run --dataset dummy --method pq

# With recall metrics
vq-benchmark run \
    --dataset dummy \
    --num-samples 10000 \
    --method pq \
    --num-chunks 8 \
    --num-clusters 256 \
    --with-recall
```

### Run a Parameter Sweep

```bash
vq-benchmark sweep \
    --method pq \
    --dataset dummy \
    --pq-chunks "4,8,16" \
    --pq-clusters "128,256,512"
```

### Visualize Results

```bash
vq-benchmark plot
```

---

## 📖 Documentation

### User Guides
- **[USAGE.md](documentation/USAGE.md)** - Complete CLI reference
- **[METHODS.md](documentation/METHODS.md)** - Quantization methods overview
- **[DATASETS.md](documentation/DATASETS.md)** - Dataset loading guide
- **[SLURM_GUIDE.md](documentation/SLURM_GUIDE.md)** - ICE cluster guide
- **[SLURM_SUPPORT_PLAN.md](documentation/SLURM_SUPPORT_PLAN.md)** - Production readiness

### Developer Guides
- **[ADDING_NEW_METHODS.md](documentation/ADDING_NEW_METHODS.md)** - How to add quantization methods
- **[METRICS_GUIDE.md](documentation/METRICS_GUIDE.md)** - Understanding metrics
- **[PARAMETER_SWEEP_GUIDE.md](documentation/PARAMETER_SWEEP_GUIDE.md)** - Parameter sweep details

---

## 🔧 Implemented Methods

The project implements **5 vector quantization methods**:

| Method | Type | Compression Ratio | Reference |
|--------|------|-------------------|-----------|
| **PQ** | Subspace Quantization | 32-512:1 | [Jégou et al., 2011](https://hal.inria.fr/inria-00514462v2/document) |
| **OPQ** | Optimized Subspace | 32-512:1 | [Ge et al., 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/11/pami13opq.pdf) |
| **SQ** | Scalar Quantization | 4:1 | Standard |
| **SAQ** | Segmented CAQ | 8-32:1 | [Zhou et al., 2024](https://arxiv.org/abs/2410.06482) |
| **RaBitQ** | Bit-level | 100:1+ | [Gao & Long, 2024](https://dl.acm.org/doi/pdf/10.1145/3654970) |

See [METHODS.md](documentation/METHODS.md) for detailed descriptions, parameters, and usage examples.

---

## 📊 Available Datasets

The project includes loaders for several large-scale pre-embedded datasets:

| Dataset | Size | Dimensions | Source |
|---------|------|------------|--------|
| **Cohere MS MARCO v2.1** | ~53.2M passages | ~1024 | [HuggingFace](https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3) |
| **DBpedia OpenAI 1536 (100K)** | 100K entities | 1536 | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-100K) |
| **DBpedia OpenAI 1536 (1M)** | 1M entities | 1536 | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) |
| **DBpedia OpenAI 3072** | 1M entities | 3072 | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) |

All datasets are **pre-embedded** (no embedding computation required) and automatically downloaded from Hugging Face.

See [DATASETS.md](documentation/DATASETS.md) for detailed usage instructions.

---

## 🖥️ CLI Commands

```bash
vq-benchmark run           # Run single benchmark
vq-benchmark sweep         # Run parameter sweep
vq-benchmark precompute-gt # Precompute ground truth
vq-benchmark plot          # Visualize results
```

---

## 🏔️ Running on PACE/ICE Cluster

**Production-ready with automatic optimizations:**

### Quick Start (5 minutes)
```bash
# 1. SSH and navigate
ssh <username>@login-ice.pace.gatech.edu
cd /storage/ice-shared/cs8903onl/vector_quantization/vector-quantization

# 2. Setup (first time only)
module load python/3.12
source .venv312/bin/activate
mkdir -p logs results

# 3. Run sweep
./sweep dbpedia-100k  # ~2 hours, 100K vectors, all methods

# 4. Monitor
squeue -u $USER
tail -f logs/dbp100k_*.out
```

**📚 Complete Guides:**
- **[docs/QUICKSTART_PACE.md](docs/QUICKSTART_PACE.md)** - Get running in 5 minutes
- **[docs/RUNNING_SWEEPS.md](docs/RUNNING_SWEEPS.md)** - Production sweep guide
- **[docs/RUNNING_ON_PACE.md](docs/RUNNING_ON_PACE.md)** - Detailed PACE/ICE documentation
- **[docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md)** - File organization

### Available Sweeps

| Command | Dataset | Vectors | Time | Resources |
|---------|---------|---------|------|-----------|
| `./sweep dbpedia-100k` | DBpedia 100K | 100K | 2 hrs | 4GB RAM |
| `./sweep dbpedia-1m-subset` | DBpedia 1M | 500K | 4 hrs | 8GB RAM |
| `./sweep dbpedia-1m-full` | DBpedia 1M | 1M | 8 hrs | 12GB RAM |
| `./sweep custom <args>` | Custom | Variable | Variable | Variable |

### Key Features
- ✅ Configurable via CLI: `python scripts/run_sweep.py --dataset <name> --methods <list>`
- ✅ Automatic use of `$TMPDIR` for fast local NVMe storage
- ✅ Memory-efficient pre-allocated arrays (~50% reduction)
- ✅ FAISS ground truth computation (>99% memory savings)
- ✅ Dimension-aware parameter generation
- ✅ Results logged to SQLite database

---

## 🗂 Repository Structure

The project is organized into modular components for benchmarking, methods, datasets, and metrics.

```
haag-vector-quantization
├── LICENSE
├── README.md
├── __init__.py
├── logs
│   └── benchmark_runs.db
├── notebooks
├── pyproject.toml
├── requirements.txt
├── src
│   ├── __init__.py
│   └── haag_vq
│       ├── benchmarks
│       │   ├── __init__.py
│       │   └── run_benchmarks.py
│       ├── cli.py
│       ├── data
│       │   ├── __init__.py
│       │   └── datasets.py
│       ├── methods
│       │   ├── __init__.py
│       │   ├── base_quantizer.py
│       │   ├── product_quantization.py
│       │   └── scalar_quantization.py
│       ├── metrics
│       │   ├── __init__.py
│       │   ├── distortion.py
│       │   └── recall.py
│       └── utils
│           ├── __init__.py
│           └── run_logger.py
└── tests
    └── __init__.py

11 directories, 22 files
```

---

## ✍️ Contributions

This project is part of the HAAG research cohort. Contributions should align with group goals and follow any coordination protocols discussed during meetings. Please submit questions or ideas via Slack or your weekly reports.

---

## 📅 Weekly Work Expectations Resources

- Submit your **weekly update** using the [template](https://gtvault.sharepoint.com/:w:/s/HAAG/EcKDOtAbNKZEr3KrZDfRlZ4BD_IMA-4hTSc7ll52J6_79A)
- Sign up to present a method or paper on the [presentation sheet](https://gtvault-my.sharepoint.com/:x:/g/personal/byu321_gatech_edu/EUB3IKLuDwdLkG5dlPwJoccByYUJ9XJgcngZMbOa8pwq0A)

---

## 📧 Contact

Please post in the `vector-quantization` Slack channel with any questions or onboarding needs.

---
