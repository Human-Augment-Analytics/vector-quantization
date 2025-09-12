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

## 🗂 Repository Structure

_TBD: As code and benchmark notebooks are added, include directory structure and setup instructions._

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
