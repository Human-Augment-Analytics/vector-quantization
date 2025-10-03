# src/haag_vq/data/msmarco_loader.py
from typing import Optional, Iterable
import os, csv, json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    from datasets import load_dataset, get_dataset_config_names
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from .datasets import Dataset

def _read_collection_tsv(path: str, limit: Optional[int] = None) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        count = 0
        for row in reader:
            if len(row) < 2:
                continue
            yield row[1]
            count += 1
            if limit is not None and count >= limit:
                break

def load_msmarco_passages_from_local(
    collection_tsv: str,
    limit: Optional[int] = 100_000,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dataset:
    assert os.path.exists(collection_tsv), f"Not found: {collection_tsv}"
    sentences = list(_read_collection_tsv(collection_tsv, limit=limit))
    model = SentenceTransformer(model_name)
    vectors = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True, batch_size=256)
    return Dataset(vectors=vectors, queries=vectors[:100], ground_truth=np.empty((0,)), distance_metric=None)  # type: ignore

def load_msmarco_passages_from_hf(
    hf_dataset: str = "ms_marco",
    hf_config: Optional[str] = None,
    hf_split: str = "train",
    text_column: Optional[str] = None,
    limit: Optional[int] = 100_000,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dataset:
    if not HF_AVAILABLE:
        raise RuntimeError("Hugging Face 'datasets' not installed. pip install datasets")

    if hf_config is None:
        try:
            cfgs = get_dataset_config_names(hf_dataset)
            if cfgs:
                hf_config = cfgs[0]
        except Exception:
            pass

    ds = load_dataset(hf_dataset, hf_config, split=hf_split)

    candidates = [text_column] if text_column else ["passage", "text", "body", "document", "content"]
    col = None
    for c in candidates:
        if c and c in ds.column_names:
            col = c
            break
    if col is None:
        raise ValueError(f"Could not find a text column. Columns={ds.column_names}. "
                         f"Pass text_column=... explicitly.")

    n = len(ds) if limit is None else min(limit, len(ds))
    sentences = [ds[i][col] for i in range(n)]

    model = SentenceTransformer(model_name)
    vectors = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True, batch_size=256)
    return Dataset(vectors=vectors, queries=vectors[:100], ground_truth=np.empty((0,)), distance_metric=None)  # type: ignore

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="MS MARCO loader -> Dataset")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--collection-tsv", type=str, help="Path to local MS MARCO collection.tsv")
    src.add_argument("--hf", action="store_true", help="Load via Hugging Face datasets")

    p.add_argument("--limit", type=int, default=100000, help="Max passages to embed (memory/time guard)")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    p.add_argument("--hf-dataset", type=str, default="ms_marco")
    p.add_argument("--hf-config", type=str, default=None)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--text-column", type=str, default=None)
    p.add_argument("--out", type=str, default="data/msmarco-mini.npz", help="Where to save vectors")

    args = p.parse_args()

    if args.collection_tsv:
        ds = load_msmarco_passages_from_local(args.collection_tsv, limit=args.limit, model_name=args.model)
    else:
        ds = load_msmarco_passages_from_hf(
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            text_column=args.text_column,
            limit=args.limit,
            model_name=args.model,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        vectors=ds.vectors,
        queries=ds.queries,
    )
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "limit": args.limit}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.out} and {args.out}.meta.json")
