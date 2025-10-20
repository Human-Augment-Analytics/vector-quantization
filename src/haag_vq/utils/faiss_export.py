"""FAISS export and query helpers for Haag VQ quantizers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - import guard
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("FAISS is required to export codebooks. Install faiss-cpu or faiss-gpu.") from exc

from haag_vq.methods.base_quantizer import BaseQuantizer
from haag_vq.methods.product_quantization import ProductQuantizer
from haag_vq.methods.scalar_quantization import ScalarQuantizer

PathLike = Union[str, Path]

def write_fvecs(path: PathLike, vectors: np.ndarray) -> Path:
    """Write float vectors to a .fvecs file (FAISS binary format)."""
    path = Path(path)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("fvecs expects a 2D array")

    with path.open("wb") as fh:
        dim_prefix = np.int32(vectors.shape[1])
        for vec in vectors:
            fh.write(dim_prefix.tobytes())
            fh.write(np.asarray(vec, dtype=np.float32).tobytes())
    return path


def write_ivecs(path: PathLike, vectors: np.ndarray) -> Path:
    """Write int vectors to a .ivecs file (FAISS binary format)."""
    path = Path(path)
    vectors = np.asarray(vectors, dtype=np.int32)
    if vectors.ndim != 2:
        raise ValueError("ivecs expects a 2D array")

    with path.open("wb") as fh:
        dim_prefix = np.int32(vectors.shape[1])
        for vec in vectors:
            fh.write(dim_prefix.tobytes())
            fh.write(np.asarray(vec, dtype=np.int32).tobytes())
    return path


def _load_vec_file(path: PathLike, value_dtype: np.dtype) -> np.ndarray:
    """Load FAISS vector files (.fvecs/.ivecs)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = path.read_bytes()
    if not raw:
        return np.empty((0, 0), dtype=value_dtype)

    ints = np.frombuffer(raw, dtype=np.int32)
    dim = int(ints[0])
    record = dim + 1
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension ({dim}) in {path}")
    if ints.size % record != 0:
        raise ValueError(f"Corrupt vector file: {path}")

    nvecs = ints.size // record
    dims = ints.reshape(nvecs, record)[:, 0]
    if not np.all(dims == dim):
        raise ValueError(f"Non-uniform dimensions in {path}")

    if value_dtype == np.int32:
        data = ints.reshape(nvecs, record)[:, 1:]
    elif value_dtype == np.float32:
        floats = np.frombuffer(raw, dtype=np.float32).reshape(nvecs, record)
        data = floats[:, 1:]
    else:  # pragma: no cover - defensive branch
        raise TypeError(f"Unsupported dtype: {value_dtype}")

    return np.array(data, copy=True)


def load_fvecs(path: PathLike) -> np.ndarray:
    """Read `.fvecs` file into `float32` matrix."""
    return _load_vec_file(path, np.float32)


def load_ivecs(path: PathLike) -> np.ndarray:
    """Read `.ivecs` file into `int32` matrix."""
    return _load_vec_file(path, np.int32)


def _default_index_key(model: BaseQuantizer) -> str:
    if isinstance(model, ProductQuantizer):
        # For PQ, the canonical key is PQ{M}x{B}
        bits = int(getattr(model, "B", int(round(math.log2(getattr(model, "num_clusters"))))) )
        M = int(getattr(model, "M", getattr(model, "num_chunks")))
        return f"PQ{M}x{bits}"  # e.g. PQ8x8
    if isinstance(model, ScalarQuantizer):
        return "SQ8"  # 8-bit scalar quantizer
    raise TypeError(f"Unsupported quantizer type: {type(model)!r}")


def _extract_codebook(model: BaseQuantizer) -> np.ndarray:
    """Return a 2D array of codebook vectors suitable for fvec export."""
    if isinstance(model, ProductQuantizer):
        if not model.codebooks:
            raise ValueError("ProductQuantizer has no trained codebooks")
        chunks = [np.asarray(cb, dtype=np.float32) for cb in model.codebooks]
        return np.concatenate(chunks, axis=0)
    if isinstance(model, ScalarQuantizer):
        if model.min is None or model.max is None:
            raise ValueError("ScalarQuantizer must be fitted before export")
        return np.stack([model.min, model.max]).astype(np.float32)
    raise TypeError(f"Unsupported quantizer type: {type(model)!r}")


def _infer_dimensionality(model: BaseQuantizer) -> int:
    if isinstance(model, ProductQuantizer):
        if model.chunk_dim is None:
            raise ValueError("ProductQuantizer has no chunk_dim; call fit first")
        M = int(getattr(model, "M", getattr(model, "num_chunks")))
        return int(model.chunk_dim) * M
    if isinstance(model, ScalarQuantizer):
        if model.min is None:
            raise ValueError("ScalarQuantizer has no min; call fit first")
        return int(model.min.shape[0])
    raise TypeError(f"Unsupported quantizer type: {type(model)!r}")


def _normalize_training_vectors(training_vectors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Ensure training vectors are 2D when provided."""
    if training_vectors is None:
        return None
    arr = np.asarray(training_vectors)
    if arr.ndim != 2:
        raise ValueError("training_vectors must be a 2D array")
    return arr


def _estimate_codebook_size(model: BaseQuantizer) -> Optional[int]:
    """Best-effort estimate of codebook cardinality when no training data is supplied."""
    if isinstance(model, ProductQuantizer):
        M = int(getattr(model, "M", getattr(model, "num_chunks")))
        B = int(getattr(model, "B", int(round(math.log2(getattr(model, "num_clusters"))))) )
        return M * (2 ** B)
    if isinstance(model, ScalarQuantizer):
        if model.min is None or model.max is None:
            return None
        return 2
    return None


def _suggest_ivf_nlist(model: BaseQuantizer, training_vectors: Optional[np.ndarray]) -> int:
    """Heuristic for selecting the number of IVF lists."""
    training_count: Optional[int] = None
    if training_vectors is not None:
        training_count = int(training_vectors.shape[0])
    if training_count is None or training_count <= 0:
        training_count = _estimate_codebook_size(model)
    if training_count is None or training_count <= 0:
        return 1
    max_lists = max(1, min(training_count, 65536))
    guess = int(round(math.sqrt(training_count)))
    guess = max(1, guess)
    return min(max_lists, guess)


def _suggest_ivf_nprobe(nlist: int) -> int:
    """Heuristic for setting `nprobe` when using IVF indexes."""
    if nlist <= 0:
        return 1
    guess = int(round(math.sqrt(nlist)))
    return max(1, min(nlist, guess))


def build_faiss_index(
    model: BaseQuantizer,
    *,
    index_key: Optional[str] = None,
    metric: int = faiss.METRIC_L2,
    training_vectors: Optional[np.ndarray] = None,
    use_ivf: bool = True,
    nlist: Optional[int] = None,
    nprobe: Optional[int] = None,
) -> faiss.Index:
    """Create a FAISS index for a quantizer, optionally wrapping it with IVF.

    Args:
        model: Trained quantizer providing dimensionality details.
        index_key: Optional explicit FAISS factory key. When supplied, the key is
            used as-is and IVF heuristics are skipped.
        metric: FAISS distance metric constant, defaults to `METRIC_L2`.
        training_vectors: Optional matrix used to derive IVF parameters (e.g.,
            `nlist`). When omitted, heuristics fall back to model metadata.
        use_ivf: Whether to prepend an IVF coarse quantizer to the factory key.
        nlist: Explicit override for the number of IVF lists. Must be positive.
        nprobe: Optional override for `index.nprobe` when building IVF indexes.

    Returns:
        A FAISS index instance configured for the provided quantizer.
    """
    dim = _infer_dimensionality(model)
    training_array = _normalize_training_vectors(training_vectors)

    factory_key = index_key or _default_index_key(model)

    if use_ivf and index_key is None and not factory_key.upper().startswith("IVF"):
        if nlist is None:
            inferred_nlist = _suggest_ivf_nlist(model, training_array)
        else:
            if nlist <= 0:
                raise ValueError("nlist must be positive when provided")
            inferred_nlist = int(nlist)
        if training_array is not None and training_array.shape[0] > 0:
            inferred_nlist = min(inferred_nlist, int(training_array.shape[0]))
        inferred_nlist = max(1, inferred_nlist)
        factory_key = f"IVF{inferred_nlist},{factory_key}"

    index = faiss.index_factory(dim, factory_key, metric)

    ivf_base = getattr(faiss, "IndexIVF", None)
    if ivf_base is not None and isinstance(index, ivf_base):
        if nprobe is None:
            inferred_nprobe = _suggest_ivf_nprobe(index.nlist)
        else:
            if nprobe <= 0:
                raise ValueError("nprobe must be positive when provided")
            inferred_nprobe = int(nprobe)
        index.nprobe = max(1, min(index.nlist, inferred_nprobe))

    return index


def export_codebook(
    model: BaseQuantizer,
    output_dir: PathLike,
    *,
    index_key: Optional[str] = None,
    codes: Optional[np.ndarray] = None,
    codebook_filename: str = "codebook.fvecs",
    codes_filename: str = "codes.ivecs",
) -> dict[str, Union[Path, faiss.Index, np.ndarray]]:
    """Persist a quantizer's codebook (and optional codes) to FAISS vector files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    codebook = _extract_codebook(model)
    codebook_path = write_fvecs(output_path / codebook_filename, codebook)

    index: Optional[faiss.Index] = None
    if not isinstance(model, ProductQuantizer):
        index = build_faiss_index(
            model,
            index_key=index_key,
            training_vectors=codebook,
        )

    result: dict[str, Union[Path, faiss.Index, np.ndarray]] = {
        "codebook": codebook_path,
        "codebook_vectors": codebook,
    }
    if index is not None:
        result["index"] = index

    if codes is not None:
        codes_matrix = np.asarray(codes, dtype=np.int32)
        codes_path = write_ivecs(output_path / codes_filename, codes_matrix)
        result["codes"] = codes_path

    return result




def _query_product_codebook(
    queries: np.ndarray,
    model: ProductQuantizer,
    codebook_vectors: np.ndarray,
    topk: int,
    metric: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS search tailored for ProductQuantizer codebooks."""
    if model.chunk_dim is None:
        raise ValueError("ProductQuantizer has no chunk_dim; call fit first")
    chunk_dim = int(model.chunk_dim)
    num_chunks = int(getattr(model, "M", getattr(model, "num_chunks")))
    num_clusters = int(2 ** int(getattr(model, "B", int(round(math.log2(getattr(model, "num_clusters"))))) ))
    expected_rows = num_chunks * num_clusters
    if codebook_vectors.shape != (expected_rows, chunk_dim):
        raise ValueError(
            "ProductQuantizer codebook has unexpected shape; expected "
            f"({expected_rows}, {chunk_dim}) but received {codebook_vectors.shape}"
        )
    if queries.shape[1] != chunk_dim * num_chunks:
        raise ValueError(
            "Query dimensionality does not match ProductQuantizer. "
            f"Expected {chunk_dim * num_chunks} but received {queries.shape[1]}"
        )
    per_chunk_topk = min(topk, num_clusters)
    if per_chunk_topk <= 0:
        raise ValueError("topk must be positive and <= number of clusters")
    query_chunks = queries.reshape(queries.shape[0], num_chunks, chunk_dim)
    codebook_chunks = codebook_vectors.reshape(num_chunks, num_clusters, chunk_dim)
    if metric == faiss.METRIC_L2:
        index_factory = faiss.IndexFlatL2
    elif metric == faiss.METRIC_INNER_PRODUCT:
        index_factory = faiss.IndexFlatIP
    else:
        raise ValueError(
            "ProductQuantizer queries currently support only METRIC_L2 "
            "and METRIC_INNER_PRODUCT"
        )
    distance_chunks = []
    index_chunks = []
    for chunk_idx in range(num_chunks):
        chunk_vectors = np.ascontiguousarray(codebook_chunks[chunk_idx])
        if chunk_vectors.size == 0:
            raise ValueError("ProductQuantizer codebook chunk is empty")
        index = index_factory(chunk_dim)
        index.add(chunk_vectors)
        chunk_queries = np.ascontiguousarray(query_chunks[:, chunk_idx, :])
        d, idx = index.search(chunk_queries, per_chunk_topk)
        distance_chunks.append(d)
        index_chunks.append(idx + chunk_idx * num_clusters)
    distances = np.concatenate(distance_chunks, axis=1)
    indices = np.concatenate(index_chunks, axis=1)
    return distances, indices


def query_codebook(
    queries: np.ndarray,
    *,
    model: Optional[BaseQuantizer] = None,
    codebook_vectors: Optional[np.ndarray] = None,
    codebook_path: Optional[PathLike] = None,
    topk: int = 1,
    metric: int = faiss.METRIC_L2,
    index_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search the exported codebook for the closest entries to `queries`.

    Args:
        queries: Query vectors shaped `(N, D)` or `(D,)`.
        model: Optional trained quantizer. If provided, its configuration is used
            to instantiate the FAISS index via :func:`build_faiss_index`.
        codebook_vectors: Preloaded codebook vectors (`float32` matrix).
        codebook_path: Path to `.fvecs` file storing the codebook. Used when
            `codebook_vectors` is not supplied.
        topk: Number of nearest codebook entries to retrieve. For
            :class:`ProductQuantizer` models this value is applied per chunk and
            the returned arrays are flattened across chunks.
        metric: FAISS distance metric constant (defaults to `METRIC_L2`).
        index_key: Optional override for FAISS factory key when `model` is set.

    Returns:
        Tuple `(distances, indices)` from the FAISS search routine.
    """
    if codebook_vectors is None:
        if codebook_path is None:
            raise ValueError("Provide either codebook_vectors or codebook_path")
        codebook_vectors = load_fvecs(codebook_path)
    codebook_vectors = np.ascontiguousarray(codebook_vectors, dtype=np.float32)

    if codebook_vectors.ndim != 2:
        raise ValueError("Codebook vectors must be 2D")

    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2:
        raise ValueError("Queries must be a 1D or 2D array")
    if queries.size == 0:
        raise ValueError("No queries provided for search")

    if topk <= 0:
        raise ValueError("topk must be positive")
    topk = int(topk)

    if isinstance(model, ProductQuantizer):
        return _query_product_codebook(
            queries,
            model,
            codebook_vectors,
            topk,
            metric,
        )

    num_entries, dim = codebook_vectors.shape
    if num_entries == 0:
        raise ValueError("Codebook is empty; cannot run queries")

    if queries.shape[1] != dim:
        raise ValueError(
            f"Query dimensionality ({queries.shape[1]}) does not match codebook ({dim})"
        )

    topk = min(topk, num_entries)

    if model is not None:
        index = build_faiss_index(
            model,
            index_key=index_key,
            metric=metric,
            training_vectors=codebook_vectors,
        )
    else:
        if metric == faiss.METRIC_L2:
            index = faiss.IndexFlatL2(dim)
        elif metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError("Provide a quantizer model to build complex FAISS indexes")

    if not index.is_trained:
        index.train(codebook_vectors)
    index.add(codebook_vectors)

    distances, indices = index.search(queries, topk)
    return distances, indices
