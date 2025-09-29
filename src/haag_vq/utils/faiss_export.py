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
    """Read ``.fvecs`` file into ``float32`` matrix."""
    return _load_vec_file(path, np.float32)


def load_ivecs(path: PathLike) -> np.ndarray:
    """Read ``.ivecs`` file into ``int32`` matrix."""
    return _load_vec_file(path, np.int32)


def _default_index_key(model: BaseQuantizer) -> str:
    if isinstance(model, ProductQuantizer):
        bits = int(round(math.log2(model.num_clusters)))
        return f"PQ{model.num_chunks}x{bits}"  # e.g. PQ8x8
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
        return model.chunk_dim * model.num_chunks
    if isinstance(model, ScalarQuantizer):
        if model.min is None:
            raise ValueError("ScalarQuantizer has no min; call fit first")
        return int(model.min.shape[0])
    raise TypeError(f"Unsupported quantizer type: {type(model)!r}")


def build_faiss_index(
    model: BaseQuantizer,
    *,
    index_key: Optional[str] = None,
    metric: int = faiss.METRIC_L2,
) -> faiss.Index:
    """Create a FAISS index for a quantizer using ``index_factory``."""
    dim = _infer_dimensionality(model)
    factory_key = index_key or _default_index_key(model)
    return faiss.index_factory(dim, factory_key, metric)


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

    index = build_faiss_index(model, index_key=index_key)

    result: dict[str, Union[Path, faiss.Index, np.ndarray]] = {
        "codebook": codebook_path,
        "index": index,
        "codebook_vectors": codebook,
    }

    if codes is not None:
        codes_matrix = np.asarray(codes, dtype=np.int32)
        codes_path = write_ivecs(output_path / codes_filename, codes_matrix)
        result["codes"] = codes_path

    return result


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
    """Search the exported codebook for the closest entries to ``queries``.

    Args:
        queries: Query vectors shaped ``(N, D)`` or ``(D,)``.
        model: Optional trained quantizer. If provided, its configuration is used
            to instantiate the FAISS index via :func:`build_faiss_index`.
        codebook_vectors: Preloaded codebook vectors (``float32`` matrix).
        codebook_path: Path to ``.fvecs`` file storing the codebook. Used when
            ``codebook_vectors`` is not supplied.
        topk: Number of nearest codebook entries to retrieve.
        metric: FAISS distance metric constant (defaults to ``METRIC_L2``).
        index_key: Optional override for FAISS factory key when ``model`` is set.

    Returns:
        Tuple ``(distances, indices)`` from ``faiss.Index.search``.
    """
    if codebook_vectors is None:
        if codebook_path is None:
            raise ValueError("Provide either codebook_vectors or codebook_path")
        codebook_vectors = load_fvecs(codebook_path)
    codebook_vectors = np.asarray(codebook_vectors, dtype=np.float32)

    if codebook_vectors.ndim != 2:
        raise ValueError("Codebook vectors must be 2D")

    num_entries, dim = codebook_vectors.shape
    if num_entries == 0:
        raise ValueError("Codebook is empty; cannot run queries")

    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2:
        raise ValueError("Queries must be a 1D or 2D array")
    if queries.shape[1] != dim:
        raise ValueError(
            f"Query dimensionality ({queries.shape[1]}) does not match codebook ({dim})"
        )

    if topk <= 0:
        raise ValueError("topk must be positive")
    topk = min(topk, num_entries)

    if model is not None:
        index = build_faiss_index(model, index_key=index_key, metric=metric)
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
