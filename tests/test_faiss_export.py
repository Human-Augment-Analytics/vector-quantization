import numpy as np
import faiss

from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.utils.faiss_export import export_codebook, query_codebook


def test_query_codebook_with_model(tmp_path):
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    quantizer = ScalarQuantizer()
    quantizer.fit(data)

    export_result = export_codebook(quantizer, tmp_path)
    codebook_vectors = export_result["codebook_vectors"]

    queries = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)
    distances, indices = query_codebook(
        queries,
        model=quantizer,
        codebook_vectors=codebook_vectors,
        topk=1,
    )

    assert distances.shape == (1, 1)
    assert indices.shape == (1, 1)
    assert indices[0, 0] == 1  # the max vector is closest to the query


def test_query_codebook_from_disk(tmp_path):
    data = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    quantizer = ScalarQuantizer()
    quantizer.fit(data)

    export_result = export_codebook(quantizer, tmp_path, codebook_filename="cb.fvecs")
    codebook_path = export_result["codebook"]

    queries = np.array([[0.1, 0.1], [1.9, 1.9]], dtype=np.float32)
    distances, indices = query_codebook(
        queries,
        codebook_path=codebook_path,
        topk=2,
    )

    assert distances.shape == (2, 2)
    assert indices.shape == (2, 2)
    # First query is closer to the min vector (index 0)
    assert indices[0, 0] == 0
    # Second query is closer to the max vector (index 1)
    assert indices[1, 0] == 1


def test_export_codebook_uses_ivf(tmp_path):
    data = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    quantizer = ScalarQuantizer()
    quantizer.fit(data)

    export_result = export_codebook(quantizer, tmp_path)
    index = export_result["index"]

    ivf_base = getattr(faiss, "IndexIVF")
    assert isinstance(index, ivf_base)
    assert index.nlist >= 1
    assert 1 <= index.nprobe <= index.nlist
