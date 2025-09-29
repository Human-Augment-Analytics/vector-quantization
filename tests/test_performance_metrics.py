import numpy as np

from haag_vq.methods.scalar_quantization import ScalarQuantizer
from haag_vq.metrics.performance import measure_qps, time_compress, time_decompress
from haag_vq.utils.faiss_export import export_codebook


def test_compression_and_decompression_latency():
    X = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.5, 0.6, 0.7],
            [1.0, 0.8, 0.6],
        ],
        dtype=np.float32,
    )
    quantizer = ScalarQuantizer()
    quantizer.fit(X)

    codes, compress_time = time_compress(quantizer, X)
    assert codes.shape == X.shape
    assert compress_time >= 0.0
    print("\nCompress Time: ", compress_time)

    recon, decompress_time = time_decompress(quantizer, codes)
    assert recon.shape == X.shape
    assert decompress_time >= 0.0
    print("Decompress Time: ", decompress_time)


def test_measure_qps(tmp_path):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    quantizer = ScalarQuantizer()
    quantizer.fit(X)

    export_result = export_codebook(quantizer, tmp_path)
    queries = np.array([[0.1, 0.2], [1.9, 2.1]], dtype=np.float32)

    qps_metrics = measure_qps(
        queries,
        model=quantizer,
        codebook_vectors=export_result["codebook_vectors"],
        repeats=2,
    )

    assert qps_metrics["qps"] > 0
    assert qps_metrics["avg_query_latency_ms"] >= 0
    print("qps: ", qps_metrics["qps"])
    print("avg_query_latency_ms: ", qps_metrics["avg_query_latency_ms"])
