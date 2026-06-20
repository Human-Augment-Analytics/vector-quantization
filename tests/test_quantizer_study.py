import numpy as np
import pandas as pd

from haag_vq.benchmarks.quantizer_study import run_study_arrays


def test_run_study_arrays_faiss_methods():
    rng = np.random.default_rng(0)
    D = 48
    X = rng.standard_normal((1000, D)).astype(np.float32)
    Q = rng.standard_normal((50, D)).astype(np.float32)

    df = run_study_arrays(
        X, Q,
        methods=["pq", "sq"],
        bpd_values=[4, 8],
        ks=(1, 10),
        chunk_size=256,
        mse_sample=1000,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    for col in ["method", "bpd", "compression_factor", "code_bytes",
                "mse", "recall_at_1", "recall_at_10", "n_db", "n_queries", "D"]:
        assert col in df.columns
    assert df["recall_at_10"].between(0, 1).all()
    assert (df["compression_factor"] > 1).all()
    assert df["mse"].min() >= 0
