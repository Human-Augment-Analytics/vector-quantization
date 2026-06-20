import matplotlib
matplotlib.use("Agg")

import pandas as pd

from haag_vq.benchmarks.study_plots import pareto_curves


def _df():
    return pd.DataFrame([
        {"method": "pq", "bpd": 4, "compression_factor": 8.0,
         "mse": 0.01, "recall_at_1": 0.6, "recall_at_10": 0.7, "recall_at_100": 0.8},
        {"method": "pq", "bpd": 8, "compression_factor": 4.0,
         "mse": 0.005, "recall_at_1": 0.7, "recall_at_10": 0.8, "recall_at_100": 0.9},
        {"method": "sq", "bpd": 4, "compression_factor": 8.0,
         "mse": 0.012, "recall_at_1": 0.55, "recall_at_10": 0.66, "recall_at_100": 0.77},
        {"method": "sq", "bpd": 8, "compression_factor": 4.0,
         "mse": 0.006, "recall_at_1": 0.65, "recall_at_10": 0.78, "recall_at_100": 0.88},
    ])


def test_pareto_curves_writes_file(tmp_path):
    out = tmp_path / "pareto.png"
    pareto_curves(_df(), out, ks=(1, 10, 100))
    assert out.exists() and out.stat().st_size > 0


def test_pareto_curves_single_k(tmp_path):
    # Regression: len(ks)==1 must still render the MSE panel separately.
    out = tmp_path / "pareto_single.png"
    pareto_curves(_df(), out, ks=(10,))
    assert out.exists() and out.stat().st_size > 0
