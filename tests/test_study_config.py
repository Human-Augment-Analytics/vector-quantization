import textwrap
import numpy as np

from haag_vq.benchmarks.study_config import StudyConfig, load_study_config


def test_load_study_config(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent("""
        dataset:
          name: toy
          base_fvecs: /tmp/base.fvecs
          query_fvecs: /tmp/query.fvecs
          n_queries: 1000
        methods: [pq, sq]
        bpd: [1, 2, 4, 8]
        ks: [1, 10, 100]
        chunk_size: 50000
        mse_sample: 100000
        output_dir: results/dev
    """))
    cfg = load_study_config(p)
    assert isinstance(cfg, StudyConfig)
    assert cfg.methods == ["pq", "sq"]
    assert cfg.bpd == [1, 2, 4, 8]
    assert cfg.ks == [1, 10, 100]
    assert cfg.dataset["n_queries"] == 1000
    assert cfg.chunk_size == 50000
