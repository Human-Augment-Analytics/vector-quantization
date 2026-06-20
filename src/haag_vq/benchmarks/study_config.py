# src/haag_vq/benchmarks/study_config.py
"""Typed config for the quantizer benchmark study (loaded from YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class StudyConfig:
    dataset: Dict[str, Any]
    methods: List[str]
    bpd: List[float]
    ks: List[int] = field(default_factory=lambda: [1, 10, 100])
    chunk_size: int = 50_000
    mse_sample: int = 100_000
    output_dir: str = "results"


def load_study_config(path: str | Path) -> StudyConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return StudyConfig(
        dataset=raw["dataset"],
        methods=list(raw["methods"]),
        bpd=list(raw["bpd"]),
        ks=list(raw.get("ks", [1, 10, 100])),
        chunk_size=int(raw.get("chunk_size", 50_000)),
        mse_sample=int(raw.get("mse_sample", 100_000)),
        output_dir=str(raw.get("output_dir", "results")),
    )
