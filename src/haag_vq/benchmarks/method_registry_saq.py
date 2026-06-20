# src/haag_vq/benchmarks/method_registry_saq.py
"""Registry for the SAQ-study methods outside the faiss family.

- saq_paper : SAQ engine, CAQ + DP allocation (paper baseline)
- ours      : SAQ engine, CAQ + Greedy allocation + native k-means codebook
- rabitq    : standalone Extended (multi-bit) RaBitQ
- lvq       : standalone LVQ

saq_paper/ours require the `saq` wheel (imported lazily in SaqEngineAdapter);
rabitq/lvq are pure numpy. All method-class imports are lazy so a missing
dependency in one path never breaks another.
"""

from __future__ import annotations

SAQ_METHODS = ("saq_paper", "ours", "rabitq", "lvq", "rankaware", "perdim_mse")


def build_saq_quantizer(method: str, bpd: float, D: int):
    b = int(round(bpd))
    if method == "saq_paper":
        from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter
        return SaqEngineAdapter(
            quant_type="CAQ", avg_bits=bpd, greedy=False, derive_codebooks=False
        )
    if method == "ours":
        from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter
        return SaqEngineAdapter(
            quant_type="CAQ", avg_bits=bpd, greedy=True, derive_codebooks=True
        )
    if method == "rabitq":
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.extended_rabitq import ExtendedRaBitQuantizer
        return FaissQuantizerAdapter(ExtendedRaBitQuantizer(num_bits=b))
    if method == "lvq":
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.lvq_quantization import LVQQuantizer
        return FaissQuantizerAdapter(LVQQuantizer(num_bits=b))
    if method == "rankaware":
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
        return FaissQuantizerAdapter(RankAwareQuantizer(avg_bits=bpd, alpha=0.5, packing="ffd"))
    if method == "perdim_mse":
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
        return FaissQuantizerAdapter(RankAwareQuantizer(avg_bits=bpd, alpha=0.0, packing="ffd"))
    raise ValueError(f"Unknown SAQ-study method: {method!r}")
