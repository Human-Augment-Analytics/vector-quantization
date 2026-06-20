# src/haag_vq/benchmarks/method_registry.py
"""Map (method_name, bpd, D) to a fitted-able Quantizer.

Faiss family (pq/opq/sq) lives here. SAQ-engine family (saq_paper, ours,
rabitq, lvq) is added in method_registry_saq.py once the wheel is built.
"""

from __future__ import annotations

from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter

FAISS_METHODS = ("pq", "opq", "sq")
PQ_BITS_PER_SUB = 8


def largest_divisor_leq(D: int, m: int) -> int:
    """Largest divisor of D that is <= m (and >= 1)."""
    m = max(1, min(m, D))
    for cand in range(m, 0, -1):
        if D % cand == 0:
            return cand
    return 1


def _pq_subquantizers(bpd: float, D: int) -> int:
    total_bits = int(round(bpd * D))
    m = max(1, total_bits // PQ_BITS_PER_SUB)
    return largest_divisor_leq(D, m)


def build_faiss_quantizer(method: str, bpd: float, D: int) -> FaissQuantizerAdapter:
    if method == "pq":
        from haag_vq.methods.product_quantization import ProductQuantizer
        M = _pq_subquantizers(bpd, D)
        return FaissQuantizerAdapter(ProductQuantizer(M=M, B=PQ_BITS_PER_SUB))
    if method == "opq":
        from haag_vq.methods.optimized_product_quantization import (
            OptimizedProductQuantizer,
        )
        M = _pq_subquantizers(bpd, D)
        return FaissQuantizerAdapter(OptimizedProductQuantizer(M=M, B=PQ_BITS_PER_SUB))
    if method == "sq":
        from haag_vq.methods.scalar_quantization import ScalarQuantizer
        nb = 4 if bpd <= 4.5 else (8 if bpd <= 12 else 16)
        return FaissQuantizerAdapter(ScalarQuantizer(num_bits=nb))
    raise ValueError(f"Unknown faiss method: {method!r}")


SAQ_METHODS = ("saq_paper", "ours", "rabitq", "lvq", "rankaware", "perdim_mse")
ALL_METHODS = FAISS_METHODS + SAQ_METHODS


def build_quantizer(method: str, bpd: float, D: int):
    """Dispatch to the faiss family or the SAQ-study family."""
    if method in FAISS_METHODS:
        return build_faiss_quantizer(method, bpd=bpd, D=D)
    if method in SAQ_METHODS:
        from haag_vq.benchmarks.method_registry_saq import build_saq_quantizer
        return build_saq_quantizer(method, bpd=bpd, D=D)
    raise ValueError(f"Unknown method: {method!r}")
