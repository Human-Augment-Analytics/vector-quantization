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

SAQ_METHODS = ("saq_paper", "ours", "ours_exact", "rabitq", "lvq",
               "rankaware", "perdim_mse", "rankaware_exact", "perdim_mse_exact")


def build_saq_quantizer(method: str, bpd: float, D: int):
    b = int(round(bpd))
    # VQ_SAQ_APPLY_PCA=0 lets SAQ skip its (slow Eigen) internal PCA when the
    # input data is already PCA-rotated (e.g. vectors_pca.fvecs). Default 1.
    import os
    _apply_pca = os.environ.get("VQ_SAQ_APPLY_PCA", "1") != "0"
    if method == "saq_paper":
        from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter
        return SaqEngineAdapter(
            quant_type="CAQ", avg_bits=bpd, greedy=False, derive_codebooks=False,
            apply_pca=_apply_pca,
        )
    if method == "ours":
        from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter
        return SaqEngineAdapter(
            quant_type="CAQ", avg_bits=bpd, greedy=True, derive_codebooks=True,
            apply_pca=_apply_pca,
        )
    if method == "ours_exact":  # ours, but exact 1-D codebook instead of Lloyd
        from haag_vq.benchmarks.quantizer_adapters import SaqEngineAdapter
        return SaqEngineAdapter(
            quant_type="CAQ", avg_bits=bpd, greedy=True,
            derive_codebooks=True, exact_codebooks=True,
            apply_pca=_apply_pca,
        )
    if method == "rabitq":
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.extended_rabitq import ExtendedRaBitQuantizer
        return FaissQuantizerAdapter(ExtendedRaBitQuantizer(num_bits=b))
    if method == "lvq":
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.lvq_quantization import LVQQuantizer
        return FaissQuantizerAdapter(LVQQuantizer(num_bits=b))
    # Per-dim methods ("Ours" family). Default codebook = LLOYD (data-fit cumsum-
    # k-means) -- the real method's codebook, not the analytic-Gaussian prototype.
    # The *_exact variants use the EXACT (optimal-DP / kmeans1d) codebook as the
    # optimal reference, to show Lloyd leaves ~nothing on the table (it's also the
    # faster of the two -- exact is optimal but ~5-14x slower to build).
    if method == "rankaware":           # PROPOSED method ("Ours"): rank-aware var^0.5 greedy
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
        return FaissQuantizerAdapter(RankAwareQuantizer(avg_bits=bpd, alpha=0.5, packing="ffd", codebook="lloyd"))
    if method == "perdim_mse":          # MSE-greedy ablation (alpha=0)
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
        return FaissQuantizerAdapter(RankAwareQuantizer(avg_bits=bpd, alpha=0.0, packing="ffd", codebook="lloyd"))
    if method == "rankaware_exact":     # optimal-DP-codebook reference for rankaware
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
        return FaissQuantizerAdapter(RankAwareQuantizer(avg_bits=bpd, alpha=0.5, packing="ffd", codebook="exact"))
    if method == "perdim_mse_exact":    # optimal-DP-codebook reference for perdim_mse
        from haag_vq.benchmarks.quantizer_adapters import FaissQuantizerAdapter
        from haag_vq.methods.rank_aware_quantization import RankAwareQuantizer
        return FaissQuantizerAdapter(RankAwareQuantizer(avg_bits=bpd, alpha=0.0, packing="ffd", codebook="exact"))
    raise ValueError(f"Unknown SAQ-study method: {method!r}")
