# src/haag_vq/methods/search/rabitq_index.py
"""RaBitQIndex — wraps RaBitQuantizer as a BaseSearchIndex."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from haag_vq.methods.base_search_index import BaseSearchIndex
from haag_vq.methods.search.flat_quantized_index import FlatQuantizedIndex


class RaBitQIndex(BaseSearchIndex):
    """RaBitQ quantizer exposed through the BaseSearchIndex contract.

    RaBitQ has no unique search structure — codes are decoded to
    approximate float vectors and then searched by exact distance — so this
    class delegates to ``FlatQuantizedIndex`` internally. The wrapper exists to
    give RaBitQ a named, discoverable method class alongside ``SaqIndex`` /
    ``FaissIvfPqIndex``, and to provide an explicit save/load contract.

    Notes:
        * RaBitQ encodes each (normalised) vector at ~1 bit per dimension by
          construction; bits-per-dim is not a tunable knob for this method.
        * The underlying ``faiss.RaBitQuantizer`` is a SWIG C++ object and is
          not picklable, so ``save`` / ``load`` raise ``NotImplementedError``.
    """

    def __init__(self) -> None:
        self._inner: Optional[FlatQuantizedIndex] = None

    def fit(self, X: np.ndarray, metric: Literal['l2', 'ip'] = 'l2') -> None:
        # Deferred imports: faiss / RaBitQuantizer are only required when this
        # method is actually used, so importing the package itself stays cheap.
        from haag_vq.methods.rabit_quantization import RaBitQuantizer
        from haag_vq.utils.faiss_utils import MetricType

        mt = MetricType.INNER_PRODUCT if metric == 'ip' else MetricType.L2
        self._inner = FlatQuantizedIndex(RaBitQuantizer(metric_type=mt))
        self._inner.fit(X, metric=metric)

    def _require_fit(self) -> FlatQuantizedIndex:
        if self._inner is None:
            raise RuntimeError("RaBitQIndex must be fit() before use.")
        return self._inner

    def search(self, Q: np.ndarray, k: int) -> np.ndarray:
        return self._require_fit().search(Q, k)

    def search_with_scores(
        self, Q: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._require_fit().search_with_scores(Q, k)

    def memory_footprint(self) -> int:
        return self._inner.memory_footprint() if self._inner is not None else 0

    def reconstruction_mse(
        self,
        X: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        if self._inner is None:
            return None
        return self._inner.reconstruction_mse(X, sample_ids=sample_ids)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(
            "RaBitQIndex.save is not implemented: faiss.RaBitQuantizer is a "
            "SWIG C++ object and cannot be pickled. Re-fit the index in-memory "
            "for each benchmark run."
        )

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(
            "RaBitQIndex.load is not implemented: see RaBitQIndex.save."
        )
