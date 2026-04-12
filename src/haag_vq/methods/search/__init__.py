# src/haag_vq/methods/search/__init__.py
from .flat_quantized_index import FlatQuantizedIndex
from .ivf_quantized_index import IvfQuantizedIndex
from .faiss_ivfpq_index import FaissIvfPqIndex

__all__ = ["FlatQuantizedIndex", "IvfQuantizedIndex", "FaissIvfPqIndex"]

try:
    from .saq_index import SaqIndex
    __all__.append("SaqIndex")
except ImportError:
    pass
