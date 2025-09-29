from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


class BaseQuantizer(ABC):
    """
    Abstract base class for quantizers.
    All quantizers must implement the following methods:
    - fit: learn any codebooks or parameters from training data
    - encode: transform input vectors into compressed codes
    - decode: reconstruct approximations from compressed codes
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Learn quantization parameters from training data.

        Args:
            X (np.ndarray): Training data of shape (N, D)
        """
        pass

    @abstractmethod
    def compress(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize the input data into discrete codes.

        Args:
            X (np.ndarray): Data to quantize of shape (N, D)

        Returns:
            np.ndarray: Encoded representation (e.g., int indices)
        """
        pass

    @abstractmethod
    def decompress(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode compressed codes back to approximate vectors.

        Args:
            codes (np.ndarray): Encoded representations

        Returns:
            np.ndarray: Approximate reconstructions of shape (N, D)
        """
        pass

    def save_codebooks(
        self,
        *,
        codes: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None,
        codebook_filename: Optional[str] = None,
        codes_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export codebooks (and optional assignments) to FAISS-compatible files.

        Args:
            codes: Optional quantized codes to store alongside the codebook.
            output_dir: Destination directory; defaults to ``codebooks`` in the
                project root (one level above ``src``).
            codebook_filename: Optional override for the ``.fvecs`` filename.
            codes_filename: Optional override for the ``.ivecs`` filename.

        Returns:
            Mapping describing the exported artifacts (matches
            :func:`haag_vq.utils.faiss_export.export_codebook`).
        """
        # Deferred import avoids circular dependency with haag_vq.utils.
        from haag_vq.utils.faiss_export import export_codebook

        project_root = Path(__file__).resolve().parents[3]
        target_dir = Path(output_dir) if output_dir is not None else project_root / "codebooks"
        target_dir.mkdir(parents=True, exist_ok=True)

        prefix = type(self).__name__.lower()
        cb_name = codebook_filename or f"{prefix}_codebook.fvecs"
        codes_name = codes_filename or f"{prefix}_codes.ivecs"

        return export_codebook(
            self,
            target_dir,
            codes=codes,
            codebook_filename=cb_name,
            codes_filename=codes_name,
        )
