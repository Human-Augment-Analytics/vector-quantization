from abc import ABC, abstractmethod
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