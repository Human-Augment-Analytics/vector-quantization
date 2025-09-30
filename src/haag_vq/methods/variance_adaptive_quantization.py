import numpy as np

from .base_quantizer import BaseQuantizer

from pyvaq import VAQ


class VarianceAdaptiveQuantizer(BaseQuantizer):
    def __init__(
        self,
        bit_budget: int,
        subspace_num: int,
        min_bits_per_subs: int,
        max_bits_per_subs: int,
        percent_var_explained: float,
    ):
        """ VAQ
        Paparrizos, J., Edian, I., Liu, C., Elmore, A. J., & Franklin, M. J. (2022, May). Fast adaptive similarity search through variance-aware quantization. In 2022 IEEE 38th International Conference on Data Engineering (ICDE) (pp. 2969-2983). IEEE.
        https://ieeexplore.ieee.org/iel7/9835153/9835154/09835314.pdf

        Args:
            bit_budget (int)
            subspace_num (int)
            min_bits_per_subs (int)
            max_bits_per_subs (int)
            percent_var_explained (float)
        """
        self.vaq = VAQ(f"VAQ{bit_budget}m{subspace_num}min{min_bits_per_subs}max{max_bits_per_subs}var{percent_var_explained},EA")

    def fit(self, X: np.ndarray):
        self.vaq.train(X)

    def compress(self, X: np.ndarray) -> np.ndarray:
        return self.vaq.encode(X)

    def decompress(self, compressed: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
