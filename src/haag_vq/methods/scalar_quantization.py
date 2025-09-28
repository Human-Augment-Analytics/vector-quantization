import numpy as np

from .base_quantizer import BaseQuantizer


class ScalarQuantizer(BaseQuantizer):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def compress(self, X):
        scaled = (X - self.min) / (self.max - self.min + 1e-8)
        return np.round(scaled * 255).astype(np.uint8)

    def decompress(self, X_q):
        scaled = X_q.astype(np.float32) / 255.0
        return scaled * (self.max - self.min + 1e-8) + self.min

    # def get_compression_ratio(self, X):
    #     original_size = X.shape[1] * 4  # float32
    #     compressed_size = X.shape[1]    # 1 byte per dimension
    #     return original_size / compressed_size