import numpy as np

from .base_quantizer import BaseQuantizer


class ScalarQuantizer(BaseQuantizer):
    """
    Scalar Quantization with configurable bit depth.

    Quantizes each dimension independently by scaling values to fit within
    a fixed number of bits (4, 8, or 16).

    Args:
        num_bits: Number of bits per dimension (4, 8, or 16)
                  - 4-bit: 16 quantization levels, ~0.5 bytes/dim
                  - 8-bit: 256 quantization levels, 1 byte/dim
                  - 16-bit: 65536 quantization levels, 2 bytes/dim
    """

    def __init__(self, num_bits: int = 8):
        if num_bits not in [4, 8, 16]:
            raise ValueError(f"num_bits must be 4, 8, or 16, got {num_bits}")

        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.min = None
        self.max = None

        # Determine storage dtype based on bits
        if num_bits <= 8:
            self.dtype = np.uint8
            self.bytes_per_dim = 1
        else:  # 16 bits
            self.dtype = np.uint16
            self.bytes_per_dim = 2

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)

    def compress(self, X):
        # Scale to [0, num_levels-1]
        scaled = (X - self.min) / (self.max - self.min + 1e-8)
        quantized = np.round(scaled * (self.num_levels - 1)).astype(self.dtype)

        # For 4-bit, pack two values into one byte
        if self.num_bits == 4:
            # Reshape to pairs and pack
            if X.shape[1] % 2 == 1:
                # Pad with zeros if odd number of dimensions
                quantized = np.pad(quantized, ((0, 0), (0, 1)), mode='constant')

            # Pack pairs: high nibble (bits 4-7) and low nibble (bits 0-3)
            packed = (quantized[:, 0::2] << 4) | quantized[:, 1::2]
            return packed

        return quantized

    def decompress(self, codes):
        # Unpack 4-bit values if needed
        if self.num_bits == 4:
            # Unpack nibbles
            high_nibbles = (codes >> 4).astype(np.float32)
            low_nibbles = (codes & 0x0F).astype(np.float32)

            # Interleave to restore original shape
            unpacked = np.empty((codes.shape[0], codes.shape[1] * 2), dtype=np.float32)
            unpacked[:, 0::2] = high_nibbles
            unpacked[:, 1::2] = low_nibbles

            # Remove padding if it was added
            if len(self.min) % 2 == 1:
                unpacked = unpacked[:, :-1]

            scaled = unpacked / (self.num_levels - 1)
        else:
            scaled = codes.astype(np.float32) / (self.num_levels - 1)

        return scaled * (self.max - self.min + 1e-8) + self.min

    def get_compression_ratio(self, X):
        original_size = X.shape[1] * 4  # float32 = 4 bytes per dimension

        if self.num_bits == 4:
            # Two 4-bit values per byte
            compressed_size = (X.shape[1] + 1) // 2  # Ceiling division
        else:
            compressed_size = X.shape[1] * self.bytes_per_dim

        return original_size / compressed_size