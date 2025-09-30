import numpy as np


def bits_to_dtype(bits: int) -> type:
    if bits <= 8:
        return np.uint8
    elif bits <= 16:
        return np.uint16
    elif bits <= 32:
        return np.uint32
    elif bits <= 64:
        return np.uint64
    else:
        raise ValueError(f"{bits} > 64")