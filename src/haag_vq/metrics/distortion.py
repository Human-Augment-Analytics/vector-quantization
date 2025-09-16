import numpy as np


def compute_distortion(X_original, X_compressed_codes, model):
    X_reconstructed = model.decompress(X_compressed_codes)
    diffs = X_original - X_reconstructed
    return np.mean(np.sum(diffs ** 2, axis=1))