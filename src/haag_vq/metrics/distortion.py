import numpy as np


def compute_distortion(X_original, X_compressed_codes, pq):
    X_reconstructed = pq.decompress(X_compressed_codes)
    diffs = X_original - X_reconstructed
    return np.mean(np.linalg.norm(diffs, axis=1))
