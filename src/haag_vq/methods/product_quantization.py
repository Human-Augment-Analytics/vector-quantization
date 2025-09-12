import numpy as np
from sklearn.cluster import KMeans

from .base_quantizer import BaseQuantizer


class ProductQuantizer(BaseQuantizer):
    def __init__(self, num_chunks=8, num_clusters=256):
        self.num_chunks = num_chunks
        self.num_clusters = num_clusters
        self.codebooks = []
        self.chunk_dim = None

    def fit(self, X):
        N, D = X.shape
        assert D % self.num_chunks == 0, "D must be divisible by num_chunks"
        self.chunk_dim = D // self.num_chunks
        self.codebooks = []

        for i in range(self.num_chunks):
            chunk = X[:, i * self.chunk_dim:(i + 1) * self.chunk_dim]
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=1, max_iter=100)
            kmeans.fit(chunk)
            self.codebooks.append(kmeans.cluster_centers_)

    def compress(self, X):
        compressed = []
        for i in range(self.num_chunks):
            chunk = X[:, i * self.chunk_dim:(i + 1) * self.chunk_dim]
            centers = self.codebooks[i]
            assignments = np.argmin(np.linalg.norm(chunk[:, np.newaxis] - centers, axis=2), axis=1)
            compressed.append(assignments)
        return np.stack(compressed, axis=1)  # shape: (N, num_chunks)

    def decompress(self, compressed):
        vectors = []
        for i in range(self.num_chunks):
            centers = self.codebooks[i]
            vectors.append(centers[compressed[:, i]])
        return np.hstack(vectors)

    def get_compression_ratio(self, X):
        original_size = X.shape[1] * 4  # float32 = 4 bytes
        compressed_size = self.num_chunks  # one byte per chunk
        return original_size / compressed_size
