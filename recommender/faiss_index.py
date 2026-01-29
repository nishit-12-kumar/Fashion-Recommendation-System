# recommender/faiss_index.py

import os
import faiss
import numpy as np
import pandas as pd


class FaissImageIndex:
    def __init__(
        self,
        embedding_path="embeddings/image_embeddings.npy",
        metadata_path="embeddings/metadata.csv",
    ):
        self.embedding_path = embedding_path
        self.metadata_path = metadata_path

        self.embeddings = None
        self.metadata = None
        self.index = None

    def load_data(self):
        """Load embeddings and metadata"""
        self.embeddings = np.load(self.embedding_path).astype("float32")
        self.metadata = pd.read_csv(self.metadata_path)

        assert len(self.embeddings) == len(self.metadata), \
            "Embeddings and metadata size mismatch!"

    def build_index(self):
        """Build FAISS index"""
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def search(self, query_embedding, top_k=5):
        """
        Search similar images

        Args:
            query_embedding (np.ndarray): shape (D,) or (1, D)
            top_k (int): number of similar items

        Returns:
            pd.DataFrame: metadata of similar images
        """

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k
        )

        results = self.metadata.iloc[indices[0]].copy()
        results["distance"] = distances[0]

        return results


if __name__ == "__main__":
    faiss_index = FaissImageIndex()
    faiss_index.load_data()
    faiss_index.build_index()
