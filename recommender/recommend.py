# recommender/recommend.py

import numpy as np
from .faiss_index import FaissImageIndex


class FashionRecommender:
    def __init__(self):
        self.faiss_index = FaissImageIndex()
        self.faiss_index.load_data()
        self.faiss_index.build_index()
        self.embeddings = self.faiss_index.embeddings

    def recommend_by_image_id(self, image_id, top_k=5):
        """
        Recommend similar items for a single image.

        Args:
            image_id (int): index of the image
            top_k (int): number of recommendations

        Returns:
            DataFrame: recommended items
        """

        query_embedding = self.embeddings[image_id]

        results = self.faiss_index.search(
            query_embedding=query_embedding,
            top_k=top_k + 1  # +1 to exclude itself
        )

        # Remove the query image itself
        results = results[results.index != image_id]

        return results.head(top_k)

    def recommend_by_multiple_images(self, image_ids, top_k=5):
        """
        Recommend items based on multiple images (outfit-level).

        Args:
            image_ids (list[int]): list of image indices
            top_k (int): number of recommendations

        Returns:
            DataFrame: recommended items
        """

        selected_embeddings = self.embeddings[image_ids]

        # Compute centroid embedding
        centroid_embedding = np.mean(selected_embeddings, axis=0)

        results = self.faiss_index.search(
            query_embedding=centroid_embedding,
            top_k=top_k + len(image_ids)
        )

        # Remove selected images from results
        results = results[~results.index.isin(image_ids)]

        return results.head(top_k)


if __name__ == "__main__":
    recommender = FashionRecommender()

    print("\n--- Single Image Recommendation ---")
    recs = recommender.recommend_by_image_id(image_id=10, top_k=5)
    print(recs)

    print("\n--- Multi Image Recommendation ---")
    recs_multi = recommender.recommend_by_multiple_images(
        image_ids=[10, 20, 30],
        top_k=5
    )
    print(recs_multi)
