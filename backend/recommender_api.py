# backend/recommender_api.py

from recommender.recommend import FashionRecommender


# Load once when server starts
recommender = FashionRecommender()


# def get_recommendations(image_ids, top_k=5):
#     """
#     image_ids: list[int]
#     returns: list[dict]
#     """

#     if len(image_ids) == 1:
#         results = recommender.recommend_by_image_id(
#             image_id=image_ids[0],
#             top_k=top_k
#         )
#     else:
#         results = recommender.recommend_by_multiple_images(
#             image_ids=image_ids,
#             top_k=top_k
#         )

#     return results.to_dict(orient="records")


import pandas as pd

def get_recommendations(image_ids, top_k=5):
    candidate_k = 30  # ask FAISS for more

    # Step 1: Get FAISS candidates
    if len(image_ids) == 1:
        results = recommender.recommend_by_image_id(
            image_ids[0],
            candidate_k
        )
    else:
        results = recommender.recommend_by_multiple_images(
            image_ids,
            candidate_k
        )

    # Step 2: Remove duplicate products
    results = results.drop_duplicates(subset="product_id")

    diverse_results = []
    used_product_ids = set()
    used_category_gender = set()

    # Step 3: Enforce diversity
    for _, row in results.iterrows():
        product_id = row["product_id"]
        category_gender = f"{row['category']}-{row['gender']}"

        if product_id in used_product_ids:
            continue

        if category_gender in used_category_gender:
            continue

        diverse_results.append(row)
        used_product_ids.add(product_id)
        used_category_gender.add(category_gender)

        if len(diverse_results) == top_k:
            break

    # Step 4: Fallback if diversity is insufficient
    if len(diverse_results) < top_k:
        for _, row in results.iterrows():
            product_id = row["product_id"]

            if product_id not in used_product_ids:
                diverse_results.append(row)
                used_product_ids.add(product_id)

            if len(diverse_results) == top_k:
                break

    return pd.DataFrame(diverse_results).to_dict(orient="records")
