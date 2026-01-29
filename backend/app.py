# # backend/app.py

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask import send_from_directory
# import os

# from backend.recommender_api import get_recommendations

# app = Flask(__name__)
# CORS(app)  # allow frontend access


# @app.route("/images/<path:filename>")
# def serve_images(filename):
#     # This serves files from the data directory
#     return send_from_directory(
#         directory=os.path.abspath("data"),
#         path=filename
#     )

# @app.route("/recommend", methods=["POST"])
# def recommend():
#     data = request.json
#     image_ids = data.get("image_ids", [])

#     if not image_ids:
#         return jsonify({"error": "No image IDs provided"}), 400

#     recommendations = get_recommendations(image_ids)

#     return jsonify({
#         "recommendations": recommendations
#     })


# if __name__ == "__main__":
#     app.run(debug=True)








from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import pandas as pd

from backend.recommender_api import get_recommendations

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

CORS(app)


# ---------------- HOME ROUTE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- API ----------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    image_ids = data.get("image_ids", [])

    if not image_ids:
        return jsonify({"error": "No image IDs provided"}), 400

    recommendations = get_recommendations(image_ids)

    return jsonify({"recommendations": recommendations})


# ---------------- IMAGE SERVING ----------------
@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(
        directory=os.path.abspath("data"),
        path=filename
    )


import random

@app.route("/products", methods=["GET"])
def get_products():
    df = pd.read_csv("embeddings/metadata.csv")

    # One image per product
    df = df.drop_duplicates(subset="product_id")

    sampled_products = []

    groups = [
        ("Apparel", "Boys"),
        ("Apparel", "Girls"),
        ("Footwear", "Men"),
        ("Footwear", "Women"),
    ]

    # sample items per group
    for category, gender in groups:
        group_df = df[
            (df["category"] == category) &
            (df["gender"] == gender)
        ]

        if len(group_df) > 0:
            sampled = group_df.sample(
                n=min(3, len(group_df))
            )
            sampled_products.append(sampled)

    # combine all groups
    final_df = pd.concat(sampled_products)

    # 🔥 SHUFFLE EVERY REQUEST
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    products = final_df[
        ["image_id", "image_path", "product_id", "category", "gender"]
    ].to_dict(orient="records")

    return jsonify(products)




if __name__ == "__main__":
    app.run(debug=True)
