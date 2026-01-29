import os
import pandas as pd

ROOT_DIR = "data"
CSV_PATH = "data/fashion.csv"
OUTPUT_METADATA = "data/metadata.csv"

fashion_df = pd.read_csv(CSV_PATH)

records = []
image_id = 0

for root, _, files in os.walk(ROOT_DIR):
    if "images_with_p" in root.lower():
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file).replace("\\", "/")
                product_id = os.path.splitext(file)[0]

                parts = image_path.split("/")
                category = parts[1]        # Apparel / Footwear
                gender = parts[2]          # Boys / Girls / Men / Women

                record = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "product_id": product_id,
                    "category": category,
                    "gender": gender
                }

                records.append(record)
                image_id += 1

metadata_df = pd.DataFrame(records)

# Merge with fashion.csv (if product_id exists there)
if "product_id" in fashion_df.columns:
    metadata_df = metadata_df.merge(
        fashion_df,
        on="product_id",
        how="left"
    )

metadata_df.to_csv(OUTPUT_METADATA, index=False)

print("metadata.csv created successfully")
print(metadata_df.head())
