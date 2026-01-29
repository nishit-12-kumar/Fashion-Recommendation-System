# recommender/dataset_loader.py

import os
import pandas as pd
from torchvision.datasets import FashionMNIST
from PIL import Image


def prepare_fashion_mnist_dataset(
    output_dir="data/images",
    metadata_path="data/metadata.csv",
    limit=2000
):
    """
    Downloads Fashion-MNIST, saves images to disk,
    and creates metadata CSV for recommendation system.
    """

    os.makedirs(output_dir, exist_ok=True)

    label_map = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    dataset = FashionMNIST(
        root="data",
        train=True,
        download=True
    )

    records = []

    for idx, (img, label) in enumerate(dataset):
        if idx >= limit:
            break

        # FashionMNIST image is already PIL (grayscale)
        # Convert to RGB for ResNet compatibility
        img = img.convert("RGB")

        filename = f"fashion_{idx}.jpg"
        img_path = os.path.join(output_dir, filename)
        img.save(img_path)

        records.append({
            "image_id": idx,
            "image_path": img_path,
            "category": label_map[label]
        })

    df = pd.DataFrame(records)
    df.to_csv(metadata_path, index=False)

    return df


if __name__ == "__main__":
    df = prepare_fashion_mnist_dataset()
    print("Dataset prepared successfully!")
    print(df.head())
