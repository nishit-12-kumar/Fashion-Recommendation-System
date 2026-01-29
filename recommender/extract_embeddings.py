# recommender/extract_embeddings.py

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


def extract_image_embeddings(
    image_dir="data/images",
    metadata_path="data/metadata.csv",
    output_embedding_path="embeddings/image_embeddings.npy",
    output_metadata_path="embeddings/metadata.csv",
):
    """
    Extracts image embeddings using pretrained ResNet18.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load metadata ----------
    df = pd.read_csv(metadata_path)

    # ---------- Load pretrained ResNet ----------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # remove classification layer
    model = model.to(device)
    model.eval()

    # ---------- Image preprocessing ----------
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    embeddings = []

    with torch.no_grad():
        for img_path in df["image_path"]:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            emb = model(img_tensor)
            emb = emb.cpu().numpy().flatten()

            embeddings.append(emb)

    embeddings = np.array(embeddings, dtype="float32")

    # ---------- Save outputs ----------
    os.makedirs(os.path.dirname(output_embedding_path), exist_ok=True)

    np.save(output_embedding_path, embeddings)
    df.to_csv(output_metadata_path, index=False)

    print("Embeddings extracted successfully!")
    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    extract_image_embeddings()
