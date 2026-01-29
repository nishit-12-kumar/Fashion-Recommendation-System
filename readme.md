# 👗 Fashion Recommendation System using Deep Learning & FAISS

## 📌 Overview

This project is a **content-based Fashion Recommendation System** that recommends visually similar fashion products using **deep learning image embeddings** and **efficient similarity search**.

Instead of relying on user ratings or purchase history, the system analyzes **product images and metadata** to understand fashion styles and suggests similar items. This approach is especially useful when **user interaction data is limited or unavailable**.

---

## 🎯 What This Project Does

- Accepts a fashion product (image or product ID) as input  
- Extracts **deep visual features** from product images using a pretrained **ResNet** model  
- Converts images into **numerical embeddings**  
- Uses **FAISS (Facebook AI Similarity Search)** to find visually similar products  
- Returns **Top-K recommended fashion items**  
- Exposes the recommendation logic via a **Flask backend API**  
- Displays recommendations through a simple **frontend interface**

---

## 🧠 Key Features

- Content-based recommendation (no user ratings required)
- Image-based similarity using deep learning
- Fast and scalable similarity search using FAISS
- Modular and clean project architecture
- Precomputed embeddings for high performance
- Easily extensible for future improvements

---

## 🛠️ Tech Stack & Tools Used

### Programming & Frameworks
- Python
- Flask (Backend API)
- HTML, CSS (Frontend)

### Machine Learning & Deep Learning
- PyTorch
- Pretrained ResNet model for image feature extraction
- NumPy, Pandas for data processing

### Similarity Search
- FAISS for efficient nearest neighbor search

### Data Handling
- CSV-based metadata
- Image datasets (Apparel & Footwear)

---

## ⚙️ How the Recommendation System Works

1. **Image Loading**  
   Fashion product images are loaded from the dataset.

2. **Feature Extraction**  
   Images are passed through a pretrained **ResNet** model to extract high-level visual features.

3. **Embedding Generation**  
   Each image is converted into a fixed-length numerical vector (embedding).

4. **FAISS Indexing**  
   All embeddings are indexed using FAISS for fast similarity search.

5. **Similarity Search**  
   Given a query image, FAISS retrieves the most similar embeddings.

6. **Recommendation Output**  
   The system returns the **Top-K visually similar fashion products**.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/fashion-recommendation-system.git
cd fashion-recommendation-system
```
### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Flask Application
```bash
python backend/app.py
```

### 4️⃣ Open in Browser
```bash
http://127.0.0.1:5000/
```

## 📈 Future Enhancements

- Hybrid recommendation system (image + user preferences)
- Use CLIP for image-text joint embeddings
- User authentication and personalization
- Deployment using Docker and cloud platforms
- Frontend enhancement using React

## 👤 Author

- Nishit Kumar
- B.Tech, NITK Surathkal
- Data Science & Machine Learning Enthusiast

