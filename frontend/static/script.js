const imageGrid = document.getElementById("imageGrid");
const recommendationGrid = document.getElementById("recommendationGrid");
const recommendBtn = document.getElementById("recommendBtn");

let selectedImageIds = [];

/* ---------------- LOAD PRODUCTS (HOME PAGE) ---------------- */
async function loadProducts() {
    const response = await fetch("/products");
    const products = await response.json();

    imageGrid.innerHTML = ""; // clear grid

    products.forEach(product => {
        const card = document.createElement("div");
        card.className = "image-card";
        card.dataset.id = product.image_id;

        const img = document.createElement("img");
        img.src = `/images/${product.image_path.replace("data/", "")}`;
        img.alt = `Product ${product.product_id}`;

        card.appendChild(img);
        imageGrid.appendChild(card);

        // selection logic
        card.addEventListener("click", () => {
            card.classList.toggle("selected");

            const id = product.image_id;

            if (selectedImageIds.includes(id)) {
                selectedImageIds = selectedImageIds.filter(x => x !== id);
            } else {
                selectedImageIds.push(id);
            }

            recommendBtn.disabled = selectedImageIds.length === 0;
            
        });
    });
}

/* ---------------- GET RECOMMENDATIONS ---------------- */
recommendBtn.addEventListener("click", async () => {
    recommendationGrid.innerHTML = "";

    const response = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: selectedImageIds })
    });

    const data = await response.json();

    data.recommendations.forEach(item => {
        const card = document.createElement("div");
        card.className = "image-card";

        const img = document.createElement("img");
        img.src = `/images/${item.image_path.replace("data/", "")}`;
        img.alt = "Recommended product";

        card.appendChild(img);
        recommendationGrid.appendChild(card);
    });
});

// Load on page start
loadProducts();
