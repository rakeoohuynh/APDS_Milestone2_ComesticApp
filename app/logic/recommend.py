import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessed import preprocess_text, vocab_dict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"

df = pd.read_csv(DATA_PATH)

for col in ["brand_name", "product_title", "product_tags", "review_text"]:
    df[col] = df[col].fillna("")

product_df = df.groupby("product_id").agg({
    "brand_name": "first",
    "product_title": "first",
    "product_tags": "first",
    "review_text": lambda x: " ".join(x)
}).reset_index()

product_df["combined_text"] = (
    product_df["brand_name"] + " " +
    product_df["product_title"] + " " +
    product_df["product_tags"] + " " +
    product_df["review_text"]
)

tfidf = TfidfVectorizer(vocabulary=vocab_dict)
X_products = tfidf.fit_transform(product_df["combined_text"])


# Recommendation engine
def recommend_products(product_id, top_n = 5):

    if product_id not in product_df["product_id"].values:
        return "Product not found"

    idx = product_df[product_df["product_id"] == product_id].index[0]
    product_vec = X_products[idx]

    similarity = cosine_similarity(product_vec, X_products)

    top_indices = similarity[0].argsort()[::-1][1:top_n+1]
    recommendations = product_df.iloc[top_indices][["product_id", "brand_name", "product_title"]]
    return recommendations

if __name__ == "__main__":
    # Thay thử bằng một ID sản phẩm có thật trong file CSV của bạn
    test_id = product_df["product_id"].iloc[0] 
    print(f"Sản phẩm được đề xuất cho ID {test_id}:")
    print(recommend_products(test_id))
