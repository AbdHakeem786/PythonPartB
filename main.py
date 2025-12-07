from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ---------------- Load CSV ----------------
df = pd.read_csv("scrapping_results.csv")

# Ensure required columns exist
required_cols = ["Text", "Keywords", "Title", "URL", "Claps"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing in CSV!")

# ---------------- Combine text + keywords ----------------
df["combined_text"] = df["Text"].fillna("") + " " + df["Keywords"].fillna("")

# ---------------- TF-IDF Vectorization ----------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

# ---------------- Helper: Convert Claps ----------------
def claps_to_number(clap_str):
    if pd.isna(clap_str):
        return 0
    clap_str = str(clap_str).strip().upper()
    try:
        if "K" in clap_str:
            return float(clap_str.replace("K", "")) * 1000
        elif "M" in clap_str:
            return float(clap_str.replace("M", "")) * 1000000
        else:
            return float(clap_str)
    except:
        return 0

df["Claps_Num"] = df["Claps"].apply(claps_to_number)

# ---------------- Recommendation Function ----------------
def get_recommendations(query_text, top_n=10):
    # Transform query into vector
    query_vec = vectorizer.transform([query_text])

    # Compute cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Create a temporary DataFrame to sort
    temp_df = df.copy()
    temp_df["similarity"] = similarities

    # Sort by similarity first, then by claps
    results = temp_df.sort_values(
        by=["similarity", "Claps_Num"], ascending=False
    ).head(top_n)

    # Return list of dicts
    return [
        {
            "title": row["Title"],
            "url": row["URL"],
            "claps": row["Claps"]
        }
        for _, row in results.iterrows()
    ]

# ---------------- API Endpoints ----------------
@app.get("/")
def home():
    return {"message": "Medium Article Similarity API is running."}

@app.get("/recommend")
def recommend(q: str):
    results = get_recommendations(q)
    return {"query": q, "results": results}
