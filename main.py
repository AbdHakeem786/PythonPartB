from fastapi import FastAPI, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Top Similar Articles API")

# Load data
df = pd.read_csv("2212101___scrapping_results.csv")

# Create combined text
df["combined"] = df["title"].fillna("") + " " + df["content"].fillna("")

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"])

@app.get("/search")
def search_articles(query: str = Query(..., description="Enter text or keywords")):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()

    top_idx = similarities.argsort()[::-1][:10]

    results = []
    for i in top_idx:
        results.append({
            "title": df.iloc[i]["title"],
            "url": df.iloc[i]["url"],
            "score": float(similarities[i])
        })

    return {"query": query, "results": results}
