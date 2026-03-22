from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import os
import re

app = Flask(__name__)

# Load datasets
ratings = pd.read_csv("ratings.csv").head(15000)
movies = pd.read_csv("movies.csv").head(2000)

# Encode IDs
user_ids = ratings.userId.unique().tolist()
movie_ids = movies.movieId.tolist()

user_map = {x:i for i,x in enumerate(user_ids)}
movie_map = {x:i for i,x in enumerate(movie_ids)}

ratings["user"] = ratings["userId"].map(user_map)
ratings["movie"] = ratings["movieId"].map(movie_map)

# 🔥 SAFE MODEL LOAD (Render fix)
try:
    model = tf.keras.models.load_model("recommender_model.h5", compile=False)
except Exception as e:
    print("Model loading failed:", e)
    model = None

# Cache
poster_cache = {}

# Clean title
def clean_title(title):
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    return title.strip()

# Poster fetch
def get_poster(title):

    if not title:
        return "https://via.placeholder.com/200x300?text=No+Poster"

    if title in poster_cache:
        return poster_cache[title]

    api_key = os.environ.get("OMDB_API_KEY", "da205fe2")
    title_clean = clean_title(title)

    urls = [
        f"http://www.omdbapi.com/?t={title_clean}&apikey={api_key}",
        f"http://www.omdbapi.com/?s={title_clean}&apikey={api_key}"
    ]

    for url in urls:
        try:
            res = requests.get(url, timeout=5).json()

            if res.get("Poster") and res["Poster"] != "N/A":
                poster_cache[title] = res["Poster"]
                return res["Poster"]

            if res.get("Search"):
                for movie in res["Search"]:
                    if movie.get("Poster") and movie["Poster"] != "N/A":
                        poster_cache[title] = movie["Poster"]
                        return movie["Poster"]

        except:
            continue

    fallback = f"https://via.placeholder.com/200x300?text={title_clean[:15]}"
    poster_cache[title] = fallback
    return fallback


# Recommendation function (UNCHANGED)
def recommend(movie_name):
    
    movie_name = movie_name.lower()

    match = movies[movies["title"].str.lower().str.contains(movie_name, na=False)]

    if match.empty:
        sample = movies.sample(10)
        return [{"title": row.title, "poster": get_poster(row.title)} for _, row in sample.iterrows()]

    movie_id = match.iloc[0]["movieId"]

    if movie_id not in movie_map:
        return []

    movie_index = movie_map[movie_id]

    movie_embeddings = model.layers[3].get_weights()[0]
    target = movie_embeddings[movie_index]

    similarity = movie_embeddings @ target

    similar_indices = np.argsort(similarity)[::-1][1:11]

    results = []

    for idx in similar_indices:

        if idx >= len(movie_ids):
            continue

        mid = movie_ids[idx]

        movie_row = movies[movies.movieId == mid]

        if movie_row.empty:
            continue

        title = movie_row.title.values[0]

        results.append({
            "title": title,
            "poster": get_poster(title)
        })

    return results


@app.route("/", methods=["GET","POST"])
def home():

    recs = []

    if request.method == "POST":
        watched = request.form["watched"]
        recs = recommend(watched)

    return render_template("index.html", recs=recs)


# 🔥 RENDER FIX (IMPORTANT PART)
if __name__ == "__main__":
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT env
    app.run(host="0.0.0.0", port=port, debug=False)