
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import os

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

# Load trained model
model = tf.keras.models.load_model("recommender_model.h5", compile=False)

# Placeholder poster with caching and better fallback
poster_cache = {}

def _omdb_poster_search(title):
    api_key = (os.environ.get("OMDB_API_KEY") or "da205fe2").strip()
    if not api_key or api_key == "da205fe2":
        print("[WARNING] OMDB_API_KEY missing or default; fallback to placeholder if not found")

    title_clean = title.split("(")[0].strip()
    uri_title = requests.utils.requote_uri(title_clean)
    url = f"http://www.omdbapi.com/?t={uri_title}&apikey={api_key}"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data.get("Response") == "True" and data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]

        # Try with search endpoint (more tolerant of exact match issues)
        search_url = f"http://www.omdbapi.com/?s={uri_title}&apikey={api_key}"
        resp2 = requests.get(search_url, timeout=5)
        resp2.raise_for_status()
        data2 = resp2.json()

        if data2.get("Response") == "True" and "Search" in data2 and data2["Search"]:
            best = data2["Search"][0]
            if best.get("Poster") and best["Poster"] != "N/A":
                return best["Poster"]

    except requests.RequestException as e:
        print(f"[WARNING] OMDB request error for '{title}': {e}")
    except ValueError as e:
        print(f"[WARNING] invalid JSON from OMDB for '{title}': {e}")

    return "https://via.placeholder.com/200x300?text=No+Poster"


def get_poster(title):
    if not title:
        return "https://via.placeholder.com/200x300?text=No+Poster"

    if title in poster_cache:
        return poster_cache[title]

    poster_url = _omdb_poster_search(title)
    poster_cache[title] = poster_url
    return poster_url

# Recommendation function
def recommend(movie_name):
    
    movie_name = movie_name.lower()

    match = movies[movies["title"].str.lower().str.contains(movie_name, na=False)]

    if match.empty:
        # if movie not found, return random suggestions
        sample = movies.sample(10)
        return [{"title": row.title, "poster": get_poster(row.title)} for _, row in sample.iterrows()]

    movie_id = match.iloc[0]["movieId"]

    if movie_id not in movie_map:
        return []

    movie_index = movie_map[movie_id]

    # get movie embedding layer
    movie_embeddings = model.layers[3].get_weights()[0]

    # embedding of selected movie
    target = movie_embeddings[movie_index]

    # compute similarity
    similarity = movie_embeddings @ target

    # top similar movies
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


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
