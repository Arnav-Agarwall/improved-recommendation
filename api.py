import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
movies_df = pd.read_csv("tmdb_5000_movies.csv", usecols=['id', 'original_title', 'vote_average', 'vote_count', 'genres', 'keywords'])

# Filter and reduce dataset size for memory efficiency
movies_df = movies_df[movies_df['vote_count'] > 50]  # Keep movies with more than 50 votes

# Normalize numerical columns
scaler = MinMaxScaler()
movies_df[['vote_average', 'vote_count']] = scaler.fit_transform(movies_df[['vote_average', 'vote_count']])

# Process genres and keywords into a single "content" column
movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(genre['name'] for genre in eval(x)) if pd.notnull(x) else '')
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join(keyword['name'] for keyword in eval(x)) if pd.notnull(x) else '')
movies_df['content'] = movies_df['genres'] + ' ' + movies_df['keywords']

# Precompute content similarity matrix
content_similarity_path = "content_similarity.pkl"
if os.path.exists(content_similarity_path):
    with open(content_similarity_path, "rb") as f:
        content_similarity = pickle.load(f)
else:
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)  # Limit features for memory efficiency
    content_matrix = tfidf.fit_transform(movies_df['content'])
    content_similarity = cosine_similarity(content_matrix, dense_output=False)
    with open(content_similarity_path, "wb") as f:
        pickle.dump(content_similarity, f)

# Precompute collaborative similarity matrix
collaborative_similarity_path = "collaborative_similarity.pkl"
if os.path.exists(collaborative_similarity_path):
    with open(collaborative_similarity_path, "rb") as f:
        collaborative_similarity = pickle.load(f)
else:
    user_item_matrix = csr_matrix((movies_df['vote_average'], (movies_df['id'], range(len(movies_df)))), shape=(movies_df['id'].max() + 1, len(movies_df)))
    svd = TruncatedSVD(n_components=20)  # Reduce dimensions to save memory
    latent_factors = svd.fit_transform(user_item_matrix)
    collaborative_similarity = cosine_similarity(latent_factors, dense_output=False)
    with open(collaborative_similarity_path, "wb") as f:
        pickle.dump(collaborative_similarity, f)

# Recommendation function
def recommend_movies(movie_ratings, num_recommendations=10):
    input_movies = movies_df[movies_df['original_title'].str.lower().isin([title.lower() for title in movie_ratings.keys()])]
    if input_movies.empty:
        return None, "None of the input movies were found in the dataset."

    max_rating = max(movie_ratings.values())
    min_rating = min(movie_ratings.values())
    normalized_ratings = {title.lower(): (rating - min_rating) / (max_rating - min_rating) for title, rating in movie_ratings.items()}

    weighted_scores = np.zeros(content_similarity.shape[0])
    for title, rating in normalized_ratings.items():
        movie_idx = movies_df[movies_df['original_title'].str.lower() == title].index
        if not movie_idx.empty:
            movie_idx = movie_idx[0]
            weighted_scores += rating * (
                content_similarity[movie_idx].toarray().flatten() + collaborative_similarity[movie_idx].toarray().flatten()
            )

    recommended_indices = np.argsort(weighted_scores)[::-1][:num_recommendations]
    recommendations = movies_df.iloc[recommended_indices]['original_title'].tolist()
    return recommendations, None

# Flask route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        movie_ratings = data.get("movie_ratings", {})

        if not movie_ratings or not isinstance(movie_ratings, dict):
            return jsonify({"error": "Invalid input. Expected a dictionary of movie ratings."}), 400

        recommendations, error = recommend_movies(movie_ratings)
        if error:
            return jsonify({"error": error}), 404

        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
