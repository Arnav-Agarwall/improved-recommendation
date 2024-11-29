import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
movies_df = pd.read_csv("tmdb_5000_movies.csv", usecols=['id', 'original_title', 'vote_average', 'vote_count', 'genres', 'keywords'])

# Limit dataset size (Top 1000 by vote_count)
movies_df = movies_df.nlargest(1000, 'vote_count')  # Keep only the top 1000 movies

# Process genres and keywords into "content" for TF-IDF
movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(genre['name'] for genre in eval(x)) if pd.notnull(x) else '')
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join(keyword['name'] for keyword in eval(x)) if pd.notnull(x) else '')
movies_df['content'] = movies_df['genres'] + ' ' + movies_df['keywords']

# Create sparse TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)  # Further limit features
content_matrix = tfidf.fit_transform(movies_df['content'])  # Sparse matrix

# Collaborative filtering with SVD
user_item_matrix = csr_matrix((movies_df['vote_average'], (movies_df['id'], range(len(movies_df)))), shape=(movies_df['id'].max() + 1, len(movies_df)))
svd = TruncatedSVD(n_components=10)  # Reduce SVD components even more
latent_factors = svd.fit_transform(user_item_matrix)

# Recommendation function
def recommend_movies(movie_ratings, num_recommendations=10):
    """
    Recommend movies based on user ratings.
    Args:
        movie_ratings: A dictionary of {movie_title: user_rating}.
        num_recommendations: Number of recommendations to return.
    Returns:
        List of recommended movies or an error message.
    """
    # Validate input
    input_movies = movies_df[movies_df['original_title'].str.lower().isin([title.lower() for title in movie_ratings.keys()])]
    if input_movies.empty:
        return None, "None of the input movies were found in the dataset."

    # Normalize user ratings
    max_rating = max(movie_ratings.values())
    min_rating = min(movie_ratings.values())
    normalized_ratings = {title.lower(): (rating - min_rating) / (max_rating - min_rating) for title, rating in movie_ratings.items()}

    # Weighted scoring
    weighted_scores = np.zeros(len(movies_df))
    for title, rating in normalized_ratings.items():
        movie_idx = movies_df[movies_df['original_title'].str.lower() == title].index
        if not movie_idx.empty:
            movie_idx = movie_idx[0]
            # Get cosine similarity without converting to array
            content_sim = cosine_similarity(content_matrix[movie_idx], content_matrix).flatten()
            collaborative_sim = cosine_similarity(latent_factors[movie_idx].reshape(1, -1), latent_factors).flatten()
            weighted_scores += rating * (content_sim + collaborative_sim)

    # Get top recommendations
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
