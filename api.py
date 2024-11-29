import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
movies_df = pd.read_csv("tmdb_5000_movies.csv", usecols=['id', 'original_title', 'vote_average', 'vote_count', 'genres', 'keywords'])

# Preprocess genres and keywords into "content" column
movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join(genre['name'] for genre in eval(x)) if pd.notnull(x) else '')
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join(keyword['name'] for keyword in eval(x)) if pd.notnull(x) else '')
movies_df['content'] = movies_df['genres'] + ' ' + movies_df['keywords']

# Create TF-IDF matrix for content-based filtering
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
content_matrix = tfidf.fit_transform(movies_df['content'])

# Collaborative filtering with SVD
user_item_matrix = csr_matrix((movies_df['vote_average'], (movies_df['id'], range(len(movies_df)))), shape=(movies_df['id'].max() + 1, len(movies_df)))
svd = TruncatedSVD(n_components=10)  # Reduce dimensions
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
    # Normalize user ratings to a scale of 0-1
    max_rating = max(movie_ratings.values())
    min_rating = min(movie_ratings.values())
    normalized_ratings = {title.lower(): (rating - min_rating) / (max_rating - min_rating) for title, rating in movie_ratings.items()}

    # Initialize weighted scores for each movie in the dataset
    weighted_scores = np.zeros(len(movies_df))

    # Process each movie the user has rated
    for title, rating in normalized_ratings.items():
        # Find the index of the movie in the dataset
        movie_idx = movies_df[movies_df['original_title'].str.lower() == title].index
        if movie_idx.empty:
            continue
        
        movie_idx = movie_idx[0]

        # Calculate content and collaborative similarity for this movie
        content_sim = cosine_similarity(content_matrix[movie_idx], content_matrix)  # shape (1, n_movies)
        collaborative_sim = cosine_similarity(latent_factors[movie_idx].reshape(1, -1), latent_factors)  # shape (1, n_movies)

        # Flatten similarity results (from 2D to 1D arrays)
        content_sim = content_sim.flatten()
        collaborative_sim = collaborative_sim.flatten()

        # Add the weighted content and collaborative similarities to the scores
        weighted_scores += rating * (content_sim + collaborative_sim)

    # Normalize scores to make sure no movie is left out
    weighted_scores = weighted_scores / weighted_scores.max()

    # Get top recommendations by sorting based on the weighted scores
    recommended_indices = np.argsort(weighted_scores)[::-1][:num_recommendations]
    recommendations = movies_df.iloc[recommended_indices]['original_title'].tolist()

    return recommendations, None

# Flask route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()  # Get the JSON data from the POST request
        movie_ratings = data.get("movie_ratings", {})

        if not movie_ratings or not isinstance(movie_ratings, dict):
            return jsonify({"error": "Invalid input. Expected a dictionary of movie ratings."}), 400

        # Call the recommendation function
        recommendations, error = recommend_movies(movie_ratings)

        if error:
            return jsonify({"error": error}), 404

        return jsonify({"recommendations": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use port 10000 or any other available port
    app.run(host="0.0.0.0", port=port)
