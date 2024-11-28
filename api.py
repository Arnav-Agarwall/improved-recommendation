import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
movies_df = pd.read_csv("tmdb_5000_movies.csv")

# Select relevant columns
movies_df = movies_df[['id', 'original_title', 'vote_average', 'vote_count', 'genres', 'keywords']]

# Normalize review-related columns
scaler = MinMaxScaler()
movies_df[['vote_average', 'vote_count']] = scaler.fit_transform(movies_df[['vote_average', 'vote_count']])

# Handle genres and keywords
movies_df['genres'] = movies_df['genres'].apply(lambda x: [genre['name'] for genre in eval(x)] if pd.notnull(x) else [])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [keyword['name'] for keyword in eval(x)] if pd.notnull(x) else [])

# Function to compute content-based similarity (based on genres and keywords)
def compute_content_similarity(movies_df):
    # Combine genres and keywords into a single string for each movie
    movies_df['content'] = movies_df['genres'].apply(lambda x: ' '.join(x)) + ' ' + movies_df['keywords'].apply(lambda x: ' '.join(x))
    
    # Use TF-IDF to vectorize content (genres + keywords)
    tfidf = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf.fit_transform(movies_df['content'])
    
    # Calculate cosine similarity between movies based on content
    content_similarity = cosine_similarity(content_matrix, content_matrix)
    return content_similarity

# Function to compute collaborative filtering similarity using SVD
def compute_collaborative_similarity(movies_df):
    # Create a user-item matrix (using vote_average and vote_count)
    user_item_matrix = movies_df.pivot_table(index='id', columns='original_title', values='vote_average', aggfunc='mean', fill_value=0)

    # Perform SVD (matrix factorization) to get the latent factors
    svd = TruncatedSVD(n_components=50)
    matrix = svd.fit_transform(user_item_matrix)
    
    # Calculate cosine similarity between the latent factors
    collaborative_similarity = cosine_similarity(matrix, matrix)
    return collaborative_similarity

# Function to recommend movies based on both content and collaborative filtering
def recommend_movies(movie_ratings, num_recommendations=10):
    """
    Args:
        movie_ratings: A dictionary where keys are movie titles and values are user ratings.
        num_recommendations: Number of recommendations to return.
        
    Returns:
        A list of recommended movies or an error message.
    """
    # Validate input movies
    input_movies = movies_df[movies_df['original_title'].str.lower().isin([title.lower() for title in movie_ratings.keys()])]
    if input_movies.empty:
        return None, "None of the input movies were found in the dataset."

    # Compute content-based similarity (genre + keyword)
    content_similarity = compute_content_similarity(movies_df)

    # Compute collaborative filtering similarity (SVD)
    collaborative_similarity = compute_collaborative_similarity(movies_df)

    # Normalize user ratings to a scale of 0-1
    max_rating = max(movie_ratings.values())
    min_rating = min(movie_ratings.values())
    normalized_ratings = {title: (rating - min_rating) / (max_rating - min_rating) for title, rating in movie_ratings.items()}

    # Calculate weighted scores for each movie based on the user ratings and similarity
    weighted_scores = []
    for idx, row in input_movies.iterrows():
        movie_idx = row['id']
        weighted_score = 0
        for title, rating in normalized_ratings.items():
            if title.lower() in row['original_title'].lower():
                weighted_score += rating * content_similarity[movie_idx].mean() * collaborative_similarity[movie_idx].mean()
        weighted_scores.append((row['original_title'], weighted_score))

    # Sort by weighted score and get the top N recommendations
    recommendations = sorted(weighted_scores, key=lambda x: x[1], reverse=True)[:num_recommendations]

    return [rec[0] for rec in recommendations], None

# Flask route for movie recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    POST /recommend
    Payload: {"movie_ratings": {"Movie1": 5, "Movie2": 3}}
    Response: {"recommendations": ["MovieA", "MovieB", ...]}
    """
    try:
        # Parse input JSON
        data = request.get_json()
        movie_ratings = data.get("movie_ratings", {})

        # Validate input
        if not movie_ratings or not isinstance(movie_ratings, dict):
            return jsonify({"error": "Invalid input. Expected a dictionary of movie ratings."}), 400

        # Get recommendations
        recommendations, error = recommend_movies(movie_ratings)
        if error:
            return jsonify({"error": error}), 404

        # Return recommendations
        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
