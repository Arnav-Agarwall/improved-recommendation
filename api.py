from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Step 1: Load Dataset
movies_df = pd.read_csv('tmdb_5000_movies.csv')

# Step 2: Data Preprocessing
def extract_names(data):
    try:
        data = ast.literal_eval(data)  # Convert string to list of dictionaries
        return [item['name'] for item in data]
    except (ValueError, SyntaxError):
        return []

movies_df['genres_processed'] = movies_df['genres'].apply(extract_names)
movies_df['keywords_processed'] = movies_df['keywords'].apply(extract_names)

movies_df['combined_features'] = (
    movies_df['genres_processed'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['keywords_processed'].apply(lambda x: ' '.join(x)) + ' ' +
    movies_df['overview'].fillna('')
)

# Step 3: Vectorize Combined Features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['combined_features'])

# Step 4: Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Define Recommendation Functions
def recommend_movies(title, cosine_sim=cosine_sim):
    try:
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:11]]
        return movies_df['title'].iloc[sim_indices].tolist()
    except IndexError:
        return ["Movie not found. Please check the title."]

def recommend_based_on_ratings(user_ratings):
    high_rated_movies = [movie for movie, rating in user_ratings if rating >= 4]
    recommended_movies = set()
    for movie in high_rated_movies:
        recommended_movies.update(recommend_movies(movie))
    rated_movies = {movie for movie, _ in user_ratings}
    final_recommendations = recommended_movies - rated_movies
    return list(final_recommendations)

# API Endpoints
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommend movies based on a single movie title.
    Request format:
    {
        "title": "Movie Title"
    }
    """
    data = request.get_json()
    title = data.get('title')
    if not title:
        return jsonify({"error": "Please provide a movie title."}), 400

    recommendations = recommend_movies(title)
    return jsonify({"recommendations": recommendations})

@app.route('/recommend_by_ratings', methods=['POST'])
def recommend_by_ratings():
    """
    Recommend movies based on user ratings.
    Request format:
    {
        "ratings": [
            {"title": "Movie Title 1", "rating": 5},
            {"title": "Movie Title 2", "rating": 3}
        ]
    }
    """
    data = request.get_json()
    ratings = data.get('ratings')
    if not ratings:
        return jsonify({"error": "Please provide movie ratings."}), 400

    user_ratings = [(r['title'], r['rating']) for r in ratings]
    recommendations = recommend_based_on_ratings(user_ratings)
    return jsonify({"recommendations": recommendations})

# Run Flask App
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)