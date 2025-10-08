####not runnable. Do not use in interview

import numpy as np
import pandas as pd
from io import StringIO
import random

# --- 1. Mock Data Simulation ---
# Simulating the contents of movies.csv, ratings.csv, and users.csv
# This allows the entire script to be self-contained and runnable.

MOVIES_CSV = """
movieId,title
1,Toy Story (1995)
2,Jumanji (1995)
3,Grumpier Old Men (1995)
4,Waiting to Exhale (1995)
5,Father of the Bride Part II (1995)
6,Heat (1995)
7,Sabrina (1995)
"""

RATINGS_CSV = """
userId,movieId,rating,timestamp
1,1,5.0,838983549
1,3,4.0,838983549
1,6,5.0,838983549
2,1,4.0,838983549
2,2,3.0,838983549
3,3,1.0,838983549
3,4,2.0,838983549
3,5,3.0,838983549
4,1,5.0,838983549
4,2,5.0,838983549
4,5,4.0,838983549
"""


class MatrixFactorizationModel:
    """
    Implements Matrix Factorization using Stochastic Gradient Descent (SGD)
    to learn user and item embeddings for collaborative filtering.
    """

    def __init__(self, n_users, n_items, n_factors=10, learning_rate=0.01, reg_lambda=0.05):
        """
        Initialize the model parameters.

        Args:
            n_users (int): Total number of unique users (U).
            n_items (int): Total number of unique movies (M).
            n_factors (int): The dimensionality of the embedding vectors (D).
            learning_rate (float): Alpha in SGD.
            reg_lambda (float): Regularization term (Lambda).
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg_lambda

        # Initialize user (P) and item (Q) embedding matrices
        # P: (U, D), Q: (M, D)
        # Initialization with small random values ensures symmetry breaking and stability.
        scale = 1.0 / np.sqrt(n_factors)
        self.P = np.random.normal(scale=scale, size=(n_users, n_factors))
        self.Q = np.random.normal(scale=scale, size=(n_items, n_factors))

        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = 0.0

    def predict(self, user_idx, item_idx):
        """
        Predict the rating for a given user and item index.
        Prediction formula: r_hat = global_mean + user_bias_u + item_bias_i + P_u @ Q_i.T

        Args:
            user_idx (int): Internal index of the user.
            item_idx (int): Internal index of the item.

        Returns:
            float: The predicted rating, clipped between 1.0 and 5.0.
        """
        prediction = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]

        # Calculate the dot product of user embedding vector and item embedding vector
        # This models the latent interaction between the user and item.
        prediction += self.P[user_idx, :].dot(self.Q[item_idx, :].T)

        # Clip the prediction to the valid rating range (1 to 5)
        return np.clip(prediction, 1.0, 5.0)

    def train(self, data, n_epochs=50, verbose=True):
        """
        Train the model using Stochastic Gradient Descent (SGD).

        Args:
            data (list): List of training tuples (user_idx, item_idx, rating).
            n_epochs (int): Number of training iterations.
            verbose (bool): Whether to print epoch loss.
        """
        # Calculate the global mean rating, a baseline prediction
        self.global_mean = np.mean([r for (_, _, r) in data])
        print(f"Global mean rating initialized to: {self.global_mean:.4f}")

        for epoch in range(1, n_epochs + 1):
            random.shuffle(data)  # Shuffling the data for better convergence

            total_error = 0.0

            for u, i, r_ui in data:
                # 1. Calculate the error (e_ui)
                r_hat_ui = self.predict(u, i)
                error = r_ui - r_hat_ui
                total_error += error ** 2

                # Cache embeddings and biases for update calculation
                P_u = self.P[u, :]
                Q_i = self.Q[i, :]

                # 2. Update Biases (with regularization)
                # bias_update = lr * (error - reg * bias)
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])

                # 3. Update Embeddings (P and Q) (with regularization)
                # This must be done simultaneously using the original values of P_u and Q_i

                # P_u update = lr * (error * Q_i - reg * P_u)
                self.P[u, :] += self.lr * (error * Q_i - self.reg * P_u)

                # Q_i update = lr * (error * P_u - reg * Q_i)
                self.Q[i, :] += self.lr * (error * P_u - self.reg * Q_i)

            rmse = np.sqrt(total_error / len(data))
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:2d}/{n_epochs}, RMSE: {rmse:.4f}")

        print(f"Training complete. Final RMSE: {rmse:.4f}")


# --- 2. Data Loading and Preprocessing ---

def load_and_prepare_data():
    """Load mock data and create necessary mappings."""
    # Load dataframes from simulated CSV strings
    ratings_df = pd.read_csv(StringIO(RATINGS_CSV))
    movies_df = pd.read_csv(StringIO(MOVIES_CSV))

    # --- Indexing and Mapping ---
    # Create mapping from original IDs to continuous indices (0 to N-1)
    # This is essential for array indexing in NumPy.
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()

    user_to_index = {uid: i for i, uid in enumerate(unique_users)}

    # Include all movies from movies.csv to handle movies with no ratings (edge case)
    all_movie_ids = movies_df['movieId'].unique()
    movie_to_index = {mid: i for i, mid in enumerate(all_movie_ids)}
    index_to_movie = {i: mid for mid, i in movie_to_index.items()}

    # Map original IDs in ratings to new indices
    ratings_df['user_idx'] = ratings_df['userId'].map(user_to_index)
    ratings_df['item_idx'] = ratings_df['movieId'].map(movie_to_index)

    # Create training data list: (user_idx, item_idx, rating)
    # Note: We filter out ratings for movies not present in the final movie_to_index map
    training_data = ratings_df.dropna(subset=['user_idx', 'item_idx']).astype({'user_idx': int, 'item_idx': int})
    training_data = training_data[['user_idx', 'item_idx', 'rating']].values.tolist()

    # Create a map from movie index to movie title for display
    movie_titles = movies_df.set_index('movieId')['title'].to_dict()

    n_users = len(unique_users)
    n_items = len(all_movie_ids)  # Use total movies for Q matrix size

    return (n_users, n_items, training_data, user_to_index,
            index_to_movie, movie_titles, ratings_df)


# --- 3. Recommendation Function & Similarity ---

def get_recommendations(model, user_id, top_n, data_maps):
    """
    Calculates predicted ratings for all unrated movies for a given user
    and returns the top N recommendations.
    """
    (n_users, n_items, _, user_to_index, index_to_movie, movie_titles, ratings_df) = data_maps

    # Edge Case 1: User ID not found (Cold Start)
    if user_id not in user_to_index:
        print(
            f"\n--- ERROR: User ID {user_id} not found in training data (Cold Start problem). Returning most popular or default items. ---")
        # For simplicity, returning a placeholder list of all available movies
        return [(movie_titles.get(mid), 0.0) for mid in movie_titles.keys() if mid != 7][:top_n]

    user_idx = user_to_index[user_id]

    # Movies the user has already rated (using original IDs for quick lookup)
    rated_movie_ids = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())

    predictions = []

    # Iterate through all movie indices
    for item_idx in range(n_items):
        movie_id = index_to_movie[item_idx]

        # Only consider movies the user HAS NOT rated
        if movie_id not in rated_movie_ids:
            # Predict the rating using the learned embeddings and biases
            predicted_rating = model.predict(user_idx, item_idx)

            movie_title = movie_titles.get(movie_id, f"Movie {movie_id} (Title Missing)")

            predictions.append((movie_title, predicted_rating))

    # Rank the predictions by predicted rating (descending)
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_n]


def calculate_movie_similarity(model, movie_id_a, movie_id_b, data_maps):
    """
    Calculates the cosine similarity between two movie embeddings.
    """
    (_, _, _, _, _, _, _) = data_maps

    movie_to_index = {i: mid for mid, i in data_maps[4].items()}
    movie_to_index = {mid: i for i, mid in movie_to_index.items()}

    if movie_id_a not in movie_to_index or movie_id_b not in movie_to_index:
        return "One or both movie IDs not found."

    idx_a = movie_to_index[movie_id_a]
    idx_b = movie_to_index[movie_id_b]

    # Extract the embeddings (Q matrix rows)
    emb_a = model.Q[idx_a]
    emb_b = model.Q[idx_b]

    # Cosine Similarity Formula: (A . B) / (||A|| * ||B||)
    dot_product = np.dot(emb_a, emb_b)
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    # Edge Case: Division by zero if embedding norm is zero (should not happen with random init)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# --- 4. Execution and Test Case ---

if __name__ == '__main__':
    # Configuration
    EMBEDDING_DIM = 5  # D, the number of latent factors
    EPOCHS = 100
    TOP_N = 3
    TEST_USER_ID = 2
    TEST_MOVIE_A = 1  # Toy Story (1995)
    TEST_MOVIE_B = 2  # Jumanji (1995)

    print("--- Movie Recommender System (Matrix Factorization via SGD) ---")

    # 1. Load Data
    n_users, n_items, training_data, user_to_index, index_to_movie, movie_titles, ratings_df = load_and_prepare_data()
    # Pack maps/variables used by helper functions
    data_maps = (n_users, n_items, training_data, user_to_index, index_to_movie, movie_titles, ratings_df)

    print(f"Data Loaded: {n_users} users, {n_items} movies (including unrated ones), {len(training_data)} ratings.")
    print(f"Embeddings dimension (D): {EMBEDDING_DIM}")

    # 2. Initialize Model
    model = MatrixFactorizationModel(
        n_users=n_users,
        n_items=n_items,
        n_factors=EMBEDDING_DIM
    )

    # 3. Train Model
    print("\n--- Starting Training ---")
    # Training updates P (user embeddings) and Q (movie embeddings) matrices
    model.train(training_data, n_epochs=EPOCHS, verbose=True)

    # 4. Test Case 1: Embedding Similarity Check
    print(f"\n--- Model Embedding Similarity Check ---")

    similarity = calculate_movie_similarity(model, TEST_MOVIE_A, TEST_MOVIE_B, data_maps)
    title_a = movie_titles.get(TEST_MOVIE_A)
    title_b = movie_titles.get(TEST_MOVIE_B)

    print(f"Cosine Similarity between '{title_a}' and '{title_b}': {similarity:.4f}")
    # Low similarity is expected as User 2 rated Toy Story highly (4.0) but Jumanji low (3.0)

    # 5. Test Case 2: Generate Recommendations
    print(f"\n--- Recommendations for User ID {TEST_USER_ID} (Rated: Toy Story, Jumanji) ---")
    recommendations = get_recommendations(model, TEST_USER_ID, TOP_N, data_maps)

    if recommendations:
        for i, (title, score) in enumerate(recommendations):
            print(f"Rank {i + 1}: {title} (Predicted Rating: {score:.3f})")

    # 6. Edge Case Test 3: Cold Start User (User 999 is unknown)
    print("\n--- Testing Cold Start Edge Case (User ID 999) ---")
    cold_start_recs = get_recommendations(model, 999, TOP_N, data_maps)
    if cold_start_recs:
        print(f"Cold Start Recommendations (Default/Popular): {[r[0] for r in cold_start_recs]}")

    # 7. Edge Case Test 4: Unrated Movie (Movie ID 7, 'Sabrina', has no ratings)
    sabrina_id = 7
    sabrina_idx = [i for i, mid in index_to_movie.items() if mid == sabrina_id][0]
    sabrina_title = movie_titles.get(sabrina_id)

    # Check the embedding of an unrated movie. It will retain its random initialization,
    # but its bias term should remain zero, leading to predictions near global mean.
    print(f"\n--- Testing Unrated Movie Embedding: '{sabrina_title}' ---")
    print(f"Embedding Norm: {np.linalg.norm(model.Q[sabrina_idx]):.4f}")

    # Check prediction for User 1 (ID=1, index=0) on Sabrina (ID=7, index=6)
    user_1_idx = user_to_index[1]
    prediction_sabrina = model.predict(user_1_idx, sabrina_idx)
    print(f"Predicted rating for User 1 on '{sabrina_title}': {prediction_sabrina:.3f}")

