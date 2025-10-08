import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Step 1: Generate sample data (since actual CSVs are not provided; in practice, load from 'ratings.csv', 'users.csv', 'movies.csv')
# Assume ratings.csv has columns: user_id, movie_id, rating (1-5 scale)
# users.csv and movies.csv for metadata, but we only need IDs for embeddings

np.random.seed(42)
torch.manual_seed(42)

# Sample data: 3 users, 3 movies, some ratings
ratings_data = {
    'user_id': [1, 1, 2, 2, 3, 3],
    'movie_id': [1, 2, 2, 3, 1, 3],
    'rating': [5.0, 3.0, 4.0, 2.0, 4.0, 1.0]
}
ratings = pd.DataFrame(ratings_data)

num_users = ratings['user_id'].nunique()
num_movies = ratings['movie_id'].nunique()
print(f"Number of users: {num_users}, Number of movies: {num_movies}")
print("Sample ratings:\n", ratings.head())


# Edge cases considered:
# - Sparse data: Not all user-movie pairs rated (cold start for unseen pairs)
# - New users/movies: Handled by embeddings initialized randomly; in production, use averages or zero embeddings
# - Ratings scale: Assumed 1-5; normalize if needed
# - Device: Use CPU for simplicity (runnable here)

# Step 2: Dataset
class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.ratings = ratings_df
        self.user_map = {uid: i for i, uid in enumerate(sorted(ratings_df['user_id'].unique()))}
        self.movie_map = {mid: i for i, mid in enumerate(sorted(ratings_df['movie_id'].unique()))}

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_idx = self.user_map[row['user_id']]
        movie_idx = self.movie_map[row['movie_id']]
        rating = row['rating']
        return torch.tensor(user_idx, dtype=torch.long), torch.tensor(movie_idx, dtype=torch.long), torch.tensor(rating,
                                                                                                                 dtype=torch.float)


dataset = MovieLensDataset(ratings)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Step 3: Model - Simple dot product of user and movie embeddings for predicted rating
class EmbeddingModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=10):  # Small dim for sample
        super(EmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        # No bias for simplicity; add nn.Parameter(torch.zeros(1)) if needed

    def forward(self, user_idx, movie_idx):
        user_emb = self.user_embedding(user_idx)
        movie_emb = self.movie_embedding(movie_idx)
        # Predicted rating as dot product (can add sigmoid/scale for 1-5 range)
        pred_rating = torch.sum(user_emb * movie_emb, dim=1)
        return pred_rating


model = EmbeddingModel(num_users, num_movies)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 4: Training (few epochs for sample)
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for user_batch, movie_batch, rating_batch in dataloader:
        optimizer.zero_grad()
        pred = model(user_batch, movie_batch)
        loss = criterion(pred, rating_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Step 5: Test case - For user 1, rank movies by predicted rating (similarity via dot product)
# Get embeddings
model.eval()
with torch.no_grad():
    user_embs = model.user_embedding(torch.arange(num_users)).numpy()
    movie_embs = model.movie_embedding(torch.arange(num_movies)).numpy()

# For user 0 (user_id=1), compute predicted ratings for all movies
user_id_test = 1  # Map to index
user_idx_test = dataset.user_map[user_id_test]
user_emb_test = user_embs[user_idx_test].reshape(1, -1)

# Predicted ratings via dot product
pred_ratings = np.dot(user_emb_test, movie_embs.T).flatten()

# For similarity, use cosine (normalized dot product)
cosine_sims = cosine_similarity(user_emb_test, movie_embs).flatten()

# Ranking by predicted rating
movie_ranking = np.argsort(pred_ratings)[::-1] + 1  # +1 for movie_id
print(f"\nTest case: Ranking for user {user_id_test} by predicted rating: {movie_ranking}")
print(f"Predicted ratings: {pred_ratings}")

# Ranking by cosine similarity
cosine_ranking = np.argsort(cosine_sims)[::-1] + 1
print(f"Ranking by cosine similarity: {cosine_ranking}")
print(f"Cosine similarities: {cosine_sims}")

# Edge case test: Unseen user (simulate new user with zero embedding or average)
new_user_emb = np.zeros((1, model.user_embedding.embedding_dim))  # Cold start
pred_new = np.dot(new_user_emb, movie_embs.T).flatten()  # All zero
print(f"\nEdge case - New user predicted ratings: {pred_new} (all low/zero)")

# In production: For cold start, use average movie emb or content-based features