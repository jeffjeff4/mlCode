import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm


# MF Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super(MatrixFactorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(item_ids)
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

    def predict(self, user_ids, item_ids):
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids, item_ids)
        return scores


# Data Generation
def generate_synthetic_data(num_users=1000, num_items=1000, min_interactions=10, max_interactions=50):
    data = []
    np.random.seed(42)
    for user_id in range(1, num_users + 1):
        num_interactions = np.random.randint(min_interactions, max_interactions + 1)
        item_ids = np.random.choice(range(1, num_items + 1), size=num_interactions, replace=False)
        ratings = np.random.randint(1, 6, size=num_interactions)
        for item_id, rating in zip(item_ids, ratings):
            data.append([user_id, item_id, rating])
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
    return df


class RatingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = self.df.iloc[idx]['user_id']
        item_id = self.df.iloc[idx]['item_id']
        rating = self.df.iloc[idx]['rating']
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(item_id, dtype=torch.long), torch.tensor(rating,
                                                                                                              dtype=torch.float)


# Training
def train_mf(model, train_loader, num_epochs=10, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for user_ids, item_ids, ratings in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'mf_model.pth')
    return model


# Evaluation
def evaluate_mf(model, test_loader, k_values=[5, 10], num_items=1000,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    mse = 0
    mae = 0
    hr = {k: 0 for k in k_values}
    ndcg = {k: 0 for k in k_values}
    total = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            predictions = model(user_ids, item_ids)
            mse += ((predictions - ratings) ** 2).sum().item()
            mae += (predictions - ratings).abs().sum().item()
            total += len(ratings)

            unique_users = user_ids.unique()
            for user_id in unique_users:
                all_item_ids = torch.arange(1, num_items + 1, device=device)
                user_ids_batch = user_id.repeat(num_items).to(device)
                scores = model(user_ids_batch, all_item_ids)
                user_test_items = item_ids[user_ids == user_id]
                user_test_ratings = ratings[user_ids == user_id]
                for k in k_values:
                    _, top_k_indices = torch.topk(scores, k)
                    top_k_items = all_item_ids[top_k_indices]
                    hits = any(item in top_k_items for item in user_test_items)
                    hr[k] += hits
                    if hits:
                        for test_item, test_rating in zip(user_test_items, user_test_ratings):
                            if test_item in top_k_items:
                                rank = (top_k_items == test_item).nonzero(as_tuple=True)[0].item()
                                ndcg[k] += 1.0 / np.log2(rank + 2)

    rmse = np.sqrt(mse / total)
    mae = mae / total
    for k in k_values:
        hr[k] = hr[k] / len(unique_users)
        ndcg[k] = ndcg[k] / len(unique_users)

    print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}')
    for k in k_values:
        print(f'HR@{k}: {hr[k]:.4f}, NDCG@{k}: {ndcg[k]:.4f}')


# Run the pipeline
if __name__ == "__main__":
    # Generate data
    data_df = generate_synthetic_data(num_users=1000, num_items=1000)
    train_df = data_df.sample(frac=0.8, random_state=42)
    test_df = data_df.drop(train_df.index)
    train_dataset = RatingDataset(train_df)
    test_dataset = RatingDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model
    model = MatrixFactorization(num_users=1001, num_items=1001, embed_dim=64)  # +1 for 1-based indexing

    # Train
    model = train_mf(model, train_loader, num_epochs=10)

    # Evaluate
    evaluate_mf(model, test_loader, k_values=[5, 10], num_items=1000)