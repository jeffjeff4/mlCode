import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm


# YouTube Recall Model
class YouTubeRecallModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super(YouTubeRecallModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.fc1 = nn.Linear(embed_dim + embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_items)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, user_ids, context_emb=None):
        user_emb = self.user_emb(user_ids)
        if context_emb is None:
            context_emb = torch.zeros_like(user_emb)
        x = torch.cat([user_emb, context_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        scores = self.fc3(x)
        return scores

    def generate_candidates(self, user_ids, top_k=100):
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids)
            _, top_k_indices = torch.topk(scores, k=top_k, dim=1)
        return top_k_indices


# Data Generation
def generate_synthetic_youtube_data(num_users=1000, num_items=1000, min_interactions=10, max_interactions=50):
    data = []
    np.random.seed(42)
    for user_id in range(1, num_users + 1):
        num_interactions = np.random.randint(min_interactions, max_interactions + 1)
        item_ids = np.random.choice(range(1, num_items + 1), size=num_interactions, replace=False)
        for item_id in item_ids:
            data.append([user_id, item_id])
    df = pd.DataFrame(data, columns=['user_id', 'item_id'])
    return df


class YouTubeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.users = df['user_id'].unique()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        user_items = self.df[self.df['user_id'] == user_id]['item_id'].values
        target_item = np.random.choice(user_items)
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(target_item, dtype=torch.long)


# Training
def train_youtube_recall(model, train_loader, num_epochs=10, lr=0.001,
                         device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for user_ids, target_items in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            user_ids, target_items = user_ids.to(device), target_items.to(device)
            optimizer.zero_grad()
            scores = model(user_ids)
            loss = criterion(scores, target_items)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    torch.save(model.state_dict(), 'youtube_recall_model.pth')
    return model


# Evaluation
def evaluate_youtube_recall(model, test_loader, k_values=[10, 50, 100], num_items=1000,
                            device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    recall = {k: 0 for k in k_values}
    precision = {k: 0 for k in k_values}
    total = 0

    with torch.no_grad():
        for user_ids, target_items in test_loader:
            user_ids, target_items = user_ids.to(device), target_items.to(device)
            top_k_indices = model.generate_candidates(user_ids, max(k_values))
            for k in k_values:
                top_k = top_k_indices[:, :k]
                for i in range(len(user_ids)):
                    if target_items[i] in top_k[i]:
                        recall[k] += 1
                        precision[k] += 1 / k
            total += len(user_ids)

    for k in k_values:
        recall[k] = recall[k] / total
        precision[k] = precision[k] / total
        print(f'Recall@{k}: {recall[k]:.4f}, Precision@{k}: {precision[k]:.4f}')


# Run the pipeline
if __name__ == "__main__":
    # Generate data
    data_df = generate_synthetic_youtube_data(num_users=1000, num_items=1000)
    train_df = data_df.sample(frac=0.8, random_state=42)
    test_df = data_df.drop(train_df.index)
    train_dataset = YouTubeDataset(train_df)
    test_dataset = YouTubeDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model
    model = YouTubeRecallModel(num_users=1001, num_items=1001, embed_dim=64)  # +1 for 1-based indexing

    # Train
    model = train_youtube_recall(model, train_loader, num_epochs=10)

    # Evaluate
    evaluate_youtube_recall(model, test_loader, k_values=[10, 50, 100], num_items=1000)