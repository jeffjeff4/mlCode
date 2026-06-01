import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 1. 数据准备
class MovieRecommendationDataset(Dataset):
    def __init__(self, interactions, num_users, num_items):
        """
        interactions: List of (user_id, item_id, label) tuples, label=1 for positive
        num_users: Total number of users
        num_items: Total number of items (movies)
        """
        self.interactions = interactions  # [(user_id, item_id, label), ...]
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, pos_item_id, label = self.interactions[idx]
        # Randomly sample a negative item
        neg_item_id = np.random.randint(0, self.num_items)
        while neg_item_id in [i[1] for i in self.interactions if i[0] == user_id]:
            neg_item_id = np.random.randint(0, self.num_items)
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'pos_item_id': torch.tensor(pos_item_id, dtype=torch.long),
            'neg_item_id': torch.tensor(neg_item_id, dtype=torch.long)
        }

# 2. 双塔模型定义
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embed_dim=64):
        super(TwoTowerModel, self).__init__()
        # User Tower
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embed_dim + 1, 128),  # +1 for age
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        # Item Tower
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.item_fc = nn.Sequential(
            nn.Linear(embed_dim + num_genres, 128),  # +num_genres for one-hot genres
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, user_ids, user_ages, item_ids, item_genres):
        # User Tower
        user_emb = self.user_embedding(user_ids)  # [batch_size, embed_dim]
        user_features = torch.cat([user_emb, user_ages.unsqueeze(1)], dim=1)  # [batch_size, embed_dim+1]
        user_output = self.user_fc(user_features)  # [batch_size, embed_dim]
        # Item Tower
        item_emb = self.item_embedding(item_ids)  # [batch_size, embed_dim]
        item_features = torch.cat([item_emb, item_genres], dim=1)  # [batch_size, embed_dim+num_genres]
        item_output = self.item_fc(item_features)  # [batch_size, embed_dim]
        # Normalize embeddings for stable training
        user_output = nn.functional.normalize(user_output, p=2, dim=1)
        item_output = nn.functional.normalize(item_output, p=2, dim=1)
        return user_output, item_output

# 3. BPR损失函数
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """
    user_emb: User embeddings [batch_size, embed_dim]
    pos_item_emb: Positive item embeddings [batch_size, embed_dim]
    neg_item_emb: Negative item embeddings [batch_size, embed_dim]
    """
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # [batch_size]
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)  # [batch_size]
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    return loss

# 4. 训练函数
def train(model, dataloader, num_epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            user_ids = batch['user_id'].to(device)
            pos_item_ids = batch['pos_item_id'].to(device)
            neg_item_ids = batch['neg_item_id'].to(device)
            # Dummy user ages and item genres (replace with real data)
            user_ages = torch.randn(user_ids.size(0)).to(device)  # [batch_size]
            item_genres = torch.randn(user_ids.size(0), 10).to(device)  # [batch_size, num_genres]

            # Forward pass
            user_emb, pos_item_emb = model(user_ids, user_ages, pos_item_ids, item_genres)
            _, neg_item_emb = model(user_ids, user_ages, neg_item_ids, item_genres)

            # Compute BPR loss
            loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 5. 推理函数（召回）
def recall(model, user_id, user_age, all_item_ids, all_item_genres, top_k=1000, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Compute user embedding
        user_id_tensor = torch.tensor([user_id], dtype=torch.long).to(device)
        user_age_tensor = torch.tensor([user_age], dtype=torch.float).to(device)
        dummy_item_ids = torch.zeros(1, dtype=torch.long).to(device)  # Dummy item ID
        dummy_item_genres = torch.zeros(1, 10).to(device)  # Dummy genres
        user_emb, _ = model(user_id_tensor, user_age_tensor, dummy_item_ids, dummy_item_genres)
        user_emb = user_emb.squeeze(0)  # [embed_dim]

        # Compute all item embeddings
        item_ids_tensor = torch.tensor(all_item_ids, dtype=torch.long).to(device)
        item_genres_tensor = torch.tensor(all_item_genres, dtype=torch.float).to(device)
        dummy_user_ids = torch.zeros(len(all_item_ids), dtype=torch.long).to(device)  # Dummy user IDs
        _, item_embs = model(dummy_user_ids, user_age_tensor.repeat(len(all_item_ids)),
                            item_ids_tensor, item_genres_tensor)  # [num_items, embed_dim]

        # Compute scores (dot product)
        scores = torch.matmul(item_embs, user_emb)  # [num_items]
        _, top_indices = torch.topk(scores, k=top_k)
        return top_indices.cpu().numpy()  # Indices of top-k items

# 6. 示例运行
if __name__ == "__main__":
    # Dummy data
    num_users = 1000
    num_items = 10000
    num_genres = 10
    interactions = [(np.random.randint(0, num_users), np.random.randint(0, num_items), 1) for _ in range(10000)]
    dataset = MovieRecommendationDataset(interactions, num_users, num_items)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(num_users, num_items, num_genres, embed_dim=64).to(device)

    # Train
    train(model, dataloader, num_epochs=5, device=device)

    # Recall for a user
    user_id = 42
    user_age = 25.0
    all_item_ids = list(range(num_items))
    all_item_genres = np.random.randn(num_items, num_genres)  # Dummy genres
    top_k_items = recall(model, user_id, user_age, all_item_ids, all_item_genres, top_k=1000, device=device)
    print(f"Top-1000 recommended movie indices for user {user_id}: {top_k_items[:10]}")
