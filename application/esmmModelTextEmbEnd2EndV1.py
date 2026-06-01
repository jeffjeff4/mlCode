##python and pytorch only code
##1. crating ranking log, has 1000 samples, it has
##1) product title
##2) product id
##3) user id
##4) price
##5) ranking score, an int from 1-5, 1 means bad, 5 means good
##6) whether the product is clicked or not
##7) whether the product is added to cart or not
##8) whether the product is purchased or not
##9) comments sharing users' feedback to the product, few natural language sentences, means like it or not, etc
##10) time stam
##11) a product id must get click first, then add to cart, then purchase
##12) positive samples means product has click, or add to cart, or purchase
##13) negative samples means product has no click, no add to cart, no purchase
##14) number of positive samples is much less than negative samples
##15) ranking position, an int from 1-60, 1 means top ranking position in ranking list, 60 means the last ranking position
##16) generate text embedding for comments. please do not use any pretrained model to generate embedding, using pure pytorch to build embedding
##
##
##2. generate model training sample based on:
##1) group by user
##2) sort by time stamp
##3) generate behavior sequence sorted based on time stamp
##4) make sure the training dataset us usable in ESMM model
##5) 在完整的样本数据空间同时学习点击率和转化率（post-view clickthrough&conversion rate，CTCVR），解决了传统CVR预估模型难以克服的样本选择偏差（sample selection bias）和训练数据过于稀疏（data sparsity ）的问题
##6) training sample including text embedding for comments
##
##3. train ESMM model ,including text embedding for comments
##
##4. evaluate model, using ndcg, mse, auc metrics, etc


import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def random_comment():
    sentiments = ["great product", "not worth it", "excellent", "bad quality", "loved it", "terrible experience"]
    return ". ".join(random.choices(sentiments, k=random.randint(1, 3)))


def generate_ranking_log(num_samples=1000):
    product_ids = [f"product_{i}" for i in range(100)]
    user_ids = [f"user_{i}" for i in range(50)]

    records = []
    for _ in range(num_samples):
        product_id = random.choice(product_ids)
        user_id = random.choice(user_ids)
        title = f"Title of {product_id}"
        price = round(random.uniform(5, 500), 2)
        rank_score = random.randint(1, 5)
        ts = datetime(2023, 1, 1) + timedelta(seconds=random.randint(0, 2592000))

        is_clicked = np.random.choice([0, 1], p=[0.9, 0.1])
        is_carted = np.random.choice([0, 1], p=[0.95, 0.05]) if is_clicked else 0
        is_purchased = np.random.choice([0, 1], p=[0.98, 0.02]) if is_carted else 0
        comment = random_comment() if is_clicked else ""
        rank_pos = random.randint(1, 60)

        records.append([
            title, product_id, user_id, price, rank_score,
            is_clicked, is_carted, is_purchased,
            comment, ts, rank_pos
        ])

    df = pd.DataFrame(records, columns=[
        "title", "product_id", "user_id", "price", "rank_score",
        "is_clicked", "is_added_to_cart", "is_purchased",
        "comment", "timestamp", "rank_position"
    ])

    df["label_click"] = df["is_clicked"]
    df["label_purchase"] = df["is_purchased"]
    return df


# Example usage
df = generate_ranking_log(1000)
print(df.head(3))


import torch
import torch.nn as nn
import string
import pandas as pd
import numpy as np

# Sample mini data
df = df.head(200)  # reduce for memory safety

# Define char vocab
CHARS = list(string.ascii_lowercase + " .,!?")
VOCAB = {c: i+1 for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(VOCAB) + 1
MAX_LEN = 100

def tokenize_comment(text, max_len=MAX_LEN, vocab=VOCAB):
    text = text.lower()
    tokens = [vocab.get(c, 0) for c in text[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    return tokens[:max_len]

df["comment_token"] = df["comment"].apply(tokenize_comment)

# Define text embedding module
class CharTextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, output_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (B, L, D)
        pooled = embedded.mean(dim=1)  # (B, D)
        return self.fc(pooled)  # (B, output_dim)

# Initialize model
embed_model = CharTextEmbedding(VOCAB_SIZE, embed_dim=16, output_dim=32)
embed_model.eval()

# Generate embeddings in mini-batches
batch_size = 32
embeddings = []
with torch.no_grad():
    for i in range(0, len(df), batch_size):
        batch_tokens = df["comment_token"].iloc[i:i+batch_size].tolist()
        token_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        emb = embed_model(token_tensor)
        embeddings.extend(emb.tolist())

# Attach to DataFrame
df["comment_embedding"] = embeddings
print(df[["comment", "comment_embedding"]].head(3))

# Make sure click/purchase labels exist
df['click'] = df.get('is_clicked', 0)
df['purchase'] = df.get('is_purchased', 0)

# Create user/item ID to index mappings
unique_users = df["user_id"].unique()
unique_items = df["product_id"].unique()

user2id = {u: i for i, u in enumerate(unique_users)}
item2id = {i: j for j, i in enumerate(unique_items)}

# Map to numeric indices
df["user_idx"] = df["user_id"].map(user2id)
df["item_idx"] = df["product_id"].map(item2id)


class ESMMDataset(Dataset):
    def __init__(self, df):
        self.user_idx = df["user_idx"].values
        self.item_idx = df["item_idx"].values
        self.price = df["price"].values.astype(np.float32)
        self.rank_score = df["rank_score"].values.astype(np.float32)
        self.rank_position = df["rank_position"].values.astype(np.float32)
        self.comment_embedding = df["comment_embedding"].values
        self.click = df["click"].values.astype(np.float32)
        self.purchase = df["purchase"].values.astype(np.float32)

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        return {
            "user_idx": torch.tensor(self.user_idx[idx], dtype=torch.long),
            "item_idx": torch.tensor(self.item_idx[idx], dtype=torch.long),
            "price": torch.tensor(self.price[idx]),
            "rank_score": torch.tensor(self.rank_score[idx]),
            "rank_position": torch.tensor(self.rank_position[idx]),
            "comment_embedding": torch.tensor(np.array(self.comment_embedding[idx]), dtype=torch.float),
            "click": torch.tensor(self.click[idx]),
            "purchase": torch.tensor(self.purchase[idx]),
        }


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ESMMDataset(train_df)
test_dataset = ESMMDataset(test_df)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class ESMMModel(nn.Module):
    def __init__(self, num_users, num_items, text_emb_dim=32, embed_dim=16, mlp_dims=[64, 32]):
        super(ESMMModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        input_dim = embed_dim * 2 + 3 + text_emb_dim  # user + item + 3 dense + text
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
        )

        self.ctr_head = nn.Linear(mlp_dims[1], 1)
        self.cvr_head = nn.Linear(mlp_dims[1], 1)

    def forward(self, user_idx, item_idx, dense_feats, comment_emb):
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        x = torch.cat([u_emb, i_emb, dense_feats, comment_emb], dim=-1)
        shared = self.shared_bottom(x)
        ctr = torch.sigmoid(self.ctr_head(shared)).squeeze()
        cvr = torch.sigmoid(self.cvr_head(shared)).squeeze()
        return ctr, cvr

# Assuming your main DataFrame is named `df`
unique_users = df["user_id"].unique()
unique_items = df["product_id"].unique()

user2id = {user_id: idx for idx, user_id in enumerate(unique_users)}
item2id = {item_id: idx for idx, item_id in enumerate(unique_items)}

# Map user_id and product_id to numeric indices for embedding
df["user_idx"] = df["user_id"].map(user2id)
df["item_idx"] = df["product_id"].map(item2id)

model = ESMMModel(num_users=len(user2id), num_items=len(item2id))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bce_loss = nn.BCELoss()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(3):
    model.train()
    total_ctr_loss, total_cvr_loss = 0, 0
    for batch in train_loader:
        optimizer.zero_grad()

        user_idx = batch["user_idx"]
        item_idx = batch["item_idx"]
        dense_feats = torch.stack([
            batch["price"], batch["rank_score"], batch["rank_position"]
        ], dim=1)
        comment_emb = batch["comment_embedding"]
        ctr_label = batch["click"]
        cvr_label = batch["purchase"]

        ctr_pred, cvr_pred = model(user_idx, item_idx, dense_feats, comment_emb)

        ctr_loss = bce_loss(ctr_pred, ctr_label)
        mask = ctr_label > 0
        cvr_loss = bce_loss(cvr_pred[mask], cvr_label[mask]) if mask.any() else torch.tensor(0.0)

        loss = ctr_loss + cvr_loss
        loss.backward()
        optimizer.step()

        total_ctr_loss += ctr_loss.item()
        total_cvr_loss += cvr_loss.item()

    print(f"Epoch {epoch+1} | CTR Loss: {total_ctr_loss:.4f} | CVR Loss: {total_cvr_loss:.4f}")


from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np

def ndcg_score(y_true, y_score, k=10):
    """Compute NDCG@k"""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    dcg = np.sum((2**y_true_sorted - 1) / np.log2(np.arange(2, 2 + len(y_true_sorted))))
    ideal_dcg = np.sum((2**np.array(sorted(y_true, reverse=True)) - 1) / np.log2(np.arange(2, 2 + len(y_true))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# Evaluate ESMM model
model.eval()
all_ctr_preds, all_ctr_labels = [], []
all_cvr_preds, all_cvr_labels = [], []
all_ndcg_scores = []

with torch.no_grad():
    for batch in test_loader:
        user_idx = batch["user_idx"]
        item_idx = batch["item_idx"]
        dense_feats = torch.stack([
            batch["price"], batch["rank_score"], batch["rank_position"]
        ], dim=1)
        comment_emb = batch["comment_embedding"]
        ctr_label = batch["click"]
        cvr_label = batch["purchase"]

        ctr_pred, cvr_pred = model(user_idx, item_idx, dense_feats, comment_emb)

        # CTR metrics
        all_ctr_preds.extend(ctr_pred.tolist())
        all_ctr_labels.extend(ctr_label.tolist())

        # CVR metrics only on clicked samples
        mask = ctr_label > 0
        if mask.sum() > 0:
            all_cvr_preds.extend(cvr_pred[mask].tolist())
            all_cvr_labels.extend(cvr_label[mask].tolist())

        # NDCG per batch (assumes CTR score as ranking score)
        ndcg_batch = ndcg_score(ctr_label.numpy(), ctr_pred.numpy(), k=10)
        all_ndcg_scores.append(ndcg_batch)

# Calculate CTR metrics
ctr_auc = roc_auc_score(all_ctr_labels, all_ctr_preds)
ctr_mse = mean_squared_error(all_ctr_labels, all_ctr_preds)

# Calculate CVR metrics
if all_cvr_labels:
    cvr_auc = roc_auc_score(all_cvr_labels, all_cvr_preds)
    cvr_mse = mean_squared_error(all_cvr_labels, all_cvr_preds)
else:
    cvr_auc = cvr_mse = None

# NDCG
avg_ndcg = np.mean(all_ndcg_scores)

# Print results
print(f"CTR  - AUC: {ctr_auc:.4f}, MSE: {ctr_mse:.4f}")
if cvr_auc is not None:
    print(f"CVR  - AUC: {cvr_auc:.4f}, MSE: {cvr_mse:.4f}")
else:
    print("CVR  - Not enough clicked samples for evaluation.")
print(f"NDCG@10: {avg_ndcg:.4f}")


