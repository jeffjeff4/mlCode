##generate python and pytorch only code
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
##4) make sure the training dataset is usable in BPR (Bayesian Personalized Ranking) model
##5) training sample including text embedding for comments
##6) training sample including user embedding and product id embedding
##
##3. train BPR (Bayesian Personalized Ranking) model, including text embedding for comments
##
##4. evaluate model, using ndcg, mse, auc metrics, etc
##


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import ndcg_score, mean_squared_error, roc_auc_score

# Seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# 1. Create Ranking Log
N_USERS = 100
N_PRODUCTS = 200
N_SAMPLES = 1000
MAX_RANK = 60

user_ids = [f"user_{i}" for i in range(N_USERS)]
product_ids = [f"prod_{i}" for i in range(N_PRODUCTS)]

product_titles = {pid: f"Product {i}" for i, pid in enumerate(product_ids)}
comments_pool = [
    "Great product, I loved it!",
    "Not good, very disappointing.",
    "Will buy again.",
    "Quality is poor.",
    "Satisfied with the purchase.",
    "Do not recommend.",
    "Best product ever!",
    "Waste of money.",
    "Value for money.",
    "Excellent performance."
]


def generate_sample():
    uid = random.choice(user_ids)
    pid = random.choice(product_ids)
    clicked = random.random() < 0.2
    added = clicked and random.random() < 0.5
    purchased = added and random.random() < 0.5
    return {
        "user_id": uid,
        "product_id": pid,
        "product_title": product_titles[pid],
        "price": round(random.uniform(5, 500), 2),
        "ranking_score": random.randint(1, 5),
        "clicked": int(clicked),
        "add_to_cart": int(added),
        "purchased": int(purchased),
        "comments": random.choice(comments_pool),
        "timestamp": datetime.now() - timedelta(days=random.randint(0, 30)),
        "ranking_position": random.randint(1, MAX_RANK)
    }


data = pd.DataFrame([generate_sample() for _ in range(N_SAMPLES)])


# 2. Pure PyTorch Text Embedding (no pretrained model)
class TextEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        return embedded.mean(dim=1)


# Simple tokenizer & vocab
from collections import defaultdict


def build_vocab(comments):
    vocab = defaultdict(lambda: len(vocab))
    vocab["<PAD>"]  # Ensure padding index 0
    for text in comments:
        for word in text.lower().split():
            vocab[word]
    return vocab


def tokenize(text, vocab):
    return [vocab[word] for word in text.lower().split()]


vocab = build_vocab(data["comments"])
VOCAB_SIZE = len(vocab)
EMBED_DIM = 32
text_embedder = TextEmbedder(VOCAB_SIZE, EMBED_DIM)


# Encode comments
def get_text_tensor(comment):
    token_ids = tokenize(comment, vocab)
    return torch.tensor(token_ids, dtype=torch.long)


data["comment_tensor"] = data["comments"].apply(get_text_tensor)
data["timestamp"] = pd.to_datetime(data["timestamp"])

# 3. Generate training samples for BPR
user2seq = {}
for uid, group in data.groupby("user_id"):
    sorted_group = group.sort_values("timestamp")
    user2seq[uid] = sorted_group

# Product & user ID mapping
prod2id = {pid: i for i, pid in enumerate(product_ids)}
user2id = {uid: i for i, uid in enumerate(user_ids)}


def generate_bpr_samples(user2seq):
    samples = []
    for uid, df in user2seq.items():
        pos_items = df[df[["clicked", "add_to_cart", "purchased"]].any(axis=1)]
        neg_items = df[~df[["clicked", "add_to_cart", "purchased"]].any(axis=1)]
        if len(pos_items) == 0 or len(neg_items) == 0:
            continue
        for _, pos in pos_items.iterrows():
            neg = neg_items.sample(1).iloc[0]
            samples.append({
                "user": user2id[uid],
                "pos_item": prod2id[pos["product_id"]],
                "neg_item": prod2id[neg["product_id"]],
                "comment_tensor": pos["comment_tensor"]
            })
    return samples


train_samples = generate_bpr_samples(user2seq)


# 4. BPR Model
class BPRModel(nn.Module):
    def __init__(self, num_users, num_items, text_embedder, embed_dim=32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.text_embedder = text_embedder

    def forward(self, user, pos_item, neg_item, comment_tensor):
        u = self.user_embed(user)
        i_pos = self.item_embed(pos_item)
        i_neg = self.item_embed(neg_item)
        c = self.text_embedder(comment_tensor)
        i_pos += c
        i_neg += c
        score_pos = (u * i_pos).sum(dim=1)
        score_neg = (u * i_neg).sum(dim=1)
        return score_pos, score_neg


# Training
model = BPRModel(len(user2id), len(prod2id), text_embedder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def bpr_loss(pos_score, neg_score):
    return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()


for epoch in range(10):
    total_loss = 0
    random.shuffle(train_samples)
    for sample in train_samples:
        user = torch.tensor([sample["user"]])
        pos_item = torch.tensor([sample["pos_item"]])
        neg_item = torch.tensor([sample["neg_item"]])
        comment_tensor = sample["comment_tensor"].unsqueeze(0)
        pos_score, neg_score = model(user, pos_item, neg_item, comment_tensor)
        loss = bpr_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss:.4f}")


# 5. Evaluation
def evaluate(model, samples):
    y_true, y_scores = [], []
    mse_scores, ndcg_scores = [], []
    for sample in samples:
        user = torch.tensor([sample["user"]])
        pos_item = torch.tensor([sample["pos_item"]])
        neg_item = torch.tensor([sample["neg_item"]])
        comment_tensor = sample["comment_tensor"].unsqueeze(0)
        pos_score, neg_score = model(user, pos_item, neg_item, comment_tensor)
        y_true.extend([1, 0])
        y_scores.extend([pos_score.item(), neg_score.item()])
        mse_scores.append((pos_score.item() - 1) ** 2)
        ndcg_scores.append([pos_score.item(), neg_score.item()])

    auc = roc_auc_score(y_true, y_scores)
    mse = np.mean(mse_scores)
    ndcg = ndcg_score(np.ones((len(ndcg_scores), 2)), np.array(ndcg_scores))
    print(f"AUC: {auc:.4f}, MSE: {mse:.4f}, NDCG: {ndcg:.4f}")


evaluate(model, train_samples)
