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
##4) make sure the training dataset us usable in din model
##5) training sample including text embedding for comments
##
##3. train din ,including text embedding for comments
##
##4. evaluate model, using ndcg, mse, auc metrics, etc


# Step 1: Create Synthetic Ranking Log with Text Embedding (no pretrained model)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import string
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from xgbPosNegSamplesV0 import user_ids

# Reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

NUM_SAMPLES = 1000
NUM_USERS = 100
NUM_PRODUCTS = 200
MAX_SEQ_LEN = 10

# Generate Vocabulary for Comment Embedding
#VOCAB = list(string.ascii_lowercase + " ")
#CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 reserved for padding

product_ids = [f"p{idx}" for idx in range(NUM_PRODUCTS)]
user_ids = [f"u{idx}" for idx in range(NUM_USERS)]

def simulateUserAction():
    clicked = np.random.rand() < 0.1
    added = clicked and np.random.rand() < 0.5
    purchased = clicked and added and np.random.rand() < 0.3
    return clicked, added, purchased

# Simple comment generator
def generate_comment(score=1):
    pos_sentiments = ["love", "like", "good", "great", "awesome"]
    neg_sentiments = ["bad", "poor", "hate", "dislike", "terrible"]
    templates = [
        "I {} this product.",
        "It is really {}.",
        "Feels so {} after using it.",
        "This is a {} item.",
        "Not sure but it's {}."
    ]
    if score<4:
        sentiment = random.choice(pos_sentiments)
    else:
        sentiment = random.choice(neg_sentiments)

    template = random.choice(templates)
    return template.format(sentiment)

def text_to_tensor(text, max_len=50):
    indices = [CHAR2IDX.get(c, 0) for c in text.lower()[:max_len]]
    return torch.tensor(indices + [0] * (max_len - len(indices)))

# Generate synthetic log
start_time = datetime.now()
data = []
for _ in range(NUM_SAMPLES):
    product_id = random.choice(product_ids)
    user_id = random.choice(user_ids)
    title = f"Product {product_id}"
    price = round(random.uniform(5, 500), 2)
    ranking_score = random.randint(1, 6)
    timestamp = start_time + timedelta(minutes=random.randint(0, 100000))

    clicked, added, purchased = simulateUserAction()

    comment = generate_comment(ranking_score)
    ts = start_time + timedelta(minutes=random.randint(0, 60*24*30))
    ranking_position = random.randint(1, 61)

    data.append([
        title, product_id, user_id, price, ranking_score,
        clicked, added, purchased, comment,
        timestamp, ranking_position
    ])

# Convert to DataFrame
columns = ["title", "product_id", "user_id", "price", "ranking_score",
           "clicked", "added", "purchased", "comment",
           "timestamp", "ranking_position"]
df = pd.DataFrame(data, columns=columns)
p0=0.3
p1=0.3
df['is_positive'] = p0 * df['clicked'] + p1 * df['added'] + (1 - p0 - p1) * df['purchased']

# Label encode users/products
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()
df['user_id_enc'] = user_encoder.fit_transform(df['user_id'])
df['product_id_enc'] = product_encoder.fit_transform(df['product_id'])


def simpleTokenize(text):
    return text.lower().replace('.', '').replace('!', '').replace(',', '').split()

from collections import Counter

all_tokens = []
for text in df['comment']:
    all_tokens.extend(simpleTokenize(text))

vocab_counter = Counter(all_tokens)
vocab = {word: idx+1 for idx , (word, _) in enumerate(vocab_counter.items())}
vocab['<PAD>'] = 0

print(vocab)

def textToIds(text, vocab):
    rst = []
    for tok in simpleTokenize(text):
        tok_id = vocab.get(tok, 0)
        rst.append(tok_id)
    return rst

df['comment_ids'] = df['comment'].apply(lambda x: textToIds(x, vocab))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 16
vocab_size = len(vocab)
word_embedding = nn.Embedding(vocab_size, embedding_dim).to(device)


def embedComment(id_list, embedding_layer):
    if len(id_list) == 0:
        return torch.zeros(embedding_dim)
    ids = torch.tensor(id_list, dtype=torch.long).to(device)
    embs = embedding_layer(ids)
    return embs.mean(dim=0).detach().cpu()

comment_embeddings = []
for ids in df['comment_ids']:
    emb = embedComment(ids, word_embedding)
    comment_embeddings.append(emb.numpy())

comment_embeddings = np.vstack(comment_embeddings)
for idx in range(embedding_dim):
    df[f'comment_emb_{idx}'] = comment_embeddings[:, idx]



# Step 2: Generate DIN training samples (history, target, etc.)
df = df.sort_values(['user_id', 'timestamp'])

user_histories = []
# Group by user and sort by timestamp
for user_id, user_df in df.groupby("user_id"):
    for idx, row in user_df.iterrows():
        past = user_df[user_df['timestamp'] < row['timestamp']]
        if past.empty:
            continue

        list_col = []
        for col in df.columns:
            if col.startswith('comment_emb_'):
                list_col.append(col)

        #hist_embs = past[[c for c in df.columns if c.startswith('comment_emb_')]].values.tolist()
        hist_embs = past[list_col].values.tolist()

        target_emb = row[list_col]
        target_emb = np.array(target_emb.values, dtype=np.float32)
        user_histories.append({
            "product_id": row['product_id_enc'],
            "price": row['price'],
            "ranking_score": row['ranking_score'],
            "ranking_position": row['ranking_position'],
            "is_positive": row['is_positive'],
            "hist_embs": hist_embs,
            "target_emb": target_emb
        })

din_df = pd.DataFrame(user_histories)
print(din_df.head())

# Step 3: Define and Train DIN Model
class DINDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        dense_feats = torch.tensor([row['price'], row['ranking_score'], row['ranking_position']]
                                   , dtype=torch.float32)
        hist_embs = torch.tensor(row['hist_embs'], dtype=torch.float32)
        target_emb = torch.tensor(row['target_emb'], dtype=torch.float32)
        label = torch.tensor(row['is_positive'], dtype=torch.float32)
        return dense_feats, hist_embs, target_emb, label

train_data = DINDataset(din_df)



def collate_fn(batch):
    dense_list, hist_list, target_list, label_list = zip(*batch)
    dense_feats = torch.stack(dense_list)
    target_emb = torch.stack(target_list)
    labels = torch.stack(label_list)

    max_len = max(h.shape[0] for h in hist_list)
    emb_dim = hist_list[0].shape[1]

    padded_hist = []
    for h in hist_list:
        pad_len = max_len - h.shape[0]
        if pad_len > 0:
            pad = torch.zeros(pad_len, emb_dim)
            h_padded = torch.cat([h, pad], dim=0)
        else:
            h_padded = h

        padded_hist.append(h_padded)

    hist_embs = torch.stack(padded_hist)
    return dense_feats, hist_embs, target_emb, labels

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)



class SimpleDIN(nn.Module):
    def __init__(self, emb_dim=32, num_dense_feats=3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + num_dense_feats, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, dense_feats, hist_embs, target_emb):
        attn_scores = self.attention(hist_embs).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vec = (attn_weights.unsqueeze(-1) * hist_embs).sum(dim=1)
        x = torch.cat([dense_feats, context_vec, target_emb], dim=-1)
        out = self.fc(x)
        return out.squeeze(-1)


# Initialize model
model = SimpleDIN(emb_dim=embedding_dim).to(device)
optimizer = optim.Adam(list(model.parameters()) +
                       list(word_embedding.parameters(())), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(3):
    total_loss = 0
    for dense_feats, hist_embs, target_emb, labels in train_loader:
        dense_feats, hist_embs, target_emb, labels = (
            dense_feats.to(device),
            hist_embs.to(device),
            target_emb.to(device),
            labels.to(device)
        )

        optimizer.zero_grad()
        preds = model(dense_feats, hist_embs, target_emb)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Next step: Evaluate model with NDCG, MSE, AUC.

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import ndcg_score

# Evaluate model
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for dense_feats, hist_embs, target_emb, labels in train_loader:
        dense_feats, hist_embs, target_emb, labels = (
            dense_feats.to(device),
            hist_embs.to(device),
            target_emb.to(device),
            labels.to(device)
        )

        logits = model(dense_feats, hist_embs, target_emb)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.numpy())

# Compute Metrics
mse = mean_squared_error(all_labels, all_preds)
#auc = roc_auc_score(all_labels, all_preds)

# NDCG: Requires relevance scores; we simulate relevance as the true click labels
ndcg0 = ndcg_score([all_labels], [all_preds])

def ndcg_at_k(labels, preds, k=5):
    order = np.argsort(-np.array(preds))
    rel = np.array(labels)[order][:k]
    dcg = np.sum(rel / np.log2(np.arange(2, rel.size+2)))
    idcg = np.sum(sorted(labels, reverse=True)[:k] / np.log2(np.arange(2, k+2)))
    if idcg > 0:
        return dcg * 1.0 / idcg
    else:
        return 0.0

ndcg1 = ndcg_at_k(all_labels, all_preds, k=5)

print("\nEvaluation Results:")
print(f"MSE: {mse:.4f}")
#print(f"AUC: {auc:.4f}")
print(f"NDCG0: {ndcg0:.4f}")
print(f"NDCG1: {ndcg1:.4f}")
