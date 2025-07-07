import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from datetime import datetime, timedelta


# ======================
# 1. Generate Ranking Log
# ======================
def generate_ranking_log(num_samples=1000):
    products = [f"Product_{i}" for i in range(1, 101)]
    users = [f"User_{i}" for i in range(1, 51)]

    data = []
    positive_count = 0

    for _ in range(num_samples):
        product_id = random.choice(products)
        user_id = random.choice(users)
        price = round(random.uniform(10, 1000), 2)
        score = random.randint(1, 5)
        ranking_pos = random.randint(1, 60)

        # Generate timestamp within last 30 days
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30),
                                               hours=random.randint(0, 23),
                                               minutes=random.randint(0, 59))

        # Generate behaviors with dependency
        clicked = random.random() < 0.1  # 10% click rate
        added_to_cart = clicked and (random.random() < 0.3)
        purchased = added_to_cart and (random.random() < 0.5)

        # Generate comments
        if purchased:
            comments = random.choice([
                "Great product, highly recommend!",
                "Exactly what I needed, works perfectly.",
                "Very satisfied with this purchase."
            ])
        elif added_to_cart:
            comments = random.choice([
                "Considering buying this",
                "Looks good, but still comparing",
                "Added to cart for later"
            ])
        elif clicked:
            comments = random.choice([
                "Interesting product",
                "Not sure about this one",
                "Might come back to this"
            ])
        else:
            comments = ""

        # Track positive samples
        if clicked or added_to_cart or purchased:
            positive_count += 1

        data.append({
            'product_title': f"Awesome {product_id}",
            'product_id': product_id,
            'user_id': user_id,
            'price': price,
            'ranking_score': score,
            'clicked': int(clicked),
            'added_to_cart': int(added_to_cart),
            'purchased': int(purchased),
            'comments': comments,
            'timestamp': timestamp,
            'ranking_position': ranking_pos
        })

    print(f"Generated {positive_count} positive samples ({(positive_count / num_samples) * 100:.1f}%)")
    return pd.DataFrame(data)


ranking_df = generate_ranking_log(1000)


# =================================
# 2. Generate Model Training Samples
# =================================
class TextEmbedder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)

    def forward(self, text_indices):
        # Simple text embedding (no pretrained models)
        embedded = self.embedding(text_indices)
        _, hidden = self.rnn(embedded)
        return hidden.squeeze(0)


def preprocess_data(df, embedder):
    # Build vocabulary from comments
    all_words = " ".join(df['comments']).lower().split()
    vocab = defaultdict(lambda: len(vocab))
    for word in all_words:
        _ = vocab[word]
    vocab_size = len(vocab)

    # Convert text to indices
    def text_to_indices(text):
        return [vocab[word] for word in text.lower().split()]

    # Group by user and sort by time
    user_sequences = defaultdict(list)
    for _, row in df.sort_values('timestamp').iterrows():
        user_sequences[row['user_id']].append({
            'product_id': row['product_id'],
            'price': row['price'],
            'ranking_score': row['ranking_score'],
            'clicked': row['clicked'],
            'added_to_cart': row['added_to_cart'],
            'purchased': row['purchased'],
            'ranking_position': row['ranking_position'],
            'text_indices': text_to_indices(row['comments']),
            'label': int(row['clicked'] or row['added_to_cart'] or row['purchased'])
        })

    # Pad sequences and create features
    max_seq_len = max(len(seq) for seq in user_sequences.values())
    max_text_len = max(len(item['text_indices']) for seq in user_sequences.values() for item in seq)

    sequences = []
    for user, items in user_sequences.items():
        seq_features = []
        seq_labels = []
        text_embeddings = []

        for item in items:
            # Pad text indices
            padded_text = item['text_indices'] + [0] * (max_text_len - len(item['text_indices']))
            text_tensor = torch.LongTensor(padded_text)
            text_embed = embedder(text_tensor.unsqueeze(0))

            seq_features.append([
                item['price'],
                item['ranking_score'],
                item['ranking_position'],
                *text_embed.squeeze().tolist()
            ])
            seq_labels.append(item['label'])

        # Pad sequence
        padded_features = seq_features + [[0] * len(seq_features[0])] * (max_seq_len - len(seq_features))
        padded_labels = seq_labels + [0] * (max_seq_len - len(seq_labels))

        sequences.append({
            'features': torch.FloatTensor(padded_features),
            'labels': torch.FloatTensor(padded_labels),
            'length': len(seq_features)
        })

    return sequences


embedder = TextEmbedder()
train_data = preprocess_data(ranking_df, embedder)


# =====================
# 3. DIEN Model
# =====================
class DIEN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Interest Extractor Layer
        self.gru_interest = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Interest Evolving Layer
        self.gru_evolve = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.aux_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Prediction Layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        # Sort by sequence length
        lengths, sort_idx = torch.sort(lengths, descending=True)
        x = x[sort_idx]

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        # Interest Extractor
        _, hidden_interest = self.gru_interest(packed)

        # Interest Evolving
        packed_evolve = nn.utils.rnn.pack_padded_sequence(
            hidden_interest.transpose(0, 1), lengths, batch_first=True)
        _, hidden_evolve = self.gru_evolve(packed_evolve)

        # Auxiliary loss (optional)
        aux_out = self.aux_net(hidden_interest.squeeze(0))

        # Final prediction
        out = self.fc(hidden_evolve.squeeze(0))

        # Unsorted the output
        _, unsort_idx = torch.sort(sort_idx)
        out = out[unsort_idx]

        return out.squeeze(), aux_out


# =====================
# 4. Training & Evaluation
# =====================
class RankingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['features'], self.data[idx]['labels'], self.data[idx]['length']


def train_model(train_data, epochs=10):
    dataset = RankingDataset(train_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DIEN(input_dim=67)  # 3 numeric + 64 text embedding
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, labels, lengths in loader:
            optimizer.zero_grad()
            preds, aux_preds = model(features, lengths)

            # Main loss
            loss = criterion(preds, labels[:, -1].float())  # Only use last label for prediction

            # Auxiliary loss (optional)
            aux_loss = criterion(aux_preds, labels[:, :-1].float().mean(dim=1))
            total_loss += (loss + 0.5 * aux_loss).item()

            (loss + 0.5 * aux_loss).backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

    return model


def evaluate_model(model, data):
    dataset = RankingDataset(data)
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, lengths in loader:
            preds, _ = model(features, lengths)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels[:, -1].tolist())

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)

    # NDCG calculation (simplified)
    def calculate_ndcg(labels, preds, k=10):
        # Sort by predictions
        sorted_indices = np.argsort(preds)[::-1]
        sorted_labels = np.array(labels)[sorted_indices]

        # Calculate DCG and IDCG
        dcg = sum((2 ** l - 1) / np.log2(i + 2) for i, l in enumerate(sorted_labels[:k]))
        ideal_labels = sorted(labels, reverse=True)
        idcg = sum((2 ** l - 1) / np.log2(i + 2) for i, l in enumerate(ideal_labels[:k]))

        return dcg / idcg if idcg > 0 else 0

    ndcg = calculate_ndcg(all_labels, all_preds)

    print(f"AUC: {auc:.4f}, MSE: {mse:.4f}, NDCG@10: {ndcg:.4f}")


# Train and evaluate
model = train_model(train_data)
evaluate_model(model, train_data)