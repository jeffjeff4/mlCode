####python and pytorch only code
####1. crating ranking log, has 1000 samples, it has
####1) product title
####2) product id
####3) user id
####4) price
####5) ranking score, an int from 1-5, 1 means bad, 5 means good
####6) whether the product is clicked or not
####7) whether the product is added to cart or not
####8) whether the product is purchased or not
####9) comments sharing users' feedback to the product, few natural language sentences, means like it or not, etc
####10) time stam
####11) a product id must get click first, then add to cart, then purchase
####12) positive samples means product has click, or add to cart, or purchase
####13) negative samples means product has no click, no add to cart, no purchase
####14) number of positive samples is much less than negative samples
####15) ranking position, an int from 1-60, 1 means top ranking position in ranking list, 60 means the last ranking position
####16) generate text embedding for comments. please do not use any pretrained model to generate embedding, using pure pytorch to build embedding
####
####
####2. generate model training sample based on:
####1) group by user
####2) sort by time stamp
####3) generate behavior sequence sorted based on time stamp
####4) make sure the training dataset us usable in din model
####5) training sample including text embedding for comments
####
####3. train PCVR model, which is an architecture designed to address challenges in predicting the conversion rate (CVR) by leveraging the click-through rate (CTR), including text embedding for comments
####
####4. evaluate model, using ndcg, mse, auc metrics, etc



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics import ndcg_score, mean_squared_error, roc_auc_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Step 1: Generate ranking log
def generate_ranking_log(num_samples=1000, num_users=50, num_products=200):
    ranking_log = []
    user_ids = [f'user_{i}' for i in range(num_users)]
    product_ids = [f'product_{i}' for i in range(num_products)]
    product_titles = {pid: f'Title for {pid}' for pid in product_ids}
    comments = [
        "love it", "great product", "highly recommend", "not bad", "good value",
        "disappointed", "poor quality", "waste of money", "doesn't work", "too expensive"
    ]

    vocab_size = len(comments)
    embedding_dim = 16
    comment_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    nn.init.uniform_(comment_embedding_layer.weight, -1.0, 1.0)
    comment_embeddings = {comment: comment_embedding_layer(torch.tensor(i)).detach() for i, comment in enumerate(comments)}

    for i in range(num_samples):
        user_id = random.choice(user_ids)
        product_id = random.choice(product_ids)
        price = round(random.uniform(10.0, 500.0), 2)
        ranking_score = random.randint(1, 5)
        ranking_position = random.randint(1, 60)
        timestamp = datetime.now() - timedelta(minutes=random.randint(1, 1000))

        is_clicked = random.random() < 0.2
        is_added_to_cart = is_clicked and random.random() < 0.3
        is_purchased = is_added_to_cart and random.random() < 0.5
        is_positive = is_clicked or is_added_to_cart or is_purchased
        comment_text = random.choice(
            [c for c in comments if "good" in c or "love" in c or "great" in c or "recommend" in c]
            if is_positive else
            [c for c in comments if "bad" in c or "disappointed" in c or "poor" in c or "waste" in c or "doesn't" in c or "expensive" in c]
        )

        entry = {
            'product_title': product_titles[product_id],
            'product_id': product_id,
            'user_id': user_id,
            'price': price,
            'ranking_score': ranking_score,
            'is_clicked': is_clicked,
            'is_added_to_cart': is_added_to_cart,
            'is_purchased': is_purchased,
            'comments': comment_text,
            'timestamp': timestamp,
            'ranking_position': ranking_position
        }
        ranking_log.append(entry)

    print(f"Generated {len(ranking_log)} log entries.")
    return ranking_log, comment_embeddings

# Step 2: PCVRDataset
class PCVRDataset(Dataset):
    def __init__(self, raw_data, comments_embeddings, max_seq_len=20, num_products=200):
        self.max_seq_len = max_seq_len
        self.comments_embeddings = comments_embeddings
        self.num_products = num_products
        self.user_histories = defaultdict(list)
        raw_data.sort(key=lambda x: x['timestamp'])

        for entry in raw_data:
            self.user_histories[entry['user_id']].append(entry)

        self.training_samples = []
        for user_id, history in self.user_histories.items():
            for i, current_item in enumerate(history):
                if current_item['is_clicked']:
                    behavior_sequence = history[:i]
                    self.training_samples.append({
                        'user_id': user_id,
                        'candidate_item': current_item,
                        'behavior_sequence': behavior_sequence,
                        'label': float(current_item['is_purchased'])
                    })

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        sample = self.training_samples[idx]
        user_id = sample['user_id']
        candidate_item = sample['candidate_item']
        candidate_product_id = int(candidate_item['product_id'].split('_')[1])
        candidate_price = candidate_item['price']
        behavior_sequence = sample['behavior_sequence']

        seq_product_ids = []
        seq_prices = []
        seq_comments_emb = []

        for item in behavior_sequence:
            prod_id = int(item['product_id'].split('_')[1])
            if prod_id >= self.num_products:
                prod_id = 0
            seq_product_ids.append(prod_id)
            seq_prices.append(item['price'])
            comment_text = item['comments']
            comment_emb = self.comments_embeddings.get(comment_text, torch.zeros(16))
            seq_comments_emb.append(comment_emb)

        if len(seq_product_ids) > self.max_seq_len:
            seq_product_ids = seq_product_ids[-self.max_seq_len:]
            seq_prices = seq_prices[-self.max_seq_len:]
            seq_comments_emb = seq_comments_emb[-self.max_seq_len:]
        else:
            padding_len = self.max_seq_len - len(seq_product_ids)
            seq_product_ids.extend([self.num_products] * padding_len)
            seq_prices.extend([0.0] * padding_len)
            seq_comments_emb.extend([torch.zeros(16)] * padding_len)

        if max(seq_product_ids) > self.num_products:
            print(f"Warning: Invalid product ID in seq_product_ids: {max(seq_product_ids)}")

        return (
            user_id,
            torch.tensor(candidate_product_id, dtype=torch.long),
            torch.tensor(candidate_price, dtype=torch.float),
            torch.tensor(seq_product_ids, dtype=torch.long),
            torch.tensor(seq_prices, dtype=torch.float),
            torch.stack(seq_comments_emb),
            torch.tensor(sample['label'], dtype=torch.float)
        )

# Step 3: PCVRModel
class PCVRModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim, max_seq_len):
        super(PCVRModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.comment_emb_dim = 16

        self.product_emb = nn.Embedding(num_products + 1, embedding_dim, padding_idx=num_products)

        # Fix: Correct input dimension for attention_net (16 + 16 + 16 + 1 = 49)
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + self.comment_emb_dim + 1, 32),  # candidate + behavior + comment + price
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.prediction_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1 + self.comment_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, candidate_prod_id, candidate_price, hist_prod_ids, hist_prices, hist_comments_emb):
        if torch.any(hist_prod_ids > self.product_emb.num_embeddings - 1):
            print(f"Invalid hist_prod_ids: {hist_prod_ids}")
        if torch.any(candidate_prod_id > self.product_emb.num_embeddings - 1):
            print(f"Invalid candidate_prod_id: {candidate_prod_id}")

        candidate_item_emb = self.product_emb(candidate_prod_id)
        hist_item_embs = self.product_emb(hist_prod_ids)

        candidate_item_emb_repeated = candidate_item_emb.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        attention_input = torch.cat([
            candidate_item_emb_repeated,
            hist_item_embs,
            hist_comments_emb,
            hist_prices.unsqueeze(-1)
        ], dim=-1)

        # Debugging: Print shape
        print(f"attention_input shape: {attention_input.shape}")

        attention_scores = self.attention_net(attention_input).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        user_interest_vector = torch.sum(hist_item_embs * attention_weights.unsqueeze(-1), dim=1)

        final_input = torch.cat([
            candidate_item_emb,
            candidate_price.unsqueeze(-1),
            user_interest_vector,
            hist_comments_emb.mean(dim=1)
        ], dim=-1)

        prediction = self.prediction_net(final_input).squeeze(-1)
        return prediction

# Step 4: Training and Evaluation
def collate_fn(batch):
    if not batch:
        return None
    user_ids = [item[0] for item in batch]
    candidate_prod_ids = torch.stack([item[1] for item in batch])
    candidate_prices = torch.stack([item[2] for item in batch])
    seq_product_ids = torch.stack([item[3] for item in batch])
    seq_prices = torch.stack([item[4] for item in batch])
    seq_comments_emb = torch.stack([item[5] for item in batch])
    labels = torch.stack([item[6] for item in batch])
    return {
        'user_ids': user_ids,
        'candidate_prod_ids': candidate_prod_ids,
        'candidate_prices': candidate_prices,
        'seq_product_ids': seq_product_ids,
        'seq_prices': seq_prices,
        'seq_comments_emb': seq_comments_emb,
        'labels': labels
    }

# Generate data
ranking_log, comment_embeddings = generate_ranking_log()

# Create dataset and dataloader
dataset = PCVRDataset(ranking_log, comment_embeddings, max_seq_len=20, num_products=200)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialize model
model = PCVRModel(num_users=50, num_products=200, embedding_dim=16, max_seq_len=20)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    for batch in train_loader:
        if batch is None:
            continue
        optimizer.zero_grad()
        outputs = model(
            batch['candidate_prod_ids'],
            batch['candidate_prices'],
            batch['seq_product_ids'],
            batch['seq_prices'],
            batch['seq_comments_emb']
        )
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    print(f"Epoch {epoch+1}, Loss: {total_loss / batch_count if batch_count > 0 else 0:.4f}")

# Evaluation
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_ranking_positions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                continue
            outputs = model(
                batch['candidate_prod_ids'],
                batch['candidate_prices'],
                batch['seq_product_ids'],
                batch['seq_prices'],
                batch['seq_comments_emb']
            )
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            indices = [loader.dataset.indices[i] for i in range(len(batch['labels']))]
            ranking_positions = [dataset.training_samples[i]['candidate_item']['ranking_position'] for i in indices]
            all_ranking_positions.extend(ranking_positions)

    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    mse = mean_squared_error(all_labels, all_preds) if len(all_preds) > 0 else 0.0

    ndcg_scores = []
    for i in range(0, len(all_preds), 32):
        batch_preds = all_preds[i:i+32]
        batch_positions = np.array(all_ranking_positions[i:i+32], dtype=np.float32)
        if len(batch_preds) > 1 and len(batch_positions) > 1:
            true_relevance = 1 / np.log2(batch_positions + 1)
            ndcg = ndcg_score([true_relevance], [batch_preds])
            ndcg_scores.append(ndcg)
    ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return {'AUC': auc, 'MSE': mse, 'NDCG': ndcg}

# Run evaluation
metrics = evaluate(model, test_loader)
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")