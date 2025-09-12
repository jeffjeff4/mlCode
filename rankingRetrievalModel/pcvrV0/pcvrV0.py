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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.metrics import ndcg_score, mean_squared_error, roc_auc_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Step 1: Generate ranking log with 1000 samples
def generate_ranking_log(num_samples=1000):
    data = []
    product_titles = [f"Product {i}" for i in range(1, 201)]
    product_ids = list(range(1, 201))
    user_ids = list(range(1, 101))
    positive_ratio = 0.1

    for _ in range(num_samples):
        product_title = random.choice(product_titles)
        product_id = random.choice(product_ids)
        user_id = random.choice(user_ids)
        price = round(random.uniform(10, 100), 2)
        ranking_score = random.randint(1, 5)
        ranking_position = random.randint(1, 60)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))

        is_positive = random.random() < positive_ratio
        clicked = False
        added_to_cart = False
        purchased = False
        comments = "No comment"

        if is_positive:
            clicked = True
            added_to_cart = random.random() < 0.7
            purchased = added_to_cart and random.random() < 0.5
            comments = random.choice(["Great product!", "Love it!", "Highly recommend!", "Not bad."])
        else:
            comments = random.choice(["Not interested.", "Too expensive.", "Bad quality.", "Irrelevant."])

        data.append({
            'product_title': product_title,
            'product_id': product_id,
            'user_id': user_id,
            'price': price,
            'ranking_score': ranking_score,
            'clicked': clicked,
            'added_to_cart': added_to_cart,
            'purchased': purchased,
            'comments': comments,
            'timestamp': timestamp,
            'ranking_position': ranking_position
        })

    return data

# Generate data
ranking_log = generate_ranking_log()

# Generate text embeddings for comments (pure PyTorch)
vocab = set()
for entry in ranking_log:
    vocab.update(entry['comments'].split())
vocab = list(vocab)
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}

embedding_dim = 10
word_embeddings = nn.Embedding(vocab_size, embedding_dim)
nn.init.uniform_(word_embeddings.weight, -1.0, 1.0)

def get_comment_embedding(comment):
    words = comment.split()
    indices = [word_to_idx.get(word, 0) for word in words]
    if not indices:
        return torch.zeros(embedding_dim)
    emb = word_embeddings(torch.tensor(indices))
    return emb.mean(dim=0).detach()

for entry in ranking_log:
    entry['comment_embedding'] = get_comment_embedding(entry['comments'])

# Step 2: Generate training samples for DIN-like model
from collections import defaultdict

user_groups = defaultdict(list)
for entry in ranking_log:
    user_groups[entry['user_id']].append(entry)

for user_id, entries in user_groups.items():
    entries.sort(key=lambda x: x['timestamp'])

training_samples = []
for user_id, entries in user_groups.items():
    for i in range(1, len(entries)):
        sequence = entries[:i]
        target = entries[i]
        seq_embeddings = [e['comment_embedding'] for e in sequence]
        target_embedding = target['comment_embedding']
        label = 1 if target['purchased'] else 0
        ctr_label = 1 if target['clicked'] else 0
        ranking_position = target['ranking_position']

        if len(seq_embeddings) > 0:
            training_samples.append({
                'user_id': user_id,
                'seq_embeddings': seq_embeddings,
                'target_embedding': target_embedding,
                'label': label,
                'ctr_label': ctr_label,
                'ranking_position': ranking_position
            })

if len(training_samples) < 10:
    print("Error: Too few training samples. Check data generation.")
    exit(1)

# Split train/test
train_size = int(0.8 * len(training_samples))
train_samples = training_samples[:train_size]
test_samples = training_samples[train_size:]

# Custom collate to pad sequences
def collate_fn(batch):
    if not batch:
        return None
    max_seq_len = max(len(sample['seq_embeddings']) for sample in batch)
    padded_seq_emb = []
    for sample in batch:
        seq_emb = sample['seq_embeddings']
        seq_len = len(seq_emb)
        pad_len = max_seq_len - seq_len
        padded_emb = torch.cat([torch.zeros(pad_len, embedding_dim), torch.stack(seq_emb)] if seq_len > 0 else torch.zeros(max_seq_len, embedding_dim))
        padded_seq_emb.append(padded_emb)

    return {
        'seq_embeddings': torch.stack(padded_seq_emb),
        'target_embedding': torch.stack([sample['target_embedding'] for sample in batch]),
        'label': torch.tensor([sample['label'] for sample in batch]).float().unsqueeze(1),
        'ctr_label': torch.tensor([sample['ctr_label'] for sample in batch]).float().unsqueeze(1),
        'ranking_position': torch.tensor([sample['ranking_position'] for sample in batch]).float()
    }

# Dataset
class DinDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

train_dataset = DinDataset(train_samples)
test_dataset = DinDataset(test_samples)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Step 3: Define PCVR model
class PCVRModel(nn.Module):
    def __init__(self, embedding_dim):
        super(PCVRModel, self).__init__()
        self.fc_seq = nn.Linear(embedding_dim, 32)
        self.fc_target = nn.Linear(embedding_dim, 32)
        self.fc_combined = nn.Linear(64, 16)
        self.ctr_head = nn.Linear(16, 1)
        self.cvr_head = nn.Linear(16, 1)

    def forward(self, seq_emb, target_emb):
        seq_emb_mean = seq_emb.mean(dim=1)
        seq_feat = torch.relu(self.fc_seq(seq_emb_mean))
        target_feat = torch.relu(self.fc_target(target_emb))
        combined = torch.cat([seq_feat, target_feat], dim=-1)
        hidden = torch.relu(self.fc_combined(combined))
        ctr_pred = torch.sigmoid(self.ctr_head(hidden))
        cvr_pred = torch.sigmoid(self.cvr_head(hidden))
        return ctr_pred, cvr_pred

model = PCVRModel(embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            print(f"Epoch {epoch+1}, Batch {batch_idx}: Empty batch, skipping")
            continue

        seq_emb = batch['seq_embeddings']
        target_emb = batch['target_embedding']
        label = batch['label']
        ctr_label = batch['ctr_label']

        optimizer.zero_grad()
        ctr_pred, cvr_pred = model(seq_emb, target_emb)
        loss_ctr = criterion(ctr_pred, ctr_label)
        loss_cvr = criterion(cvr_pred, label)
        loss = loss_ctr + loss_cvr
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Step 4: Evaluate model
def evaluate(model, loader):
    model.eval()
    all_ctr_preds = []
    all_cvr_preds = []
    all_ctr_labels = []
    all_cvr_labels = []
    all_ranking_positions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch is None:
                print(f"Evaluation, Batch {batch_idx}: Empty batch, skipping")
                continue
            seq_emb = batch['seq_embeddings']
            target_emb = batch['target_embedding']
            label = batch['label'].squeeze().cpu().numpy()
            ctr_label = batch['ctr_label'].squeeze().cpu().numpy()
            ranking_position = batch['ranking_position'].cpu().numpy()

            # Debugging: Check type of ranking_position
            print(f"Batch {batch_idx}, ranking_position type: {type(ranking_position)}, shape: {ranking_position.shape}")

            ctr_pred, cvr_pred = model(seq_emb, target_emb)
            all_ctr_preds.extend(ctr_pred.squeeze().cpu().numpy())
            all_cvr_preds.extend(cvr_pred.squeeze().cpu().numpy())
            all_ctr_labels.extend(ctr_label)
            all_cvr_labels.extend(label)
            all_ranking_positions.extend(ranking_position)

    # AUC
    auc_ctr = roc_auc_score(all_ctr_labels, all_ctr_preds) if len(set(all_ctr_labels)) > 1 else 0.5
    auc_cvr = roc_auc_score(all_cvr_labels, all_cvr_preds) if len(set(all_cvr_labels)) > 1 else 0.5

    # MSE
    mse_ctr = mean_squared_error(all_ctr_labels, all_ctr_preds) if len(all_ctr_preds) > 0 else 0.0
    mse_cvr = mean_squared_error(all_cvr_labels, all_cvr_preds) if len(all_cvr_preds) > 0 else 0.0

    # NDCG: Fix for list-to-array conversion
    ndcg_scores = []
    for i in range(0, len(all_cvr_preds), 32):
        batch_preds = all_cvr_preds[i:i+32]
        batch_positions = np.array(all_ranking_positions[i:i+32])  # Explicitly convert to NumPy array
        if len(batch_preds) > 1 and len(batch_positions) > 1:
            true_relevance = 1 / np.log2(batch_positions + 1)  # Element-wise operation
            pred_scores = np.array(batch_preds)
            ndcg = ndcg_score([true_relevance], [pred_scores])
            ndcg_scores.append(ndcg)
    ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    return {
        'AUC_CTR': auc_ctr,
        'AUC_CVR': auc_cvr,
        'MSE_CTR': mse_ctr,
        'MSE_CVR': mse_cvr,
        'NDCG': ndcg
    }

metrics = evaluate(model, test_loader)
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")