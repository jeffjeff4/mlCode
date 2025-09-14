import torch
import random
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


# --- 1. Creating Ranking Log ---

def generate_text_embedding_pytorch_only(text, vocab, embedding_dim=50):
    """
    Generates a very basic character-level or word-level embedding using pure PyTorch.
    This is a highly simplified approach and will not capture semantic meaning
    like pre-trained models. For demonstration purposes only.
    """
    if not text:
        return torch.zeros(embedding_dim)

    # Simple word-level tokenization
    tokens = text.lower().split()
    if not tokens:
        return torch.zeros(embedding_dim)

    # Create one-hot like vectors for each token based on the vocab
    # Then average them to get a crude sentence embedding
    embedding_list = []
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)  # Add new word to vocab
        idx = vocab[token]
        # Create a "one-hot" vector and then a simple linear layer for embedding
        # This is a conceptual representation of learning an embedding from scratch
        # In a real scenario, this would be a trainable nn.Embedding layer
        one_hot = torch.zeros(len(vocab))
        one_hot[idx] = 1.0
        # Simulate a very simple linear transformation as a 'learned' embedding
        # This is not how nn.Embedding works internally, but illustrates the idea
        # For actual training, you'd use nn.Embedding(num_embeddings, embedding_dim)
        # and pass token indices.
        # Here, we're just creating *some* fixed-size vector for each token
        # based on its index. In a true trainable model, the weights would adapt.

        # For this demonstration, we'll just use a simple fixed vector for each word.
        # In a real PyTorch model, you'd use nn.Embedding and pass the integer ID.
        # Since we are not training a model here for the embedding part,
        # we'll create a random but consistent vector for each word.
        # If we were to train, the embedding layer itself would be trained.

        # Let's create a fixed vector for each word for consistency in this non-training scenario
        # In a real model, this would be self.embedding(token_id) where self.embedding is nn.Embedding

        # This is a placeholder for a 'learned' embedding.
        # For a truly 'pure PyTorch' way without pre-trained, one would initialize
        # nn.Embedding and pass indices to it. Since we are not training a separate
        # embedding model here, we will just simulate a fixed embedding for each word.

        # A simple hash-based "embedding" for demonstration without training
        # For a real scenario, you'd have an actual trainable nn.Embedding layer.
        seed = hash(token) % (10 ** 9)  # Ensure seed is within a reasonable range
        rng = np.random.default_rng(seed)
        word_embedding = torch.from_numpy(rng.random(embedding_dim)).float()
        embedding_list.append(word_embedding)

    # Average word embeddings to get sentence embedding
    if embedding_list:
        return torch.stack(embedding_list).mean(dim=0)
    else:
        return torch.zeros(embedding_dim)


def create_ranking_log(num_samples=1000):
    ranking_log = []
    product_ids = [f"prod_{i:04d}" for i in range(200)]  # 200 unique products
    user_ids = [f"user_{i:03d}" for i in range(100)]  # 100 unique users

    # Store user's last interaction time to ensure chronological order for a user
    user_last_timestamp = defaultdict(lambda: datetime(2023, 1, 1, 0, 0, 0))

    # Global vocabulary for text embeddings
    comments_vocab = {}
    embedding_dim = 50  # Dimension for text embeddings

    for i in range(num_samples):
        product_id = random.choice(product_ids)
        user_id = random.choice(user_ids)

        # Ensure timestamps are somewhat sequential for a user
        base_time = user_last_timestamp[user_id]
        time_stamp = base_time + timedelta(minutes=random.randint(1, 60))
        user_last_timestamp[user_id] = time_stamp

        product_title = f"Product Title for {product_id}"
        price = round(random.uniform(10.0, 500.0), 2)
        ranking_score = random.randint(1, 5)  # 1 bad, 5 good
        ranking_position = random.randint(1, 60)

        # Simulate click, add to cart, purchase logic
        is_clicked = False
        is_added_to_cart = False
        is_purchased = False

        # Higher ranking score and lower position -> more likely to be clicked/interacted
        interaction_probability = (ranking_score / 5.0) * (1 - (ranking_position / 60.0))

        if random.random() < interaction_probability * 0.7:  # Higher chance of click
            is_clicked = True
            if random.random() < interaction_probability * 0.5:  # If clicked, chance to add to cart
                is_added_to_cart = True
                if random.random() < interaction_probability * 0.3:  # If added, chance to purchase
                    is_purchased = True

        # Generate comments based on purchase/ranking score
        comments = ""
        if is_purchased or ranking_score >= 4:
            comments = random.choice([
                "Absolutely love this product! Highly recommend.",
                "Great quality and fast delivery. Very satisfied.",
                "Exceeded my expectations, worth every penny.",
                "Fantastic item, performs as described.",
                "Very happy with my purchase, will buy again."
            ])
        elif is_clicked or ranking_score <= 2:
            comments = random.choice([
                "It's okay, nothing special.",
                "Could be better, a bit disappointed.",
                "Average product for the price.",
                "Not bad, but not great either.",
                "Had higher hopes, but it's usable."
            ])
        else:
            comments = random.choice([
                "Interesting product.",
                "Haven't fully explored yet.",
                "Seems decent.",
                "No strong feelings either way.",
                "Looks as advertised."
            ])

        comment_embedding = generate_text_embedding_pytorch_only(comments, comments_vocab, embedding_dim)

        # Determine positive/negative sample
        is_positive = is_clicked or is_added_to_cart or is_purchased

        ranking_log.append({
            "product_title": product_title,
            "product_id": product_id,
            "user_id": user_id,
            "price": price,
            "ranking_score": ranking_score,
            "is_clicked": is_clicked,
            "is_added_to_cart": is_added_to_cart,
            "is_purchased": is_purchased,
            "comments": comments,
            "comment_embedding": comment_embedding,  # PyTorch tensor
            "time_stamp": time_stamp,
            "is_positive": is_positive,
            "ranking_position": ranking_position
        })

    # Ensure positive samples are much less than negative samples (post-hoc adjustment if needed)
    # This might naturally happen with the probabilities, but we can verify.
    positive_count = sum(1 for sample in ranking_log if sample["is_positive"])
    negative_count = num_samples - positive_count
    print(f"Generated {positive_count} positive samples and {negative_count} negative samples.")

    return ranking_log, comments_vocab, embedding_dim


# Generate the data
ranking_data, vocab, embed_dim = create_ranking_log(num_samples=1000)

# Display a sample entry and embedding shape
print("\n--- Sample Ranking Log Entry ---")
if ranking_data:
    sample_entry = ranking_data[0]
    for k, v in sample_entry.items():
        if k != "comment_embedding":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v.shape} (PyTorch Tensor)")
    print(f"Comment embedding example (first 5 values): {sample_entry['comment_embedding'][:5].tolist()}")
    print(f"Total vocabulary size for comments: {len(vocab)}")
    print(f"Embedding dimension: {embed_dim}")

import torch
import random
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Assuming ranking_data, vocab, embed_dim are already generated from Part 1

# --- 2. Generate Model Training Sample ---

def prepare_dien_dataset(ranking_data):
    user_sequences = defaultdict(list)
    for record in ranking_data:
        user_sequences[record['user_id']].append(record)

    # Group by user and sort by time stamp
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x['time_stamp'])

    # Structure for DIEN-like input
    # Each sample will represent a target item for a user, along with their historical behavior
    dien_samples = []

    # Max sequence length for historical behavior
    MAX_HIST_LEN = 10  # Define a maximum historical sequence length

    # Collect unique product_ids to create a mapping for embeddings
    all_product_ids = sorted(list(set(d['product_id'] for d in ranking_data)))
    product_id_to_idx = {pid: i for i, pid in enumerate(all_product_ids)}
    num_products = len(all_product_ids)

    # Collect unique user_ids for user embeddings
    all_user_ids = sorted(list(set(d['user_id'] for d in ranking_data)))
    user_id_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}
    num_users = len(all_user_ids)

    # For comments, the embedding is already a tensor, no need for ID mapping

    for user_id, records in user_sequences.items():
        user_idx = user_id_to_idx[user_id]

        # Each record in a user's sequence can be a potential target item
        # We need to construct the historical behavior *before* that target item
        for i in range(len(records)):
            target_item = records[i]

            # Historical behavior: all items before the current target item
            history_records = records[:i]

            # Limit historical sequence length
            history_records = history_records[-MAX_HIST_LEN:]

            hist_product_ids = [product_id_to_idx[r['product_id']] for r in history_records]
            hist_is_clicked = [int(r['is_clicked']) for r in history_records]
            hist_is_added_to_cart = [int(r['is_added_to_cart']) for r in history_records]
            hist_is_purchased = [int(r['is_purchased']) for r in history_records]
            hist_comment_embeddings = [r['comment_embedding'] for r in history_records]

            # Pad sequences if shorter than MAX_HIST_LEN
            current_hist_len = len(hist_product_ids)
            if current_hist_len < MAX_HIST_LEN:
                # Pad with zeros for ID features and zero vectors for embeddings
                pad_len = MAX_HIST_LEN - current_hist_len
                hist_product_ids.extend([0] * pad_len)  # Assuming 0 is a padding ID
                hist_is_clicked.extend([0] * pad_len)
                hist_is_added_to_cart.extend([0] * pad_len)
                hist_is_purchased.extend([0] * pad_len)
                # Pad comment embeddings with zero tensors
                zero_embedding = torch.zeros(embed_dim)
                hist_comment_embeddings.extend([zero_embedding] * pad_len)

            # Target item features
            target_product_id = product_id_to_idx[target_item['product_id']]
            target_price = target_item['price']
            target_ranking_position = target_item['ranking_position']
            target_comment_embedding = target_item['comment_embedding']

            # Label
            label = float(target_item['is_clicked'] or target_item['is_added_to_cart'] or target_item[
                'is_purchased'])  # Binary classification for click/conversion

            dien_samples.append({
                'user_id': user_idx,
                'target_product_id': target_product_id,
                'target_price': target_price,
                'target_ranking_position': target_ranking_position,
                'target_comment_embedding': target_comment_embedding,
                'hist_product_ids': torch.LongTensor(hist_product_ids),
                'hist_is_clicked': torch.LongTensor(hist_is_clicked),
                'hist_is_added_to_cart': torch.LongTensor(hist_is_added_to_cart),
                'hist_is_purchased': torch.LongTensor(hist_is_purchased),
                'hist_comment_embeddings': torch.stack(hist_comment_embeddings),
                # Stack list of tensors into a single tensor
                'hist_len': current_hist_len,  # Actual length before padding
                'label': torch.FloatTensor([label])
            })

    print(f"\nGenerated {len(dien_samples)} DIEN-like samples.")
    return dien_samples, num_products, num_users, embed_dim, MAX_HIST_LEN


# Prepare the dataset
dien_dataset, num_products, num_users, embed_dim, max_hist_len = prepare_dien_dataset(ranking_data)


# Create a PyTorch Dataset and DataLoader
class DIENDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (sample['user_id'],
                sample['target_product_id'],
                sample['target_price'],
                sample['target_ranking_position'],
                sample['target_comment_embedding'],
                sample['hist_product_ids'],
                sample['hist_is_clicked'],
                sample['hist_is_added_to_cart'],
                sample['hist_is_purchased'],
                sample['hist_comment_embeddings'],
                sample['hist_len'],
                sample['label'])


dien_pytorch_dataset = DIENDataset(dien_dataset)
dien_dataloader = DataLoader(dien_pytorch_dataset, batch_size=32, shuffle=True)

print(f"\n--- Sample DIEN-like Training Sample (first batch from DataLoader) ---")
for i, batch in enumerate(dien_dataloader):
    if i == 0:
        user_ids, target_product_ids, target_prices, target_ranking_positions, \
            target_comment_embeddings, hist_product_ids, hist_is_clicked, \
            hist_is_added_to_cart, hist_is_purchased, hist_comment_embeddings, \
            hist_lens, labels = batch

        print(f"Batch user_ids shape: {user_ids.shape}")
        print(f"Batch target_product_ids shape: {target_product_ids.shape}")
        print(f"Batch target_prices shape: {target_prices.shape}")
        print(f"Batch target_ranking_positions shape: {target_ranking_positions.shape}")
        print(f"Batch target_comment_embeddings shape: {target_comment_embeddings.shape}")
        print(f"Batch hist_product_ids shape: {hist_product_ids.shape}")
        print(f"Batch hist_is_clicked shape: {hist_is_clicked.shape}")
        print(f"Batch hist_is_added_to_cart shape: {hist_is_added_to_cart.shape}")
        print(f"Batch hist_is_purchased shape: {hist_is_purchased.shape}")
        print(f"Batch hist_comment_embeddings shape: {hist_comment_embeddings.shape}")
        print(f"Batch hist_lens shape: {hist_lens.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break

import torch
import torch.nn as nn
import torch.nn.functional as F


# Assuming num_products, num_users, embed_dim, max_hist_len from Part 2 are available

# --- 3. Train DIEN Model (Simplified, Conceptual) ---

class DIENConceptual(nn.Module):
    def __init__(self, num_users, num_products, embed_dim, max_hist_len):
        super(DIENConceptual, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.product_embedding = nn.Embedding(num_products, embed_dim)

        # Embedding for historical interaction types (click, add_to_cart, purchase)
        # Using 2 classes (0 or 1) for each type
        self.click_embedding = nn.Embedding(2, embed_dim // 2)  # Embed 0/1 to a smaller dim
        self.cart_embedding = nn.Embedding(2, embed_dim // 2)
        self.purchase_embedding = nn.Embedding(2, embed_dim // 2)

        # GRU for historical sequence (Interest Extractor in DIEN)
        # Input to GRU: Concatenation of (product_embedding, click_embedding, cart_embedding, purchase_embedding, comment_embedding)
        gru_input_dim = embed_dim + (embed_dim // 2) * 3 + embed_dim  # Product + 3 behaviors + Comment
        self.gru = nn.GRU(gru_input_dim, embed_dim, batch_first=True)  # Output hidden state is user interest

        # Attention layer (simplified, not full DIEN target attention)
        # We will use simple concatenation for now, as full attention is complex

        # Prediction layer
        # Input to prediction: User embedding + Target product embedding + User interest (from GRU)
        # + Target price + Target ranking position + Target comment embedding
        prediction_input_dim = embed_dim + embed_dim + embed_dim + 1 + 1 + embed_dim  # User + Target Prod + GRU Hidden + Price + Rank Pos + Target Comment Embed
        self.fc_layers = nn.Sequential(
            nn.Linear(prediction_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # For binary classification (click/conversion prediction)
        )

    def forward(self, user_id, target_product_id, target_price, target_ranking_position,
                target_comment_embedding, hist_product_ids, hist_is_clicked,
                hist_is_added_to_cart, hist_is_purchased, hist_comment_embeddings, hist_len):
        # Embeddings
        user_emb = self.user_embedding(user_id)
        target_product_emb = self.product_embedding(target_product_id)

        hist_product_emb = self.product_embedding(hist_product_ids)
        hist_click_emb = self.click_embedding(hist_is_clicked)
        hist_cart_emb = self.cart_embedding(hist_is_added_to_cart)
        hist_purchase_emb = self.purchase_embedding(hist_is_purchased)

        # Concatenate historical item features and behavior indicators
        # hist_comment_embeddings is already a tensor of shape (batch_size, max_hist_len, embed_dim)
        hist_combined_emb = torch.cat([
            hist_product_emb,
            hist_click_emb,
            hist_cart_emb,
            hist_purchase_emb,
            hist_comment_embeddings
        ], dim=-1)

        # GRU for interest evolution (DIEN's Interest Extractor)
        # Need to handle variable sequence lengths if not padded
        # For simplicity with padding, we'll just pass to GRU directly.
        # In a more robust DIEN, you'd use nn.utils.rnn.pack_padded_sequence and pad_packed_sequence

        # Initial hidden state for GRU (usually zero-initialized)
        batch_size = hist_product_ids.size(0)
        h0 = torch.zeros(1, batch_size, self.gru.hidden_size).to(hist_combined_emb.device)

        # Pass through GRU
        # output: (batch_size, seq_len, hidden_size)
        # hn: (num_layers * num_directions, batch_size, hidden_size)
        output, hn = self.gru(hist_combined_emb, h0)

        # Take the last hidden state as the user's evolved interest
        # hn is (1, batch_size, hidden_size) if num_layers=1, num_directions=1
        user_interest = hn.squeeze(0)  # Shape: (batch_size, hidden_size)

        # For the prediction layer, concatenate all relevant features
        # Features: User embedding, Target product embedding, User interest,
        #           Target price, Target ranking position, Target comment embedding

        # Ensure target_price and target_ranking_position are float tensors and unsqueeze for concatenation
        target_price = target_price.unsqueeze(1).float()
        target_ranking_position = target_ranking_position.unsqueeze(1).float()

        # Concatenate all features
        combined_features = torch.cat([
            user_emb,
            target_product_emb,
            user_interest,
            target_price,
            target_ranking_position,
            target_comment_embedding
        ], dim=-1)

        # Pass through prediction layers
        prediction = self.fc_layers(combined_features)

        return prediction


# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DIENConceptual(num_users, num_products, embed_dim, max_hist_len).to(device)

# Loss and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for click/conversion prediction
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop (Simplified) ---
print("\n--- Starting Conceptual DIEN Training ---")
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dien_dataloader):
        user_ids, target_product_ids, target_prices, target_ranking_positions, \
            target_comment_embeddings, hist_product_ids, hist_is_clicked, \
            hist_is_added_to_cart, hist_is_purchased, hist_comment_embeddings, \
            hist_lens, labels = [t.to(device) for t in batch]  # Move data to device

        optimizer.zero_grad()

        predictions = model(user_ids, target_product_ids, target_prices, target_ranking_positions,
                            target_comment_embeddings, hist_product_ids, hist_is_clicked,
                            hist_is_added_to_cart, hist_is_purchased, hist_comment_embeddings, hist_lens)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dien_dataloader):.4f}")

print("--- Conceptual DIEN Training Finished ---")

import torch
from sklearn.metrics import roc_auc_score, mean_squared_error


# For NDCG, we typically need a separate library or a custom implementation
# as it's not directly in sklearn for ranking lists as we need them.
# We'll provide a conceptual outline for NDCG.

# Assuming model, dien_dataloader, device are available from Part 3

# --- 4. Evaluate Model ---

def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in dataloader:
            user_ids, target_product_ids, target_prices, target_ranking_positions, \
                target_comment_embeddings, hist_product_ids, hist_is_clicked, \
                hist_is_added_to_cart, hist_is_purchased, hist_comment_embeddings, \
                hist_lens, labels = [t.to(device) for t in batch]

            predictions = model(user_ids, target_product_ids, target_prices, target_ranking_positions,
                                target_comment_embeddings, hist_product_ids, hist_is_clicked,
                                hist_is_added_to_cart, hist_is_purchased, hist_comment_embeddings, hist_lens)

            all_predictions.extend(predictions.cpu().squeeze().tolist())
            all_labels.extend(labels.cpu().squeeze().tolist())

    # Convert to numpy arrays for sklearn metrics
    all_predictions_np = np.array(all_predictions)
    all_labels_np = np.array(all_labels)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels_np, all_predictions_np)
    except ValueError:
        auc = "N/A (only one class present in labels)"  # Handle cases where AUC cannot be computed

    # Calculate MSE
    mse = mean_squared_error(all_labels_np, all_predictions_np)

    # Calculate NDCG (Conceptual Outline - Requires specific data structure)
    # NDCG is typically calculated per query (e.g., for a user's search session)
    # and requires a list of predicted scores for a list of items and their true relevance.
    # Our current data provides one (target_item, history) pair per sample.
    # To compute NDCG meaningfully, you would need to:
    # 1. Group predictions and true labels by the "query" (e.g., user session or current user's interaction).
    # 2. For each query, get all candidate items and their predicted scores.
    # 3. Get the true relevance for these candidate items (e.g., 1 if clicked/purchased, 0 otherwise).
    # 4. Sort the candidates by predicted score and calculate DCG.
    # 5. Calculate IDCG (Ideal DCG) by sorting by true relevance.
    # 6. NDCG = DCG / IDCG.

    # Placeholder for NDCG calculation - not directly computable with current batching for single target items.
    ndcg_val = "Requires specific per-query/user-session ranking data structure."

    print(f"\n--- Model Evaluation Results ---")
    print(f"AUC: {auc:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"NDCG: {ndcg_val}")


# Evaluate the model
evaluate_model(model, dien_dataloader, device)

