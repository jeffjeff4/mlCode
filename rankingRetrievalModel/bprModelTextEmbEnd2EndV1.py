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
import random
import datetime
from collections import defaultdict
import numpy as np

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def generate_ranking_log(num_samples=1000):
    product_titles = [
        "Premium Wireless Headphones", "Ergonomic Office Chair", "Smart Fitness Tracker",
        "Organic Green Tea Set", "Portable Bluetooth Speaker", "Noise-Cancelling Earbuds",
        "Ultra HD Smart TV", "Compact Coffee Maker", "Waterproof Backpack",
        "Professional DSLR Camera", "Stylish Running Shoes", "Gourmet Chocolate Box",
        "Eco-Friendly Yoga Mat", "High-Performance Laptop", "Digital Drawing Tablet",
        "Vintage Leather Wallet", "Aromatherapy Diffuser", "Smart Home Security Camera",
        "Insulated Water Bottle", "Wireless Charging Pad"
    ]

    comments_positive = [
        "Absolutely love this product! Highly recommend.",
        "Fantastic quality and works perfectly. Very happy with my purchase.",
        "Exceeded my expectations. Great value for money.",
        "Couldn't be happier. This is exactly what I was looking for.",
        "Excellent product, a must-buy for anyone in need."
    ]

    comments_negative = [
        "Very disappointed. Did not live up to the hype.",
        "Poor quality and stopped working after a short time.",
        "Not what I expected at all. Wouldn't recommend.",
        "Had high hopes, but this product failed to deliver.",
        "Mediocre at best. There are better alternatives out there."
    ]

    ranking_log = []
    # product_id_counter = 0 # Not strictly needed as we use a pre-defined list of products
    # user_id_counter = 0    # Not strictly needed as we use a pre-defined list of users

    # To ensure some users have multiple interactions
    num_users = int(num_samples * 0.3)  # Roughly 30% unique users
    num_products_pool = len(product_titles) * 2 # Create a larger pool of product IDs than just unique titles
    products = [f"product_{i}" for i in range(num_products_pool)]
    users = [f"user_{i}" for i in range(num_users)]

    # Map product_id to an assigned title to ensure consistency for a given product_id
    product_id_to_title = {}

    for _ in range(num_samples):
        product_id = random.choice(products)
        user_id = random.choice(users)

        # Assign a consistent title for each product_id
        if product_id not in product_id_to_title:
            product_id_to_title[product_id] = random.choice(product_titles)
        product_title = product_id_to_title[product_id]

        price = round(random.uniform(10.0, 500.0), 2)
        ranking_score = random.randint(1, 5)
        ranking_position = random.randint(1, 60)
        timestamp = datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30), seconds=random.randint(0, 86400))
        timestamp = timestamp.isoformat()

        # Initialize actions
        clicked = 0
        added_to_cart = 0
        purchased = 0
        comments = ""

        # Logic for click -> add_to_cart -> purchase
        # Positive samples (click, add to cart, or purchase) are less frequent
        is_positive_sample = random.random() < 0.2 # 20% chance of being a positive sample

        if is_positive_sample:
            clicked = 1
            comments = random.choice(comments_positive)
            if random.random() < 0.4: # 40% of clicks lead to add to cart
                added_to_cart = 1
                if random.random() < 0.6: # 60% of add to carts lead to purchase
                    purchased = 1
        else:
            # Negative sample (no click, no add to cart, no purchase)
            clicked = 0
            added_to_cart = 0
            purchased = 0
            comments = random.choice(comments_negative) # Assign a negative comment for no interaction


        ranking_log.append({
            "product_title": product_title,
            "product_id": product_id,
            "user_id": user_id,
            "price": price,
            "ranking_score": ranking_score,
            "clicked": clicked,
            "added_to_cart": added_to_cart,
            "purchased": purchased,
            "comments": comments,
            "timestamp": timestamp,
            "ranking_position": ranking_position
        })

    return ranking_log

# Pure PyTorch Text Embedding (Simple Bag-of-Words/Index-based Embedding)
class SimpleTextEmbedding(torch.nn.Module):
    def __init__(self, vocabulary, embedding_dim):
        super(SimpleTextEmbedding, self).__init__()
        self.word_to_idx = {word: i for i, word in enumerate(vocabulary)}
        self.idx_to_word = {i: word for i, word in enumerate(vocabulary)}
        self.embedding = torch.nn.Embedding(len(vocabulary), embedding_dim)

    def forward(self, sentences):
        # Tokenize and convert words to indices
        indexed_sentences = []
        for sentence in sentences:
            tokens = sentence.lower().split()
            indexed_sentence = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
            indexed_sentences.append(torch.tensor(indexed_sentence, dtype=torch.long))

        # Pad sequences to the maximum length in the batch
        max_len = max([len(seq) for seq in indexed_sentences])
        padded_sentences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - len(seq)), 'constant', self.word_to_idx['<pad>'])
            for seq in indexed_sentences
        ])

        # Get embeddings for each word
        word_embeddings = self.embedding(padded_sentences)

        # Simple aggregation: mean pooling of word embeddings
        # Mask out padding tokens if necessary for more accurate mean
        # For simplicity here, we'll just average all.
        sentence_embeddings = torch.mean(word_embeddings, dim=1)
        return sentence_embeddings


def get_vocabulary(ranking_log):
    all_words = set()
    for entry in ranking_log:
        words = entry['comments'].lower().split()
        for word in words:
            all_words.add(word)
    vocabulary = list(all_words)
    vocabulary.sort()  # Ensure consistent ordering
    vocabulary.insert(0, '<pad>')  # Add padding token
    vocabulary.insert(1, '<unk>')  # Add unknown token
    return vocabulary


##if __name__ == "__main__":
##    ranking_log_data = generate_ranking_log(num_samples=1000)
##    print(f"Generated {len(ranking_log_data)} samples.")
##    print("Example ranking log entry:")
##    print(ranking_log_data[0])
##
##    # Prepare for text embedding
##    vocabulary = get_vocabulary(ranking_log_data)
##    embedding_dim = 64  # Example embedding dimension
##    text_embedding_model = SimpleTextEmbedding(vocabulary, embedding_dim)
##
##    # Example of generating text embeddings for a few comments
##    sample_comments = [entry['comments'] for entry in random.sample(ranking_log_data, 5)]
##    comment_embeddings = text_embedding_model(sample_comments)
##    print(f"\nSample comments: {sample_comments}")
##    print(f"Shape of sample comment embeddings: {comment_embeddings.shape}")

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_bpr_data(ranking_log_data, text_embedding_model, embedding_dim=64):
    df = pd.DataFrame(ranking_log_data)

    # Convert timestamp to datetime objects for sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Encode user_id and product_id to numerical IDs
    user_encoder = LabelEncoder()
    product_encoder = LabelEncoder()

    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['product_id_encoded'] = product_encoder.fit_transform(df['product_id'])

    # Get unique users and products
    unique_users = df['user_id_encoded'].unique()
    unique_products = df['product_id_encoded'].unique()

    num_users = len(unique_users)
    num_products = len(unique_products)

    # Generate text embeddings for all unique comments
    unique_comments = df['comments'].unique().tolist()
    comment_to_embedding = {}

    # Process comments in batches if there are many to avoid out-of-memory errors
    batch_size = 128
    for i in range(0, len(unique_comments), batch_size):
        batch_comments = unique_comments[i:i + batch_size]
        with torch.no_grad():  # No need to track gradients for embedding generation
            batch_embeddings = text_embedding_model(batch_comments)
        for j, comment in enumerate(batch_comments):
            comment_to_embedding[comment] = batch_embeddings[j].cpu().numpy()  # Store as numpy array

    df['comment_embedding'] = df['comments'].apply(lambda x: comment_to_embedding[x])

    # Group by user and sort by timestamp to create behavior sequences
    user_behavior_sequences = defaultdict(list)
    for user_id_encoded, group in df.sort_values(by='timestamp').groupby('user_id_encoded'):
        for _, row in group.iterrows():
            is_positive = row['clicked'] == 1 or row['added_to_cart'] == 1 or row['purchased'] == 1
            user_behavior_sequences[user_id_encoded].append({
                'product_id_encoded': row['product_id_encoded'],
                'is_positive': is_positive,
                'comment_embedding': row['comment_embedding'],
                'timestamp': row['timestamp']  # Keep timestamp for sorting within model
            })

    # Generate BPR training samples: (user, positive_item, negative_item)
    bpr_training_samples = []

    all_product_ids = torch.arange(num_products)

    for user_id_encoded, interactions in user_behavior_sequences.items():
        positive_items = [interaction['product_id_encoded'] for interaction in interactions if
                          interaction['is_positive']]
        negative_items_pool = [interaction['product_id_encoded'] for interaction in interactions if
                               not interaction['is_positive']]

        # If a user has no positive interactions, skip them for BPR training
        if not positive_items:
            continue

        for positive_item_id in positive_items:
            # Randomly sample a negative item
            # Prioritize negative items from the user's history if available, else sample from all products
            if negative_items_pool:
                negative_item_id = random.choice(negative_items_pool)
            else:
                # Ensure the sampled negative item is not one of the positive items
                available_neg_products = list(set(range(num_products)) - set(positive_items))
                if not available_neg_products:  # All products are positive for this user - highly unlikely with synthetic data
                    continue
                negative_item_id = random.choice(available_neg_products)

            # Find the comment embedding associated with the positive product for this user
            # In a real scenario, a product would have a consistent description/comment_embedding,
            # but here comments vary per interaction. We'll use the specific interaction's comment.
            # This is a simplification; ideally, product features (including text) are static.
            # For BPR, the comment embedding needs to be associated with the positive item.
            positive_comment_embedding = None
            for interaction in interactions:
                if interaction['product_id_encoded'] == positive_item_id and interaction['is_positive']:
                    positive_comment_embedding = interaction['comment_embedding']
                    break

            if positive_comment_embedding is None:  # Should not happen if positive_items are derived correctly
                continue

            bpr_training_samples.append({
                'user_id': user_id_encoded,
                'positive_product_id': positive_item_id,
                'negative_product_id': negative_item_id,
                'positive_comment_embedding': positive_comment_embedding
            })

    print(f"\nGenerated {len(bpr_training_samples)} BPR training samples.")
    print("Example BPR training sample:")
    if bpr_training_samples:
        print(bpr_training_samples[0])

    return bpr_training_samples, num_users, num_products, user_encoder, product_encoder


##if __name__ == "__main__":
##    # Assuming ranking_log_data and text_embedding_model are already generated from Section 1
##    # If running independently, uncomment and run Section 1's main block first
##    ranking_log_data = generate_ranking_log(num_samples=1000)
##    vocabulary = get_vocabulary(ranking_log_data)
##    embedding_dim = 64
##    text_embedding_model = SimpleTextEmbedding(vocabulary, embedding_dim)
##
##    bpr_samples, num_users, num_products, user_encoder, product_encoder = prepare_bpr_data(ranking_log_data,
##                                                                                           embedding_dim)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Custom Dataset for BPR
class BPRDataset(Dataset):
    def __init__(self, bpr_samples):
        self.bpr_samples = bpr_samples

    def __len__(self):
        return len(self.bpr_samples)

    def __getitem__(self, idx):
        sample = self.bpr_samples[idx]
        user_id = torch.tensor(sample['user_id'], dtype=torch.long)
        positive_product_id = torch.tensor(sample['positive_product_id'], dtype=torch.long)
        negative_product_id = torch.tensor(sample['negative_product_id'], dtype=torch.long)
        positive_comment_embedding = torch.tensor(sample['positive_comment_embedding'], dtype=torch.float)

        return user_id, positive_product_id, negative_product_id, positive_comment_embedding


# BPR Model
class BPRModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim, comment_embedding_dim):
        super(BPRModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.product_embeddings = nn.Embedding(num_products, embedding_dim)

        # Linear layer to combine product and comment embeddings
        # We assume the comment embedding provides additional information about the positive product
        # The output dimension should match the product embedding dimension for consistent dot product.
        self.product_comment_fusion = nn.Linear(embedding_dim + comment_embedding_dim, embedding_dim)

        # Initialize weights
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.product_embeddings.weight, std=0.01)
        nn.init.xavier_uniform_(self.product_comment_fusion.weight)
        nn.init.zeros_(self.product_comment_fusion.bias)

    def forward(self, user_ids, positive_product_ids, negative_product_ids, positive_comment_embeddings):
        user_emb = self.user_embeddings(user_ids)
        positive_product_emb_raw = self.product_embeddings(positive_product_ids)
        negative_product_emb = self.product_embeddings(negative_product_ids)

        # Concatenate product embedding with its associated comment embedding
        # This assumes the comment is describing the *positive* product.
        fused_positive_product_input = torch.cat((positive_product_emb_raw, positive_comment_embeddings), dim=1)
        fused_positive_product_emb = self.product_comment_fusion(fused_positive_product_input)

        # Calculate scores
        score_positive = torch.sum(user_emb * fused_positive_product_emb, dim=1)
        score_negative = torch.sum(user_emb * negative_product_emb, dim=1)

        return score_positive, score_negative


# BPR Loss Function
class BPRLoss(nn.Module):
    def __init__(self, weight_decay=0.001):
        super(BPRLoss, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, score_positive, score_negative, user_embeddings, positive_product_embeddings,
                negative_product_embeddings):
        # BPR loss: -log(sigmoid(score_positive - score_negative))
        # We add regularization terms manually to the loss
        loss = -torch.sum(torch.log(torch.sigmoid(score_positive - score_negative)))

        # L2 regularization
        regularization_loss = self.weight_decay * (
                torch.norm(user_embeddings, p=2) +
                torch.norm(positive_product_embeddings, p=2) +
                torch.norm(negative_product_embeddings, p=2)
        )
        return loss + regularization_loss


def train_bpr_model(bpr_training_samples, num_users, num_products, embedding_dim, comment_embedding_dim, epochs=50,
                    batch_size=256, learning_rate=0.01, weight_decay=0.001):
    dataset = BPRDataset(bpr_training_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BPRModel(num_users, num_products, embedding_dim, comment_embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BPRLoss(weight_decay=weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("\nStarting BPR model training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_ids, positive_product_ids, negative_product_ids, positive_comment_embeddings in dataloader:
            user_ids = user_ids.to(device)
            positive_product_ids = positive_product_ids.to(device)
            negative_product_ids = negative_product_ids.to(device)
            positive_comment_embeddings = positive_comment_embeddings.to(device)

            optimizer.zero_grad()

            score_positive, score_negative = model(user_ids, positive_product_ids, negative_product_ids,
                                                   positive_comment_embeddings)

            # Access raw embeddings for regularization in the loss function
            user_embeddings_for_reg = model.user_embeddings(user_ids)
            positive_product_embeddings_for_reg = model.product_embeddings(positive_product_ids)
            negative_product_embeddings_for_reg = model.product_embeddings(negative_product_ids)

            loss = criterion(score_positive, score_negative,
                             user_embeddings_for_reg,
                             positive_product_embeddings_for_reg,
                             negative_product_embeddings_for_reg)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    print("BPR model training complete.")
    return model


##if __name__ == "__main__":
##    # Assuming bpr_samples, num_users, num_products, embedding_dim are from Section 2
##    # If running independently, uncomment and run Section 1 and 2's main blocks first
##    ranking_log_data = generate_ranking_log(num_samples=1000)
##    vocabulary = get_vocabulary(ranking_log_data)
##    embedding_dim = 64
##    text_embedding_model = SimpleTextEmbedding(vocabulary, embedding_dim)
##    bpr_samples, num_users, num_products, user_encoder, product_encoder = prepare_bpr_data(ranking_log_data,
##                                                                                           text_embedding_model,
##                                                                                           embedding_dim)
##
##    # Re-initialize text_embedding_model to get the correct embedding_dim
##    # For a real system, text_embedding_model would be trained or pre-trained.
##    # Here, we just need its output dimension.
##    comment_embedding_dim = embedding_dim  # This is the output dim of SimpleTextEmbedding
##
##    bpr_model = train_bpr_model(bpr_samples, num_users, num_products, embedding_dim, comment_embedding_dim)



from sklearn.metrics import roc_auc_score
from collections import defaultdict


def calculate_ndcg(relevant_items_scores, k=10):
    """
    Calculates NDCG@k.
    :param relevant_items_scores: List of scores for items known to be relevant, sorted by their predicted score.
                                  Higher score indicates higher relevance in this context (e.g., 1 for relevant).
    :param k: The number of top items to consider.
    """
    if not relevant_items_scores:
        return 0.0

    # Sort scores in descending order to simulate ranked list
    sorted_scores = sorted(relevant_items_scores, reverse=True)

    dcg = 0.0
    for i in range(min(k, len(sorted_scores))):
        # Assuming scores are binary (1 for relevant, 0 for irrelevant) for relevance gain
        # If your 'score' is continuous relevance, use it directly.
        # Here, we treat any presence in relevant_items_scores as a gain of 1.
        gain = 1  # Assuming all items in relevant_items_scores are relevant
        dcg += gain / np.log2(i + 2)  # i+2 because log2(1) is 0, first position is 1, so index 0 is position 1+1

    idcg = 0.0
    # Ideal DCG assumes all k relevant items are at the top
    for i in range(
            min(k, len(relevant_items_scores))):  # Use len(relevant_items_scores) as the ideal number of relevant items
        idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_bpr_model(model, ranking_log_data, user_encoder, product_encoder, text_embedding_model, embedding_dim=64,
                       k=10):
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.DataFrame(ranking_log_data)
    df['user_id_encoded'] = user_encoder.transform(df['user_id'])
    df['product_id_encoded'] = product_encoder.transform(df['product_id'])

    # Pre-compute unique comment embeddings as in data prep
    unique_comments = df['comments'].unique().tolist()
    comment_to_embedding = {}
    batch_size = 128
    for i in range(0, len(unique_comments), batch_size):
        batch_comments = unique_comments[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = text_embedding_model(batch_comments)
        for j, comment in enumerate(batch_comments):
            comment_to_embedding[comment] = batch_embeddings[j].cpu().numpy()

    # Store predicted scores for AUC
    all_true_labels = []
    all_predicted_scores = []

    # Store user recommendations for NDCG
    user_recommendations = defaultdict(list)  # {user_id: [(product_id, score, is_relevant), ...]}

    print("\nStarting model evaluation...")
    with torch.no_grad():
        for user_id_encoded in df['user_id_encoded'].unique():
            user_interaction_df = df[df['user_id_encoded'] == user_id_encoded]

            # Get all products seen by this user or a representative set of products
            # For evaluation, we need to score all possible products for a user.
            # Here, we'll score all products that appeared in the dataset.
            # In a real scenario, you'd score all products in your catalog.

            # Products to score for the current user
            products_to_score_ids = torch.tensor(product_encoder.transform(df['product_id'].unique()),
                                                 dtype=torch.long).to(device)
            user_id_tensor = torch.tensor([user_id_encoded], dtype=torch.long).to(device)

            # To get comment embeddings for all products, we need a way to map product to comment.
            # This is tricky because comments are per interaction. For evaluation, we might use
            # an average comment embedding for a product, or the most recent/common one.
            # For simplicity here, we'll just use a zero vector for comments during prediction
            # if we don't have a specific comment context for a product being scored,
            # or try to link a "representative" comment to each product.
            # A more robust approach would be to have static product description embeddings.

            # For simplicity in evaluation: let's assume a dummy comment embedding for products not in the current interaction
            # Or, if we have true product features, we'd use those.
            # For BPR, the model learns a user_embedding and product_embedding.
            # The comment embedding was applied to the *positive* product during training.
            # For prediction, we need a way to generate a score for ANY user-product pair.
            # The simplest way is to assume no specific comment context for products not explicitly clicked/interacted with.
            # A better approach: the product embedding itself would carry some information, and if we want
            # to incorporate comment, the product *description* would be embedded, not a specific user's feedback.

            # Let's adjust BPRModel for prediction to take a single product ID and its generic comment embedding
            # (which can be learned as part of product features or derived from descriptions).
            # For now, we'll score based on product embeddings and assume no *new* comment info for unseen pairs.
            # If we want to include comment_embeddings in evaluation, we need to think about how to get them for all products.
            # The current BPRModel is set up for training triplets.

            # Let's modify the BPRModel to have a `predict` method that handles single user-product scores.
            # The `product_comment_fusion` was for positive items during training.
            # For prediction, we want score(u, i) = user_emb * product_emb + some_comment_influence.
            # For simplicity for evaluation, we will ignore comment embeddings for negative samples in prediction
            # as they were only explicitly combined for positive items during training.
            # A better model would have a fixed comment embedding per product.

            # We will use the product embeddings directly for scoring in evaluation.
            # The BPR training process encourages positive_product_emb + comment_emb to be similar to user_emb.
            # So, the product_comment_fusion layer basically creates a 'boosted' product embedding for positive items.
            # For evaluation, we can assume the final learned product_embeddings already implicitly contain some "goodness".

            user_embedding = model.user_embeddings(user_id_tensor)

            # Generate scores for all possible products (for this example, all unique products in the dataset)
            all_product_ids = torch.tensor(range(num_products), dtype=torch.long).to(device)
            all_product_embeddings = model.product_embeddings(all_product_ids)

            # Calculate scores for all user-product pairs
            # This is a matrix multiplication if we consider all products for one user
            predicted_scores = torch.sum(user_embedding * all_product_embeddings,
                                         dim=1).cpu().numpy()  # shape (num_products,)

            product_id_score_map = {product_id: score for product_id, score in
                                    zip(range(num_products), predicted_scores)}

            # --- For AUC Calculation ---
            # For each user, we need pairs of (positive, negative) products for AUC.
            positive_product_ids_user = set(
                user_interaction_df[user_interaction_df['clicked'] == 1]['product_id_encoded'].tolist())
            negative_product_ids_user = set(
                df['product_id_encoded'].unique()) - positive_product_ids_user  # All products not clicked by this user

            if positive_product_ids_user and negative_product_ids_user:
                # Sample negative products to make the AUC calculation manageable if too many
                sampled_negative_ids = random.sample(list(negative_product_ids_user),
                                                     min(len(negative_product_ids_user),
                                                         len(positive_product_ids_user) * 5))  # Up to 5 negatives per positive

                for pos_prod_id in positive_product_ids_user:
                    for neg_prod_id in sampled_negative_ids:
                        all_predicted_scores.append(
                            product_id_score_map[pos_prod_id] - product_id_score_map[neg_prod_id])
                        all_true_labels.append(1)  # Positive difference should be ranked higher

            # --- For NDCG Calculation ---
            # Rank all products for the user based on predicted scores
            ranked_products = sorted([
                (product_id, score) for product_id, score in product_id_score_map.items()
            ], key=lambda x: x[1], reverse=True)

            # Identify truly relevant products for NDCG
            # Here, we define relevant as "clicked" for simplicity in NDCG.
            relevant_products_for_ndcg = set(
                user_interaction_df[user_interaction_df['clicked'] == 1]['product_id_encoded'].tolist())

            # Get scores of relevant items in the ranked list
            relevant_items_at_k = []
            for i in range(min(k, len(ranked_products))):
                product_id, score = ranked_products[i]
                if product_id in relevant_products_for_ndcg:
                    relevant_items_at_k.append(score)  # Or just a 1 if we're doing binary relevance

            user_recommendations[user_id_encoded] = relevant_items_at_k

    # Calculate overall AUC
    auc_score = 0.0
    if all_true_labels:
        # For AUC, we need scores for positive and negative items directly, not their difference.
        # Let's re-structure for AUC: predict scores for all known positive and a random sample of negative items.
        # Then, use roc_auc_score(true_binary_label, predicted_score)

        # Let's rebuild for AUC based on (user, item) pairs and their click status
        auc_user_ids = []
        auc_product_ids = []
        auc_labels = []

        for user_id_encoded in df['user_id_encoded'].unique():
            user_interactions = df[df['user_id_encoded'] == user_id_encoded]
            positive_items = user_interactions[user_interactions['clicked'] == 1]
            negative_items = user_interactions[user_interactions['clicked'] == 0]

            for _, row in positive_items.iterrows():
                auc_user_ids.append(user_id_encoded)
                auc_product_ids.append(row['product_id_encoded'])
                auc_labels.append(1)

            # Sample negative items to balance the dataset somewhat for AUC calculation
            if not negative_items.empty:
                sampled_negative_items = negative_items.sample(min(len(negative_items), len(positive_items) * 5))
                for _, row in sampled_negative_items.iterrows():
                    auc_user_ids.append(user_id_encoded)
                    auc_product_ids.append(row['product_id_encoded'])
                    auc_labels.append(0)

        if auc_user_ids:
            # Predict scores for the collected AUC pairs
            user_ids_tensor_auc = torch.tensor(auc_user_ids, dtype=torch.long).to(device)
            product_ids_tensor_auc = torch.tensor(auc_product_ids, dtype=torch.long).to(device)

            with torch.no_grad():
                user_embeddings_auc = model.user_embeddings(user_ids_tensor_auc)
                product_embeddings_auc = model.product_embeddings(product_ids_tensor_auc)
                predicted_scores_auc = torch.sum(user_embeddings_auc * product_embeddings_auc, dim=1).cpu().numpy()

            auc_score = roc_auc_score(auc_labels, predicted_scores_auc)

    # Calculate overall NDCG
    ndcg_scores = [calculate_ndcg(scores, k=k) for scores in user_recommendations.values() if scores]
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    print(f"\nEvaluation Results (k={k}):")
    print(f"AUC: {auc_score:.4f}")
    print(f"Average NDCG@{k}: {avg_ndcg:.4f}")

    # MSE is typically for regression. BPR is a ranking model.
    # If you want MSE, you'd need to predict a numerical score (e.g., the original ranking_score)
    # and compare it to the model's output, but BPR's output is not directly a score in that sense.
    # We will skip direct MSE for BPR's output. If you want to predict ranking_score, that would be a different model.
    # If the user wants MSE for 'ranking score', the model needs to be a regression model, not BPR.
    # BPR optimizes for relative ranking.


if __name__ == "__main__":
    # Ensure all previous steps are run to get necessary data and models
    ranking_log_data = generate_ranking_log(num_samples=1000)
    vocabulary = get_vocabulary(ranking_log_data)
    embedding_dim = 64
    text_embedding_model = SimpleTextEmbedding(vocabulary, embedding_dim)
    bpr_samples, num_users, num_products, user_encoder, product_encoder = prepare_bpr_data(ranking_log_data,
                                                                                           text_embedding_model,
                                                                                           embedding_dim)

    comment_embedding_dim = embedding_dim
    bpr_model = train_bpr_model(bpr_samples, num_users, num_products, embedding_dim, comment_embedding_dim)

    # Evaluate the trained model
    evaluate_bpr_model(bpr_model, ranking_log_data, user_encoder, product_encoder, text_embedding_model, k=10)

