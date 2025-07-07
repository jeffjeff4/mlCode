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


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.metrics import roc_auc_score, mean_squared_error, ndcg_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import string
import re

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 1. Generate Ranking Log Dataset
class RankingLogGenerator:
    def __init__(self, n_samples=1000, n_users=100, n_products=300):
        self.n_samples = n_samples
        self.n_users = n_users
        self.n_products = n_products
        self.product_titles = self._generate_product_titles()
        self.comment_templates = self._generate_comment_templates()

    def _generate_product_titles(self):
        categories = ['Phone', 'Laptop', 'Headphones', 'Watch', 'Tablet', 'Camera', 'Speaker', 'Monitor']
        brands = ['Apple', 'Samsung', 'Sony', 'Dell', 'HP', 'Canon', 'Bose', 'LG']
        titles = []
        for i in range(self.n_products):
            category = random.choice(categories)
            brand = random.choice(brands)
            model = f"Model-{random.randint(100, 999)}"
            titles.append(f"{brand} {category} {model}")
        return titles

    def _generate_comment_templates(self):
        positive_comments = [
            "Great product, very satisfied with the quality and performance.",
            "Excellent value for money, highly recommend to others.",
            "Amazing features, works perfectly as expected.",
            "Love this product, great build quality and design.",
            "Outstanding performance, exceeded my expectations.",
            "Perfect for my needs, great customer service too.",
            "High quality product, fast delivery and good packaging.",
            "Impressive features, user-friendly interface and reliable."
        ]

        negative_comments = [
            "Poor quality, not worth the money spent.",
            "Disappointed with the performance, many issues encountered.",
            "Bad experience, product broke after few days.",
            "Overpriced for what you get, not recommended.",
            "Poor customer service, product has many defects.",
            "Not as described, quality is much lower than expected.",
            "Waste of money, many problems with this product.",
            "Terrible experience, would not buy again from this brand."
        ]

        return {'positive': positive_comments, 'negative': negative_comments}

    def generate_log(self):
        data = []
        start_time = datetime.now() - timedelta(days=30)

        for i in range(self.n_samples):
            user_id = random.randint(1, self.n_users)
            product_id = random.randint(1, self.n_products)
            product_title = self.product_titles[product_id - 1]
            price = round(random.uniform(10, 2000), 2)
            ranking_score = random.randint(1, 5)
            ranking_position = random.randint(1, 60)
            timestamp = start_time + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # Generate behavior with logical constraints
            # Higher ranking score and better position increase positive behavior probability
            click_prob = 0.1 + (ranking_score - 1) * 0.15 + (60 - ranking_position) * 0.01
            is_clicked = random.random() < click_prob

            is_added_to_cart = False
            is_purchased = False

            if is_clicked:
                # If clicked, chance to add to cart
                cart_prob = 0.1 + (ranking_score - 1) * 0.1
                is_added_to_cart = random.random() < cart_prob

                if is_added_to_cart:
                    # If added to cart, chance to purchase
                    purchase_prob = 0.2 + (ranking_score - 1) * 0.1
                    is_purchased = random.random() < purchase_prob

            # Generate comments based on behavior and ranking score
            if is_purchased or is_added_to_cart or (is_clicked and ranking_score >= 4):
                comment = random.choice(self.comment_templates['positive'])
            elif ranking_score <= 2:
                comment = random.choice(self.comment_templates['negative'])
            else:
                comment = random.choice(self.comment_templates['positive'] + self.comment_templates['negative'])

            data.append({
                'product_title': product_title,
                'product_id': product_id,
                'user_id': user_id,
                'price': price,
                'ranking_score': ranking_score,
                'is_clicked': int(is_clicked),
                'is_added_to_cart': int(is_added_to_cart),
                'is_purchased': int(is_purchased),
                'comments': comment,
                'timestamp': timestamp,
                'ranking_position': ranking_position
            })

        return pd.DataFrame(data)


# 2. Simple Text Embedding using PyTorch
class SimpleTextEmbedding:
    def __init__(self, vocab_size=5000, embed_dim=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def build_vocab(self, texts):
        # Simple tokenization and vocabulary building
        word_freq = defaultdict(int)
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freq[word] += 1

        # Keep most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        for i, (word, freq) in enumerate(sorted_words[:self.vocab_size - 2]):
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word

    def _tokenize(self, text):
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def text_to_indices(self, text, max_len=50):
        words = self._tokenize(text)
        indices = []
        for word in words[:max_len]:
            indices.append(self.word_to_idx.get(word, 1))  # 1 for <UNK>

        # Pad to max_len
        while len(indices) < max_len:
            indices.append(0)  # 0 for <PAD>

        return indices

    def get_embedding(self, texts, max_len=50):
        indices_list = []
        for text in texts:
            indices = self.text_to_indices(text, max_len)
            indices_list.append(indices)

        indices_tensor = torch.tensor(indices_list, dtype=torch.long)
        embeddings = self.embedding(indices_tensor)

        # Average pooling over sequence length
        text_embeddings = embeddings.mean(dim=1)
        return text_embeddings


# 3. ESMM Model Implementation
class ESMMModel(nn.Module):
    def __init__(self, n_users, n_products, embed_dim=64, text_embed_dim=128, hidden_dim=256):
        super(ESMMModel, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users + 1, embed_dim)
        self.product_embedding = nn.Embedding(n_products + 1, embed_dim)

        # Feature dimensions
        feature_dim = embed_dim * 2 + text_embed_dim + 3  # user_emb + product_emb + text_emb + 3 features

        # Shared bottom layers
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific towers
        # CTR tower (Click-Through Rate)
        self.ctr_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # CVR tower (Conversion Rate)
        self.cvr_tower = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, user_ids, product_ids, text_embeddings, prices, ranking_scores, ranking_positions):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        product_emb = self.product_embedding(product_ids)

        # Normalize numerical features
        prices_norm = (prices - prices.mean()) / (prices.std() + 1e-8)
        ranking_scores_norm = (ranking_scores.float() - 3.0) / 2.0
        ranking_positions_norm = (ranking_positions.float() - 30.0) / 30.0

        # Concatenate features
        features = torch.cat([
            user_emb, product_emb, text_embeddings,
            prices_norm.unsqueeze(1),
            ranking_scores_norm.unsqueeze(1),
            ranking_positions_norm.unsqueeze(1)
        ], dim=1)

        # Shared layers
        shared_output = self.shared_layers(features)

        # Task-specific outputs
        ctr_output = self.ctr_tower(shared_output)
        cvr_output = self.cvr_tower(shared_output)

        # CTCVR = CTR * CVR
        ctcvr_output = ctr_output * cvr_output

        return ctr_output.squeeze(), cvr_output.squeeze(), ctcvr_output.squeeze()


# 4. Training Sample Generation
class TrainingSampleGenerator:
    def __init__(self, df, text_embedder):
        self.df = df
        self.text_embedder = text_embedder

    def generate_samples(self):
        # Sort by user and timestamp
        df_sorted = self.df.sort_values(['user_id', 'timestamp'])

        # Create training samples without text embeddings (will be generated during training)
        samples = []
        for idx, row in df_sorted.iterrows():
            sample = {
                'user_id': row['user_id'],
                'product_id': row['product_id'],
                'comments': row['comments'],  # Keep original comments
                'price': row['price'],
                'ranking_score': row['ranking_score'],
                'ranking_position': row['ranking_position'],
                'is_clicked': row['is_clicked'],
                'is_added_to_cart': row['is_added_to_cart'],
                'is_purchased': row['is_purchased'],
                'timestamp': row['timestamp']
            }
            samples.append(sample)

        return samples


# 5. Model Training
def train_esmm_model(model, train_samples, val_samples, text_embedder, epochs=50, lr=0.001, batch_size=256):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    train_losses = []
    val_losses = []

    # Pre-compute text embeddings to avoid recomputation
    print("Pre-computing text embeddings...")
    train_comments = [s['comments'] for s in train_samples]
    val_comments = [s['comments'] for s in val_samples]

    train_text_emb = text_embedder.get_embedding(train_comments).detach()
    val_text_emb = text_embedder.get_embedding(val_comments).detach()

    # Convert all data to tensors once
    train_user_ids = torch.tensor([s['user_id'] for s in train_samples], dtype=torch.long)
    train_product_ids = torch.tensor([s['product_id'] for s in train_samples], dtype=torch.long)
    train_prices = torch.tensor([s['price'] for s in train_samples], dtype=torch.float)
    train_ranking_scores = torch.tensor([s['ranking_score'] for s in train_samples], dtype=torch.long)
    train_ranking_positions = torch.tensor([s['ranking_position'] for s in train_samples], dtype=torch.long)
    train_clicks = torch.tensor([s['is_clicked'] for s in train_samples], dtype=torch.float)
    train_purchases = torch.tensor([s['is_purchased'] for s in train_samples], dtype=torch.float)

    val_user_ids = torch.tensor([s['user_id'] for s in val_samples], dtype=torch.long)
    val_product_ids = torch.tensor([s['product_id'] for s in val_samples], dtype=torch.long)
    val_prices = torch.tensor([s['price'] for s in val_samples], dtype=torch.float)
    val_ranking_scores = torch.tensor([s['ranking_score'] for s in val_samples], dtype=torch.long)
    val_ranking_positions = torch.tensor([s['ranking_position'] for s in val_samples], dtype=torch.long)
    val_clicks = torch.tensor([s['is_clicked'] for s in val_samples], dtype=torch.float)
    val_purchases = torch.tensor([s['is_purchased'] for s in val_samples], dtype=torch.float)

    n_train = len(train_samples)
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle training data
        indices = torch.randperm(n_train)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_train)
            batch_indices = indices[start_idx:end_idx]

            # Get batch data
            batch_user_ids = train_user_ids[batch_indices]
            batch_product_ids = train_product_ids[batch_indices]
            batch_text_emb = train_text_emb[batch_indices]
            batch_prices = train_prices[batch_indices]
            batch_ranking_scores = train_ranking_scores[batch_indices]
            batch_ranking_positions = train_ranking_positions[batch_indices]
            batch_clicks = train_clicks[batch_indices]
            batch_purchases = train_purchases[batch_indices]

            optimizer.zero_grad()

            ctr_pred, cvr_pred, ctcvr_pred = model(
                batch_user_ids, batch_product_ids, batch_text_emb,
                batch_prices, batch_ranking_scores, batch_ranking_positions
            )

            # ESMM loss: CTR loss + CTCVR loss
            ctr_loss = criterion(ctr_pred, batch_clicks)
            ctcvr_loss = criterion(ctcvr_pred, batch_purchases)

            batch_loss = ctr_loss + ctcvr_loss
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        train_losses.append(epoch_loss / n_batches)

        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_ctr_pred, val_cvr_pred, val_ctcvr_pred = model(
                    val_user_ids, val_product_ids, val_text_emb,
                    val_prices, val_ranking_scores, val_ranking_positions
                )

                val_ctr_loss = criterion(val_ctr_pred, val_clicks)
                val_ctcvr_loss = criterion(val_ctcvr_pred, val_purchases)
                val_loss = val_ctr_loss + val_ctcvr_loss

                val_losses.append(val_loss.item())

                print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss.item():.4f}')

    return train_losses, val_losses


# 6. Model Evaluation
def evaluate_model(model, test_samples, text_embedder):
    model.eval()
    with torch.no_grad():
        # Generate text embeddings for test samples
        test_comments = [s['comments'] for s in test_samples]
        test_text_emb = text_embedder.get_embedding(test_comments).detach()

        test_user_ids = torch.tensor([s['user_id'] for s in test_samples], dtype=torch.long)
        test_product_ids = torch.tensor([s['product_id'] for s in test_samples], dtype=torch.long)
        test_prices = torch.tensor([s['price'] for s in test_samples], dtype=torch.float)
        test_ranking_scores = torch.tensor([s['ranking_score'] for s in test_samples], dtype=torch.long)
        test_ranking_positions = torch.tensor([s['ranking_position'] for s in test_samples], dtype=torch.long)

        test_clicks = np.array([s['is_clicked'] for s in test_samples])
        test_purchases = np.array([s['is_purchased'] for s in test_samples])

        ctr_pred, cvr_pred, ctcvr_pred = model(
            test_user_ids, test_product_ids, test_text_emb,
            test_prices, test_ranking_scores, test_ranking_positions
        )

        ctr_pred_np = ctr_pred.cpu().numpy()
        ctcvr_pred_np = ctcvr_pred.cpu().numpy()

        # Calculate metrics
        ctr_auc = roc_auc_score(test_clicks, ctr_pred_np)
        ctr_mse = mean_squared_error(test_clicks, ctr_pred_np)

        if test_purchases.sum() > 0:
            ctcvr_auc = roc_auc_score(test_purchases, ctcvr_pred_np)
            ctcvr_mse = mean_squared_error(test_purchases, ctcvr_pred_np)
        else:
            ctcvr_auc = 0.0
            ctcvr_mse = 0.0

        # NDCG calculation - group by user
        ndcg_scores = []
        user_groups = defaultdict(list)
        for i, sample in enumerate(test_samples):
            user_groups[sample['user_id']].append({
                'relevance': sample['ranking_score'],
                'prediction': ctr_pred_np[i],
                'position': sample['ranking_position']
            })

        for user_id, user_samples in user_groups.items():
            if len(user_samples) > 1:
                relevance_scores = [s['relevance'] for s in user_samples]
                prediction_scores = [s['prediction'] for s in user_samples]

                try:
                    ndcg = ndcg_score([relevance_scores], [prediction_scores])
                    ndcg_scores.append(ndcg)
                except:
                    continue

        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

        return {
            'CTR_AUC': ctr_auc,
            'CTR_MSE': ctr_mse,
            'CTCVR_AUC': ctcvr_auc,
            'CTCVR_MSE': ctcvr_mse,
            'NDCG': avg_ndcg
        }


# 7. Main Execution
if __name__ == "__main__":
    print("=== Generating Ranking Log ===")
    generator = RankingLogGenerator(n_samples=1000, n_users=100, n_products=300)
    df = generator.generate_log()

    print(f"Generated {len(df)} samples")
    print(f"Positive samples (clicked): {df['is_clicked'].sum()}")
    print(f"Negative samples (not clicked): {len(df) - df['is_clicked'].sum()}")
    print(f"Added to cart: {df['is_added_to_cart'].sum()}")
    print(f"Purchased: {df['is_purchased'].sum()}")
    print("\nSample data:")
    print(df.head())

    print("\n=== Building Text Embeddings ===")
    text_embedder = SimpleTextEmbedding(vocab_size=1000, embed_dim=128)
    text_embedder.build_vocab(df['comments'].tolist())

    print("\n=== Generating Training Samples ===")
    sample_generator = TrainingSampleGenerator(df, text_embedder)
    all_samples = sample_generator.generate_samples()

    # Split samples
    train_samples, temp_samples = train_test_split(all_samples, test_size=0.4, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    print("\n=== Training ESMM Model ===")
    model = ESMMModel(
        n_users=df['user_id'].max(),
        n_products=df['product_id'].max(),
        embed_dim=64,
        text_embed_dim=128,
        hidden_dim=256
    )

    train_losses, val_losses = train_esmm_model(model, train_samples, val_samples, text_embedder, epochs=50, lr=0.001)

    print("\n=== Evaluating Model ===")
    metrics = evaluate_model(model, test_samples, text_embedder)

    print("Final Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== Training Complete ===")
    print("Model successfully trained and evaluated!")
    print(f"Final CTR AUC: {metrics['CTR_AUC']:.4f}")
    print(f"Final CTCVR AUC: {metrics['CTCVR_AUC']:.4f}")
    print(f"Final NDCG: {metrics['NDCG']:.4f}")