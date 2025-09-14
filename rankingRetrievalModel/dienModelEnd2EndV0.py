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
from typing import List, Dict, Tuple

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
class SimpleTextEmbedding(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128):
        super(SimpleTextEmbedding, self).__init__()
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

    def forward(self, texts, max_len=50):
        indices_list = []
        for text in texts:
            indices = self.text_to_indices(text, max_len)
            indices_list.append(indices)

        indices_tensor = torch.tensor(indices_list, dtype=torch.long)
        embeddings = self.embedding(indices_tensor)

        # Average pooling over sequence length
        text_embeddings = embeddings.mean(dim=1)
        return text_embeddings


# 3. DIEN Model Implementation
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs, mask=None):
        # inputs: (batch_size, seq_len, input_dim)
        attention_weights = self.attention(inputs)  # (batch_size, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch_size, seq_len)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_output = torch.sum(inputs * attention_weights.unsqueeze(-1), dim=1)

        return weighted_output, attention_weights


class InterestEvolutionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InterestEvolutionLayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Fix: Attention layer should expect the combined feature size
        combined_dim = hidden_dim + input_dim  # GRU output + target item features
        self.attention = AttentionLayer(combined_dim, combined_dim // 2)

    def forward(self, sequence, target_item, mask=None):
        # sequence: (batch_size, seq_len, input_dim)
        # target_item: (batch_size, input_dim)

        gru_output, _ = self.gru(sequence)  # (batch_size, seq_len, hidden_dim)

        # Attention mechanism with target item
        target_expanded = target_item.unsqueeze(1).expand(-1, gru_output.size(1), -1)
        combined = torch.cat([gru_output, target_expanded], dim=-1)

        # Now combined has shape (batch_size, seq_len, hidden_dim + input_dim)
        # which matches what the attention layer expects

        # Apply attention
        final_state, attention_weights = self.attention(combined, mask)

        return final_state, attention_weights


# Alternative Solution: Use only GRU output for attention
class InterestEvolutionLayerAlternative(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InterestEvolutionLayerAlternative, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Use only GRU output for attention (simpler approach)
        self.attention = AttentionLayer(hidden_dim, hidden_dim // 2)

        # Optional: Add a separate layer to incorporate target item influence
        self.target_influence = nn.Linear(input_dim, hidden_dim)

    def forward(self, sequence, target_item, mask=None):
        # sequence: (batch_size, seq_len, input_dim)
        # target_item: (batch_size, input_dim)

        gru_output, _ = self.gru(sequence)  # (batch_size, seq_len, hidden_dim)

        # Apply attention to GRU output only
        final_state, attention_weights = self.attention(gru_output, mask)

        # Optional: Add target item influence
        target_influence = self.target_influence(target_item)
        final_state = final_state + target_influence

        return final_state, attention_weights

##class DIENModel(nn.Module):
##    def __init__(self, n_users, n_products, embed_dim=64, text_embed_dim=128,
##                 hidden_dim=256, seq_len=10):
##        super(DIENModel, self).__init__()
##
##        self.embed_dim = embed_dim
##        self.seq_len = seq_len
##
##        # Embedding layers
##        self.user_embedding = nn.Embedding(n_users + 1, embed_dim)
##        self.product_embedding = nn.Embedding(n_products + 1, embed_dim)
##
##        # Interest extraction layer (GRU)
##        self.interest_extractor = nn.GRU(
##            embed_dim + text_embed_dim + 3,  # product_emb + text_emb + 3 features
##            hidden_dim,
##            batch_first=True
##        )
##
##        # Interest evolution layer
##        self.interest_evolution = InterestEvolutionLayer(
##            embed_dim + text_embed_dim + 3,  # same as input
##            hidden_dim
##        )
##
##        # Final prediction layers
##        feature_dim = embed_dim + hidden_dim + embed_dim + text_embed_dim + 3
##        self.prediction_layers = nn.Sequential(
##            nn.Linear(feature_dim, hidden_dim),
##            nn.ReLU(),
##            nn.Dropout(0.2),
##            nn.Linear(hidden_dim, hidden_dim // 2),
##            nn.ReLU(),
##            nn.Dropout(0.2),
##            nn.Linear(hidden_dim // 2, 1),
##            nn.Sigmoid()
##        )
##
##        # Initialize weights
##        self._init_weights()
##
##    def _init_weights(self):
##        for m in self.modules():
##            if isinstance(m, nn.Linear):
##                nn.init.xavier_uniform_(m.weight)
##                nn.init.constant_(m.bias, 0)
##            elif isinstance(m, nn.Embedding):
##                nn.init.normal_(m.weight, mean=0, std=0.1)
##
##    def forward(self, user_ids, target_product_ids, sequence_product_ids,
##                sequence_text_embeddings, sequence_prices, sequence_ranking_scores,
##                sequence_ranking_positions, target_text_embedding, target_price,
##                target_ranking_score, target_ranking_position, sequence_mask):
##
##        batch_size = user_ids.size(0)
##
##        # Get embeddings
##        user_emb = self.user_embedding(user_ids)
##        target_product_emb = self.product_embedding(target_product_ids)
##
##        # Sequence embeddings
##        sequence_product_emb = self.product_embedding(sequence_product_ids)
##
##        # Normalize numerical features for sequence
##        seq_prices_norm = (sequence_prices - sequence_prices.mean()) / (sequence_prices.std() + 1e-8)
##        seq_scores_norm = (sequence_ranking_scores.float() - 3.0) / 2.0
##        seq_positions_norm = (sequence_ranking_positions.float() - 30.0) / 30.0
##
##        # Combine sequence features
##        sequence_features = torch.cat([
##            sequence_product_emb,
##            sequence_text_embeddings,
##            seq_prices_norm.unsqueeze(-1),
##            seq_scores_norm.unsqueeze(-1),
##            seq_positions_norm.unsqueeze(-1)
##        ], dim=-1)
##
##        # Target item features
##        target_price_norm = (target_price - target_price.mean()) / (target_price.std() + 1e-8)
##        target_score_norm = (target_ranking_score.float() - 3.0) / 2.0
##        target_position_norm = (target_ranking_position.float() - 30.0) / 30.0
##
##        target_features = torch.cat([
##            target_product_emb,
##            target_text_embedding,
##            target_price_norm.unsqueeze(-1),
##            target_score_norm.unsqueeze(-1),
##            target_position_norm.unsqueeze(-1)
##        ], dim=-1)
##
##        # Interest extraction
##        interest_states, _ = self.interest_extractor(sequence_features)
##
##        # Interest evolution with attention
##        evolved_interest, attention_weights = self.interest_evolution(
##            sequence_features, target_features, sequence_mask
##        )
##
##        # Final features combination
##        final_features = torch.cat([
##            user_emb,
##            evolved_interest,
##            target_features
##        ], dim=-1)
##
##        # Prediction
##        output = self.prediction_layers(final_features)
##
##        return output.squeeze(), attention_weights
##

class DIENModel(nn.Module):
    def __init__(self, n_users, n_products, embed_dim=64, text_embed_dim=128,
                 hidden_dim=256, seq_len=10):
        super(DIENModel, self).__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.text_embed_dim = text_embed_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users + 1, embed_dim)
        self.product_embedding = nn.Embedding(n_products + 1, embed_dim)

        # Calculate input dimension for sequences
        self.input_dim = embed_dim + text_embed_dim + 3  # product_emb + text_emb + 3 features

        # Interest extraction layer (GRU)
        self.interest_extractor = nn.GRU(
            self.input_dim,
            hidden_dim,
            batch_first=True
        )

        # Interest evolution layer
        self.interest_evolution = InterestEvolutionLayer(
            self.input_dim,
            hidden_dim
        )

        # Calculate the actual feature dimension for prediction layers
        # After InterestEvolutionLayer, evolved_interest has dimension: (hidden_dim + input_dim) // 2
        evolved_interest_dim = (hidden_dim + self.input_dim) // 2

        # Final prediction layers
        feature_dim = embed_dim + evolved_interest_dim + self.input_dim
        self.prediction_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
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

    def forward(self, user_ids, target_product_ids, sequence_product_ids,
                sequence_text_embeddings, sequence_prices, sequence_ranking_scores,
                sequence_ranking_positions, target_text_embedding, target_price,
                target_ranking_score, target_ranking_position, sequence_mask):

        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        target_product_emb = self.product_embedding(target_product_ids)

        # Sequence embeddings
        sequence_product_emb = self.product_embedding(sequence_product_ids)

        # Normalize numerical features for sequence
        seq_prices_norm = (sequence_prices - sequence_prices.mean()) / (sequence_prices.std() + 1e-8)
        seq_scores_norm = (sequence_ranking_scores.float() - 3.0) / 2.0
        seq_positions_norm = (sequence_ranking_positions.float() - 30.0) / 30.0

        # Combine sequence features
        sequence_features = torch.cat([
            sequence_product_emb,
            sequence_text_embeddings,
            seq_prices_norm.unsqueeze(-1),
            seq_scores_norm.unsqueeze(-1),
            seq_positions_norm.unsqueeze(-1)
        ], dim=-1)

        # Target item features
        target_price_norm = (target_price - target_price.mean()) / (target_price.std() + 1e-8)
        target_score_norm = (target_ranking_score.float() - 3.0) / 2.0
        target_position_norm = (target_ranking_position.float() - 30.0) / 30.0

        target_features = torch.cat([
            target_product_emb,
            target_text_embedding,
            target_price_norm.unsqueeze(-1),
            target_score_norm.unsqueeze(-1),
            target_position_norm.unsqueeze(-1)
        ], dim=-1)

        # Interest extraction
        interest_states, _ = self.interest_extractor(sequence_features)

        # Interest evolution with attention
        evolved_interest, attention_weights = self.interest_evolution(
            sequence_features, target_features, sequence_mask
        )

        # Final features combination
        final_features = torch.cat([
            user_emb,
            evolved_interest,
            target_features
        ], dim=-1)

        # Prediction
        output = self.prediction_layers(final_features)

        return output.squeeze(), attention_weights


class InterestEvolutionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InterestEvolutionLayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Calculate the correct combined dimension
        combined_dim = hidden_dim + input_dim  # GRU output + target item features
        self.attention = AttentionLayer(combined_dim, combined_dim // 2)

    def forward(self, sequence, target_item, mask=None):
        # sequence: (batch_size, seq_len, input_dim)
        # target_item: (batch_size, input_dim)

        gru_output, _ = self.gru(sequence)  # (batch_size, seq_len, hidden_dim)

        # Attention mechanism with target item
        target_expanded = target_item.unsqueeze(1).expand(-1, gru_output.size(1), -1)
        combined = torch.cat([gru_output, target_expanded], dim=-1)

        # Apply attention
        final_state, attention_weights = self.attention(combined, mask)

        return final_state, attention_weights


# Alternative simpler approach
class InterestEvolutionLayerSimple(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InterestEvolutionLayerSimple, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Use only GRU output for attention (simpler and more stable)
        self.attention = AttentionLayer(hidden_dim, hidden_dim // 2)

        # Optional: Add target item influence through a separate transformation
        #self.target_transform = nn.Linear(input_dim, hidden_dim // 2)
        # change for dimension
        self.target_transform = nn.Linear(input_dim, hidden_dim)

        #self.target_proj = nn.Linear(embedding_dim, hidden_size)  # hidden_size = 256

    def forward(self, sequence, target_item, mask=None):
        # sequence: (batch_size, seq_len, input_dim)
        # target_item: (batch_size, input_dim)

        gru_output, _ = self.gru(sequence)  # (batch_size, seq_len, hidden_dim)

        # Apply attention to GRU output only
        final_state, attention_weights = self.attention(gru_output, mask)

        # Add target item influence
        target_influence = self.target_transform(target_item)
        final_state = final_state + target_influence

        return final_state, attention_weights


class DIENModelSimple(nn.Module):
    def __init__(self, n_users, n_products, embed_dim=64, text_embed_dim=128,
                 hidden_dim=256, seq_len=10):
        super(DIENModelSimple, self).__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.text_embed_dim = text_embed_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(n_users + 1, embed_dim)
        self.product_embedding = nn.Embedding(n_products + 1, embed_dim)

        # Calculate input dimension for sequences
        self.input_dim = embed_dim + text_embed_dim + 3  # product_emb + text_emb + 3 features

        # Interest extraction layer (GRU)
        self.interest_extractor = nn.GRU(
            self.input_dim,
            hidden_dim,
            batch_first=True
        )

        # Interest evolution layer (using simple version)
        self.interest_evolution = InterestEvolutionLayerSimple(
            self.input_dim,
            hidden_dim
        )

        # Final prediction layers
        # With InterestEvolutionLayerSimple, evolved_interest has dimension hidden_dim
        evolved_interest_dim = hidden_dim
        feature_dim = embed_dim + evolved_interest_dim + self.input_dim

        self.prediction_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            #nn.Linear(hidden_dim // 2, 1),
            #nn.Sigmoid()
            nn.Linear(hidden_dim // 2, 1)
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

    def forward(self, user_ids, target_product_ids, sequence_product_ids,
                sequence_text_embeddings, sequence_prices, sequence_ranking_scores,
                sequence_ranking_positions, target_text_embedding, target_price,
                target_ranking_score, target_ranking_position, sequence_mask):

        batch_size = user_ids.size(0)

        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        target_product_emb = self.product_embedding(target_product_ids)

        # Sequence embeddings
        sequence_product_emb = self.product_embedding(sequence_product_ids)

        # Normalize numerical features for sequence
        seq_prices_norm = (sequence_prices - sequence_prices.mean()) / (sequence_prices.std() + 1e-8)
        seq_scores_norm = (sequence_ranking_scores.float() - 3.0) / 2.0
        seq_positions_norm = (sequence_ranking_positions.float() - 30.0) / 30.0

        # Combine sequence features
        sequence_features = torch.cat([
            sequence_product_emb,
            sequence_text_embeddings,
            seq_prices_norm.unsqueeze(-1),
            seq_scores_norm.unsqueeze(-1),
            seq_positions_norm.unsqueeze(-1)
        ], dim=-1)

        # Target item features
        target_price_norm = (target_price - target_price.mean()) / (target_price.std() + 1e-8)
        target_score_norm = (target_ranking_score.float() - 3.0) / 2.0
        target_position_norm = (target_ranking_position.float() - 30.0) / 30.0

        target_features = torch.cat([
            target_product_emb,
            target_text_embedding,
            target_price_norm.unsqueeze(-1),
            target_score_norm.unsqueeze(-1),
            target_position_norm.unsqueeze(-1)
        ], dim=-1)

        # Interest extraction
        interest_states, _ = self.interest_extractor(sequence_features)

        # Interest evolution with attention
        evolved_interest, attention_weights = self.interest_evolution(
            sequence_features, target_features, sequence_mask
        )

        # Final features combination
        final_features = torch.cat([
            user_emb,
            evolved_interest,
            target_features
        ], dim=-1)

        # Prediction
        output = self.prediction_layers(final_features)

        # FIXED: Use squeeze(-1) to only remove the last dimension, preserving batch dimension
        # This ensures output shape is (batch_size,) instead of potentially () for batch_size=1
        return output.squeeze(-1), attention_weights



# 4. Training Sample Generation for DIEN
class DIENTrainingSampleGenerator:
    def __init__(self, df, text_embedder, seq_len=10):
        self.df = df
        self.text_embedder = text_embedder
        self.seq_len = seq_len

    def generate_samples(self):
        # Sort by user and timestamp
        df_sorted = self.df.sort_values(['user_id', 'timestamp'])

        # Group by user
        user_groups = df_sorted.groupby('user_id')

        samples = []
        for user_id, group in user_groups:
            group_list = group.to_dict('records')

            # Generate sequences for each user
            for i in range(1, len(group_list)):  # Start from 1 to have at least one history
                target_item = group_list[i]

                # Get sequence (history items)
                start_idx = max(0, i - self.seq_len)
                sequence = group_list[start_idx:i]

                # Pad sequence if needed
                while len(sequence) < self.seq_len:
                    # Add padding item
                    padding_item = {
                        'product_id': 0,
                        'comments': '',
                        'price': 0.0,
                        'ranking_score': 0,
                        'ranking_position': 0
                    }
                    sequence.insert(0, padding_item)

                # Take last seq_len items
                sequence = sequence[-self.seq_len:]

                sample = {
                    'user_id': user_id,
                    'target_product_id': target_item['product_id'],
                    'target_comments': target_item['comments'],
                    'target_price': target_item['price'],
                    'target_ranking_score': target_item['ranking_score'],
                    'target_ranking_position': target_item['ranking_position'],
                    'sequence_product_ids': [item['product_id'] for item in sequence],
                    'sequence_comments': [item['comments'] for item in sequence],
                    'sequence_prices': [item['price'] for item in sequence],
                    'sequence_ranking_scores': [item['ranking_score'] for item in sequence],
                    'sequence_ranking_positions': [item['ranking_position'] for item in sequence],
                    'sequence_mask': [1 if item['product_id'] != 0 else 0 for item in sequence],
                    'is_clicked': target_item['is_clicked'],
                    'is_purchased': target_item['is_purchased'],
                    'timestamp': target_item['timestamp']
                }
                samples.append(sample)

        return samples


# 5. Model Training
def train_dien_model(model, text_embedder, train_samples, val_samples,
                     epochs=50, lr=0.001, batch_size=256):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #criterion = nn.BCELoss()
    # Change BCELoss to BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    n_train = len(train_samples)
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle training data
        random.shuffle(train_samples)

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_train)
            batch_samples = train_samples[start_idx:end_idx]

            if len(batch_samples) == 0:
                continue

            # Prepare batch data
            batch_user_ids = torch.tensor([s['user_id'] for s in batch_samples], dtype=torch.long)
            batch_target_product_ids = torch.tensor([s['target_product_id'] for s in batch_samples], dtype=torch.long)
            batch_sequence_product_ids = torch.tensor([s['sequence_product_ids'] for s in batch_samples],
                                                      dtype=torch.long)
            batch_sequence_prices = torch.tensor([s['sequence_prices'] for s in batch_samples], dtype=torch.float)
            batch_sequence_ranking_scores = torch.tensor([s['sequence_ranking_scores'] for s in batch_samples],
                                                         dtype=torch.long)
            batch_sequence_ranking_positions = torch.tensor([s['sequence_ranking_positions'] for s in batch_samples],
                                                            dtype=torch.long)
            batch_target_prices = torch.tensor([s['target_price'] for s in batch_samples], dtype=torch.float)
            batch_target_ranking_scores = torch.tensor([s['target_ranking_score'] for s in batch_samples],
                                                       dtype=torch.long)
            batch_target_ranking_positions = torch.tensor([s['target_ranking_position'] for s in batch_samples],
                                                          dtype=torch.long)
            batch_sequence_mask = torch.tensor([s['sequence_mask'] for s in batch_samples], dtype=torch.float)
            batch_clicks = torch.tensor([s['is_clicked'] for s in batch_samples], dtype=torch.float)

            # Generate text embeddings
            batch_target_comments = [s['target_comments'] for s in batch_samples]
            batch_target_text_emb = text_embedder(batch_target_comments).detach()

            # Generate sequence text embeddings
            batch_sequence_text_emb = []
            for sample in batch_samples:
                seq_emb = text_embedder(sample['sequence_comments']).detach()
                batch_sequence_text_emb.append(seq_emb)
            batch_sequence_text_emb = torch.stack(batch_sequence_text_emb)

            optimizer.zero_grad()

            predictions, attention_weights = model(
                batch_user_ids, batch_target_product_ids, batch_sequence_product_ids,
                batch_sequence_text_emb, batch_sequence_prices, batch_sequence_ranking_scores,
                batch_sequence_ranking_positions, batch_target_text_emb, batch_target_prices,
                batch_target_ranking_scores, batch_target_ranking_positions, batch_sequence_mask
            )

            loss = criterion(predictions, batch_clicks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / n_batches)

        # Validation
        if epoch % 10 == 0:
            val_loss = evaluate_model_loss(model, text_embedder, val_samples, criterion)
            val_losses.append(val_loss)
            print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses


def evaluate_model_loss(model, text_embedder, samples, criterion):
    model.eval()
    total_loss = 0
    n_samples = len(samples)

    with torch.no_grad():
        for sample in samples:
            user_id = torch.tensor([sample['user_id']], dtype=torch.long)
            target_product_id = torch.tensor([sample['target_product_id']], dtype=torch.long)
            sequence_product_ids = torch.tensor([sample['sequence_product_ids']], dtype=torch.long)
            sequence_prices = torch.tensor([sample['sequence_prices']], dtype=torch.float)
            sequence_ranking_scores = torch.tensor([sample['sequence_ranking_scores']], dtype=torch.long)
            sequence_ranking_positions = torch.tensor([sample['sequence_ranking_positions']], dtype=torch.long)
            target_price = torch.tensor([sample['target_price']], dtype=torch.float)
            target_ranking_score = torch.tensor([sample['target_ranking_score']], dtype=torch.long)
            target_ranking_position = torch.tensor([sample['target_ranking_position']], dtype=torch.long)
            sequence_mask = torch.tensor([sample['sequence_mask']], dtype=torch.float)
            click = torch.tensor([sample['is_clicked']], dtype=torch.float)

            target_text_emb = text_embedder([sample['target_comments']]).detach()
            sequence_text_emb = text_embedder(sample['sequence_comments']).unsqueeze(0).detach()

            prediction, _ = model(
                user_id, target_product_id, sequence_product_ids,
                sequence_text_emb, sequence_prices, sequence_ranking_scores,
                sequence_ranking_positions, target_text_emb, target_price,
                target_ranking_score, target_ranking_position, sequence_mask
            )

            loss = criterion(prediction, click)
            total_loss += loss.item()

    return total_loss / n_samples


# 6. Model Evaluation
def evaluate_dien_model(model, text_embedder, test_samples):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for sample in test_samples:
            user_id = torch.tensor([sample['user_id']], dtype=torch.long)
            target_product_id = torch.tensor([sample['target_product_id']], dtype=torch.long)
            sequence_product_ids = torch.tensor([sample['sequence_product_ids']], dtype=torch.long)
            sequence_prices = torch.tensor([sample['sequence_prices']], dtype=torch.float)
            sequence_ranking_scores = torch.tensor([sample['sequence_ranking_scores']], dtype=torch.long)
            sequence_ranking_positions = torch.tensor([sample['sequence_ranking_positions']], dtype=torch.long)
            target_price = torch.tensor([sample['target_price']], dtype=torch.float)
            target_ranking_score = torch.tensor([sample['target_ranking_score']], dtype=torch.long)
            target_ranking_position = torch.tensor([sample['target_ranking_position']], dtype=torch.long)
            sequence_mask = torch.tensor([sample['sequence_mask']], dtype=torch.float)

            target_text_emb = text_embedder([sample['target_comments']]).detach()
            sequence_text_emb = text_embedder(sample['sequence_comments']).unsqueeze(0).detach()

            #prediction, _ = model(
            prediction_logits, _ = model(
                user_id, target_product_id, sequence_product_ids,
                sequence_text_emb, sequence_prices, sequence_ranking_scores,
                sequence_ranking_positions, target_text_emb, target_price,
                target_ranking_score, target_ranking_position, sequence_mask
            )

            # Apply sigmoid to convert logits to probabilities for metrics
            prediction_prob = torch.sigmoid(prediction_logits).item()

            predictions.append(prediction_prob)
            true_labels.append(sample['is_clicked'])

    # Calculate metrics
    auc = roc_auc_score(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)

    # NDCG calculation - group by user
    ndcg_scores = []
    user_groups = defaultdict(list)
    for i, sample in enumerate(test_samples):
        user_groups[sample['user_id']].append({
            'relevance': sample['target_ranking_score'],
            'prediction': predictions[i]
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
        'AUC': auc,
        'MSE': mse,
        'NDCG': avg_ndcg
    }


# 7. Main Execution
if __name__ == "__main__":
    print("=== Generating Ranking Log ===")
    generator = RankingLogGenerator(n_samples=1000, n_users=50, n_products=200)
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

    print("\n=== Generating DIEN Training Samples ===")
    sample_generator = DIENTrainingSampleGenerator(df, text_embedder, seq_len=10)
    all_samples = sample_generator.generate_samples()

    # Filter samples with sufficient history
    valid_samples = [s for s in all_samples if sum(s['sequence_mask']) > 0]

    print(f"Generated {len(valid_samples)} valid sequence samples")

    # Split samples
    train_samples, temp_samples = train_test_split(valid_samples, test_size=0.4, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    print("\n=== Training DIEN Model ===")
    #model = DIENModel(
    model=DIENModelSimple(
            n_users=df['user_id'].max(),
        n_products=df['product_id'].max(),
        embed_dim=64,
        text_embed_dim=128,
        hidden_dim=256,
        seq_len=10
    )

    train_losses, val_losses = train_dien_model(
        model, text_embedder, train_samples, val_samples,
        epochs=50, lr=0.001, batch_size=64
    )

    print("\n=== Evaluating DIEN Model ===")
    metrics = evaluate_dien_model(model, text_embedder, test_samples)

    print("Final Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== Training Complete ===")
    print("DIEN Model successfully trained and evaluated!")
    print(f"Final AUC: {metrics['AUC']:.4f}")
    print(f"Final MSE: {metrics['MSE']:.4f}")
    print(f"Final NDCG: {metrics['NDCG']:.4f}")