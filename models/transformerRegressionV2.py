import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

# ======================
# 1. Synthetic Data Setup
# ======================
num_samples = 1000
feature_dim = 20  # Original feature dimension
embedding_dim = 32  # Output of Model 1

# Generate random data
X = torch.randn(num_samples, feature_dim)  # Original features
y = torch.randn(num_samples, 1)  # Regression targets

# Split into train/val
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# ======================
# 2. Model Definitions
# ======================

# Model 1: Embedding Generator
class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)


# Model 2: Transformer Regressor
class TransformerRegressor(nn.Module):
    def __init__(self, embedding_dim, nhead=4):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.regressor = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Add batch dimension for transformer (seq_len=1 for our case)
        x = x.unsqueeze(1)  # [batch, 1, embedding_dim]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.squeeze(1)  # [batch, embedding_dim]
        return self.regressor(x)


# Positional Encoding (for Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x


# ======================
# 3. Training Setup
# ======================
embedding_model = EmbeddingModel(feature_dim, embedding_dim)
transformer_model = TransformerRegressor(embedding_dim)

# We'll train them end-to-end
model = nn.Sequential(embedding_model, transformer_model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


# ======================
# 4. Training Loop
# ======================
def train(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            # Forward pass through both models
            embeddings = embedding_model(batch_X)  # [batch, embedding_dim]
            outputs = transformer_model(embeddings)  # [batch, 1]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")


# Train the model
train(model, train_loader, optimizer, criterion)


# ======================
# 5. Inference Example
# ======================
def predict(model, x):
    model.eval()
    with torch.no_grad():
        embeddings = embedding_model(x)
        predictions = transformer_model(embeddings)
    return predictions


# Test on a sample
test_sample = torch.randn(1, feature_dim)  # Single sample
print("Prediction:", predict(model, test_sample).item())