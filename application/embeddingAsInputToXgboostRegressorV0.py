import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# 1. Generate Synthetic Regression Data
# ============================================
num_samples = 1000
num_features = 20

# Create random data with a simple relationship
X = np.random.randn(num_samples, num_features)
# Create target with some non-linearity
y = 5 * X[:, 0] + 2 * np.sin(X[:, 1]) + 0.5 * np.random.randn(num_samples)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # XGB needs 1D array
X_test_tensor = torch.FloatTensor(X_test)


# ============================================
# 2. Create Embedding Model (PyTorch)
# ============================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Initialize model
embedding_dim = 10  # Reduced dimension

# Initialize as autoencoder
embedder = Autoencoder(num_features, embedding_dim)


# ============================================
# 3. Train Embedding Model (Optional)
# ============================================
def train_embedder(model, X, y, epochs=20):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# Train as autoencoder (reconstruct original features)
train_embedder(embedder, X_train_tensor, X_train_tensor)


# ============================================
# 4. Generate Embeddings
# ============================================
def get_embeddings(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).numpy()


X_train_emb = get_embeddings(embedder, X_train_tensor)
X_test_emb = get_embeddings(embedder, X_test_tensor)

# ============================================
# 5. Train XGBoost Regressor on Embeddings
# ============================================
xgb_model = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train_emb, y_train)  # y_train is original targets

# Evaluate
y_pred = xgb_model.predict(X_test_emb)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nXGBoost Regression Results:")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
print("\nEmbedding Dimension Importance:")
for i, imp in enumerate(xgb_model.feature_importances_):
    print(f"Dim {i}: {imp:.4f}")