import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # or XGBRegressor
from sklearn.metrics import accuracy_score

# ============================================
# 1. Generate Synthetic Data
# ============================================
num_samples = 1000
num_features = 20
num_classes = 3

# Create random data
X = np.random.randn(num_samples, num_features)
y = np.random.randint(0, num_classes, size=num_samples)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)


# ============================================
# 2. Create Embedding Model (PyTorch)
# ============================================
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


# Initialize model
embedding_dim = 16
embedder = EmbeddingModel(num_features, embedding_dim)


# ============================================
# 3. Train Embedding Model (Optional)
# ============================================
# Only needed if you want task-specific embeddings
def train_embedder(model, X, y, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# Uncomment to train:
train_embedder(embedder, X_train_tensor, y_train_tensor)

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
# 5. Train XGBoost on Embeddings
# ============================================
xgb_model = XGBClassifier(n_estimators=100, max_depth=3)
xgb_model.fit(X_train_emb, y_train)

# Evaluate
y_pred = xgb_model.predict(X_test_emb)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.4f}")

# Feature importance
print("\nFeature Importances:")
for i, imp in enumerate(xgb_model.feature_importances_):
    print(f"Embedding dim {i}: {imp:.4f}")