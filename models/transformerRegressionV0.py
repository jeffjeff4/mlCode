import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.linear_out = nn.Linear(d_model, 1)
        self.d_model = d_model

    #def forward(self, src):
    #    # src shape: (batch_size, seq_len, input_dim)
    #    src = self.linear_in(src) * math.sqrt(self.d_model)
    #    src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
    #    output = self.transformer(src)
    #    output = output.mean(dim=0)  # Average over sequence
    #    return self.linear_out(output).squeeze(-1)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        src = self.linear_in(src) * math.sqrt(self.d_model)
        output = self.transformer(src)
        output = output.mean(dim=1)  # Average over sequence
        return self.linear_out(output)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def predict(model, dataloader, device='cpu'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            output = model(X)
            predictions.append(output.cpu())
    return torch.cat(predictions)


# Example usage
if __name__ == "__main__":
    # 1. Generate synthetic time series data
    seq_len = 20
    n_samples = 1000
    input_dim = 5  # Number of features per timestep

    # Random data (replace with your actual data)
    X = np.random.randn(n_samples, seq_len, input_dim)
    y = np.random.randn(n_samples)  # Regression target

    # 2. Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, input_dim)).reshape(n_samples, seq_len, input_dim)
    y = (y - y.mean()) / y.std()

    # 3. Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 4. Create dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 5. Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRegressor(input_dim=input_dim, d_model=64, nhead=4, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 6. Training loop
    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}')

    # 7. Evaluation
    predictions = predict(model, test_loader, device)
    test_loss = criterion(predictions, torch.FloatTensor(y_test))
    print(f'Test MSE: {test_loss.item():.4f}')

    # 8. Make new predictions
    new_data = np.random.randn(3, seq_len, input_dim)  # 3 new samples
    new_data = scaler.transform(new_data.reshape(-1, input_dim)).reshape(3, seq_len, input_dim)
    new_loader = DataLoader(TimeSeriesDataset(new_data, np.zeros(3)), batch_size=3)
    preds = predict(model, new_loader, device)
    print('New predictions:', preds)