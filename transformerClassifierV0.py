import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src):
        # src shape: (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # Take the mean across sequence dimension
        output = output.mean(dim=0)
        return self.fc(output)


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 10000  # Size of your vocabulary
    batch_size = 32
    seq_len = 50
    num_classes = 2  # Binary classification

    # Create model
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=128,
        nhead=8,
        num_layers=3,
        num_classes=num_classes
    )

    # Create dummy data (replace with real data)
    # Shape: (seq_len, batch_size)
    dummy_input = torch.randint(0, vocab_size, (seq_len, batch_size))

    # Forward pass
    outputs = model(dummy_input)
    print(f"Output shape: {outputs.shape}")  # Should be [batch_size, num_classes]

    # Training setup example
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Dummy labels
    labels = torch.randint(0, num_classes, (batch_size,))

    # Training step
    optimizer.zero_grad()
    outputs = model(dummy_input)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item()}")

