import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample synthetic dataset (texts and binary labels: 1=positive, 0=negative)
texts = ["good movie", "bad film", "great story", "terrible plot", "awesome acting", "poor direction"]
labels = [1, 0, 1, 0, 1, 0]

# Build vocabulary (unique words to indices)
vocab = list(set(" ".join(texts).split()))
word_to_ix = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# Custom Dataset for text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_ix):
        self.texts = [torch.tensor([word_to_ix[w] for w in text.split()]) for text in texts]  # Convert words to indices
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Collate function to pad variable-length sequences
def collate(batch):
    texts = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)  # Pad with 0
    return texts_padded, labels

# Model: Embedding + RNN + Linear (deep learning with recurrent layers)
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Word embeddings
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)  # Simple RNN (can replace with LSTM/GRU)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Classifier

    def forward(self, x):
        # Forward propagation
        x = self.embedding(x)  # (batch, seq_len) -> (batch, seq_len, embed_dim)
        rnn_out, hn = self.rnn(x)  # RNN processes sequence, hn is final hidden state
        out = self.fc(hn.squeeze(0))  # (batch, hidden_dim) -> (batch, num_classes)
        return out

# Hyperparameters
embed_dim = 16
hidden_dim = 32
num_classes = 2  # Binary classification
batch_size = 2
epochs = 10
learning_rate = 0.01

# Prepare data
dataset = TextDataset(texts, labels, word_to_ix)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

# Initialize model, loss, optimizer
model = TextClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()  # Set to training mode
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(inputs)  # Forward propagation
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()  # Backward propagation (computes gradients)
        optimizer.step()  # Update weights
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

# Evaluation (simple accuracy on training data for demo)
model.eval()  # Set to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Training Accuracy: {accuracy:.2f}%")