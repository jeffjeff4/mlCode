import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Configuration
config = {
    'batch_size': 256,
    'learning_rate': 0.001,
    'epochs': 20,
    'embedding_dim': 128,
    'margin': 1.0,  # Margin for contrastive loss
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'seed': 42
}

# Set random seed for reproducibility
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])


# Dataset and DataLoader
class ContrastiveMNIST(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.labels = [label for _, label in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]

        # Randomly choose positive or negative pair
        if np.random.rand() > 0.5:
            # Positive pair (same class)
            idx2 = np.random.choice(np.where(np.array(self.labels) == label1)[0])
            target = 1.0
        else:
            # Negative pair (different class)
            idx2 = np.random.choice(np.where(np.array(self.labels) != label1)[0])
            target = 0.0

        img2, _ = self.dataset[idx2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(target, dtype=torch.float32)


# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = MNIST(root='./data', train=True, download=True)
test_dataset = MNIST(root='./data', train=False, download=True)

# Create contrastive datasets
contrastive_train = ContrastiveMNIST(train_dataset, transform=transform)
contrastive_test = ContrastiveMNIST(test_dataset, transform=transform)

# Create dataloaders
train_loader = DataLoader(
    contrastive_train,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers']
)

test_loader = DataLoader(
    contrastive_test,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers']
)


# Model definition
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, config['embedding_dim'])
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        # Calculate Euclidean distance
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)

        # Contrastive loss
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# Initialize model, loss, and optimizer
model = EmbeddingNet().to(config['device'])
criterion = ContrastiveLoss(margin=config['margin'])
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


# Training function
def train(model, criterion, optimizer, train_loader, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (img1, img2, target) in enumerate(train_loader):
        img1, img2, target = img1.to(config['device']), img2.to(config['device']), target.to(config['device'])

        optimizer.zero_grad()
        output1 = model(img1)
        output2 = model(img2)
        loss = criterion(output1, output2, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(img1)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    print(f'\nTrain set: Average loss: {avg_loss:.4f}\n')
    return avg_loss


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for img1, img2, target in test_loader:
            img1, img2, target = img1.to(config['device']), img2.to(config['device']), target.to(config['device'])
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Test set: Average loss: {avg_loss:.4f}\n')
    return avg_loss


# Function to extract embeddings
def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for img1, img2, target in loader:
            img1, img2 = img1.to(config['device']), img2.to(config['device'])
            output1 = model(img1)
            output2 = model(img2)

            # Use the first image of each pair
            embeddings.append(output1.cpu().numpy())
            labels.append(target.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels


# Training loop
train_losses = []
test_losses = []

for epoch in range(1, config['epochs'] + 1):
    train_loss = train(model, criterion, optimizer, train_loader, epoch)
    test_loss = evaluate(model, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, config['epochs'] + 1), train_losses, label='Train Loss')
plt.plot(range(1, config['epochs'] + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# Evaluate embeddings with k-NN classifier
print("Evaluating embeddings with k-NN classifier...")

# Get embeddings for training set
train_embeddings, train_labels = extract_embeddings(model, train_loader)

# Get embeddings for test set
test_embeddings, test_labels = extract_embeddings(model, test_loader)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, train_labels)

# Predict on test set
pred_labels = knn.predict(test_embeddings)

# Calculate accuracy
accuracy = accuracy_score(test_labels, pred_labels)
print(f"k-NN Accuracy: {accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'contrastive_model.pth')
print("Model saved to contrastive_model.pth")