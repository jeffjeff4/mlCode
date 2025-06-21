import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)


# 2. Dataset with proper transforms
class TripletDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = np.array(labels)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in set(self.labels)}

    def __getitem__(self, index):
        anchor_img = self.data[index]
        anchor_label = self.labels[index]

        positive_idx = np.random.choice(self.label_to_indices[anchor_label])
        negative_label = np.random.choice(list(set(self.labels) - {anchor_label}))
        negative_idx = np.random.choice(self.label_to_indices[negative_label])

        return (self.transform(anchor_img),
                self.transform(self.data[positive_idx]),
                self.transform(self.data[negative_idx]))

    def __len__(self):
        return len(self.data)


# 3. Model with proper initialization
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.backbone(x), p=2, dim=1)


# 4. Training function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0

    for anchor, positive, negative in tqdm(loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        a_emb = model(anchor)
        p_emb = model(positive)
        n_emb = model(negative)

        loss = criterion(a_emb, p_emb, n_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch} Loss: {total_loss / len(loader):.4f}')


# Main
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    embedding_dim = 128
    lr = 1e-4
    epochs = 10

    # Example data - replace with real dataset
    train_data = np.random.randint(0, 255, (1000, 224, 224, 3), dtype=np.uint8)
    train_labels = np.random.randint(0, 10, 1000)

    # Initialize
    train_dataset = TripletDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingNet(embedding_dim).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch)