

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset with numerical, categorical, and embedding features
class CustomDataset(Dataset):
    def __init__(self, n_samples=1000, n_num=5, n_cat=3, n_emb=2):
        super().__init__()
        self.num_data = torch.randn(n_samples, n_num)
        self.cat_data = torch.randint(0, 5, (n_samples, n_cat))
        self.emb_data = torch.randn(n_samples, n_emb)
        self.labels = (
            self.num_data.sum(dim=1)
            + self.cat_data.float().sum(dim=1)
            + self.emb_data.sum(dim=1)
            + torch.randn(n_samples) * 0.5
        ).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.num_data[idx], self.cat_data[idx], self.emb_data[idx], self.labels[idx]

# Simple MLP model for regression
class MLPModel(nn.Module):
    def __init__(self, n_num=5, n_cat=3, n_emb=2, cat_vocab=5, cat_emb_dim=4):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cat_vocab, cat_emb_dim) for _ in range(n_cat)
        ])
        total_input_dim = n_num + n_cat * cat_emb_dim + n_emb
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, num, cat, emb):
        cat_emb = [self.emb_layers[i](cat[:, i]) for i in range(cat.size(1))]
        x = torch.cat([num] + cat_emb + [emb], dim=1)
        return self.net(x)


class CustomOptimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

class SGD(CustomOptimizer):
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

class SGDMomentum(CustomOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.v[i] = self.momentum * self.v[i] - self.lr * p.grad
                p.data += self.v[i]


class Adam(CustomOptimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # ✅ L2 weight decay: add to gradient
            grad = p.grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


class AdamW(CustomOptimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad  # ✅ NO weight decay added to grad

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # ✅ Decoupled weight decay applied directly to weights
            p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps))
            p.data -= self.lr * self.weight_decay * p.data


class RMSprop(CustomOptimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.avg_sq = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.avg_sq[i] = self.alpha * self.avg_sq[i] + (1 - self.alpha) * g ** 2
            p.data -= self.lr * g / (self.avg_sq[i].sqrt() + self.eps)


class Adagrad(CustomOptimizer):
    def __init__(self, params, lr=0.01, eps=1e-10):
        super().__init__(params, lr)
        self.eps = eps
        self.G = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.G[i] += g ** 2
            adjusted_lr = self.lr / (self.G[i].sqrt() + self.eps)
            p.data -= adjusted_lr * g


class Nadam(CustomOptimizer):
    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            m_hat = (self.beta1 * self.m[i] + (1 - self.beta1) * g) / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


import matplotlib.pyplot as plt

# Dataset and model config
dataset = CustomDataset(n_samples=1000)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Simple train-test split
train_set, test_set = torch.utils.data.random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

# Training loop
def train_model(optimizer_class, optimizer_name, model_seed=0, epochs=20, **opt_kwargs):
    torch.manual_seed(model_seed)
    model = MLPModel()
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for num, cat, emb, label in train_loader:
            optimizer.zero_grad()
            output = model(num, cat, emb)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * num.size(0)
        train_losses.append(total_loss / len(train_loader.dataset))

        # Evaluate on test set
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for num, cat, emb, label in test_loader:
                output = model(num, cat, emb)
                loss = criterion(output, label)
                total_test_loss += loss.item() * num.size(0)
        test_losses.append(total_test_loss / len(test_loader.dataset))

    return train_losses, test_losses


# Run experiments
optimizers_to_compare = {
    "Adam (no decay)": (Adam, {"lr": 0.001, "weight_decay": 0.0}),
    "Adam (L2 0.01)": (Adam, {"lr": 0.001, "weight_decay": 0.01}),
    "AdamW (wd=0.01)": (AdamW, {"lr": 0.001, "weight_decay": 0.01}),
    "AdamW (wd=0.1)": (AdamW, {"lr": 0.001, "weight_decay": 0.1}),
}

results = {}

for name, (opt_class, kwargs) in optimizers_to_compare.items():
    print(f"Training with {name}")
    train_loss, test_loss = train_model(opt_class, name, **kwargs)
    results[name] = {"train": train_loss, "test": test_loss}

# Plot
plt.figure(figsize=(12, 6))
for name in results:
    plt.plot(results[name]["train"], label=f"{name} - Train")
    plt.plot(results[name]["test"], linestyle="--", label=f"{name} - Test")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Adam vs AdamW (Different Weight Decays)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
