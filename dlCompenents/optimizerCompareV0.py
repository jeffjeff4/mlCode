##please generate pytorch code:
##1. a training dataset with numerical features, categorical features, embedding features, and a label feature, label is a numerical feature
##2. generate code to train a simple deep learing model, mlp, for the regression task
##3. implement code for these optimizers, using 1) SGD, 2) sgd with momentum, 3) NAG as optimizer, 4) adagrad, 5) rmsprop, 6) adadelta, 7) adam, 8) adamw, 9) nadam
##do not just call these optimizers from existing libraries. generate code to implement these optimizers
##4. training and test the model, by using different optimizers
##
##part 1: PyTorch Dataset + MLP Model
##
##Part 2: Implement custom optimizers (SGD, Adam, RMSProp, etc.)
##in the above code,
##adam: p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
##adamw: p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)
##
##it does NOT include the differences as you mentioned:
##1) AdamW differs from Adam in how weight decay is applied:
##Adam applies L2 penalty through gradient.
##AdamW directly decays weights (decoupled).
##2)
##Adam  Gradient directly (L2)  ✅ Yes + λ * grad
##AdamW Parameters themselves ✅ Yes + λ * weight
##please implement the correct code of adam and adamw
##
##
##part 3:
##please do Part 3: Training & evaluation loop with each optimizer (including loss curve)
##1. full training loop that compares these optimizers on a synthetic regression task?
##2. Plot of training loss curves for visual comparison?
##especially do below
##3. training loop using both Adam and AdamW side by side?
##4. Experiments comparing Adam vs AdamW on training/test loss
##5. Plots showing effect of weight decay




import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- Dataset ---
class CustomDataset(Dataset):
    def __init__(self, n_samples=1000, n_num=5, n_cat=3, n_emb=2):
        self.num_data = torch.randn(n_samples, n_num)
        self.cat_data = torch.randint(0, 5, (n_samples, n_cat))
        self.emb_data = torch.randn(n_samples, n_emb)
        self.labels = (
            self.num_data.sum(1) +
            self.cat_data.float().sum(1) +
            self.emb_data.sum(1) +
            torch.randn(n_samples) * 0.5
        ).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.num_data[idx], self.cat_data[idx], self.emb_data[idx], self.labels[idx]

# --- MLP Model ---
class MLPModel(nn.Module):
    def __init__(self, n_num=5, n_cat=3, n_emb=2, cat_vocab=5, cat_emb_dim=4):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cat_vocab, cat_emb_dim) for _ in range(n_cat)
        ])
        input_dim = n_num + n_cat * cat_emb_dim + n_emb
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, num, cat, emb):
        cat_emb = [self.emb_layers[i](cat[:, i]) for i in range(cat.size(1))]
        x = torch.cat([num] + cat_emb + [emb], dim=1)
        return self.net(x)

# --- Optimizer Base ---
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
            g = p.grad + self.weight_decay * p.data  # L2 penalty
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
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
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

            #this is the difference between Adam and Adamw
            p.data -= self.lr * self.weight_decay * p.data  # decoupled

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


def train_and_eval(optimizer_class, optimizer_name, epochs=20, batch_size=64, plot=True, **opt_kwargs):
    dataset = CustomDataset(n_samples=1000)
    train_set, test_set = torch.utils.data.random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = MLPModel()
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for num, cat, emb, y in train_loader:
            optimizer.zero_grad()
            out = model(num, cat, emb)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * num.size(0)
        train_losses.append(running_loss / len(train_loader.dataset))

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for num, cat, emb, y in test_loader:
                pred = model(num, cat, emb)
                loss = criterion(pred, y)
                total_test_loss += loss.item() * num.size(0)
        test_losses.append(total_test_loss / len(test_loader.dataset))

    # Plot
    plt.plot(train_losses, label=f"{optimizer_name} - Train")
    plt.plot(test_losses, linestyle='--', label=f"{optimizer_name} - Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training with {optimizer_name}")
    plt.legend()
    plt.grid(True)
    if plot==True:
        plt.show()
    return train_losses, test_losses



# 1. Adam vs AdamW (with and without weight decay)
train_and_eval(Adam, "Adam (no decay)", lr=0.001, weight_decay=0.0)
train_and_eval(Adam, "Adam (L2 decay)", lr=0.001, weight_decay=0.01)
train_and_eval(AdamW, "AdamW (decoupled)", lr=0.001, weight_decay=0.0)
train_and_eval(AdamW, "AdamW (decoupled)", lr=0.001, weight_decay=0.01)

# compare adam and adamw

# Run experiments
optimizers_to_compare = {
    "Adam (no decay)": (Adam, {"lr": 0.001, "weight_decay": 0.0}),
    "Adam (L2 0.001)": (Adam, {"lr": 0.001, "weight_decay": 0.01}),
    "AdamW (wd=0.01)": (AdamW, {"lr": 0.001, "weight_decay": 0.01}),
    "AdamW (wd=0.1)": (AdamW, {"lr": 0.001, "weight_decay": 0.01}),
}

results = {}

for name, (opt_class, kwargs) in optimizers_to_compare.items():
    print(f"Training with {name}")
    #train_loss, test_loss = train_model(opt_class, name, **kwargs)
    train_loss, test_loss = train_and_eval(opt_class, name, **kwargs, plot=False)

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

# 2. RMSprop
train_and_eval(RMSprop, "RMSprop", lr=0.001)

# 3. Adagrad
train_and_eval(Adagrad, "Adagrad", lr=0.001)

# 4. Nadam
train_and_eval(Nadam, "Nadam", lr=0.001)

# 5. SGD
train_and_eval(SGD, "SGD", lr=0.001)

# 6. SGDMomentum
train_and_eval(SGDMomentum, "SGDMomentum", lr=0.001)
