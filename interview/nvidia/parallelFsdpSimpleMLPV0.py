####python code, for machine learning problem, with test case
####
####   1. the code MUST be CORRECT
####   2. consider all cases, including edge cases
####   3. make it runnable in this chat session
####   4. provide time complexity and space complexity analysis
####   5. make the code runnable on cpu
####   6. generate simple test datasets
####   7. generate train and evaluation code
####   8. world_size = int(os.environ.get("WORLD_SIZE", "2"))
####
####Question1:
####
####using a simple mlp model, manually implement fsdp, by using below logic
####
####
####FSDP forward pass:
####    for layer_i in layers:
####        all-gather full weights for layer_i
####        forward pass for layer_i
####        discard full weights for layer_i
####
####FSDP backward pass:
####    for layer_i in layers:
####        all-gather full weights for layer_i
####        backward pass for layer_i
####        discard full weights for layer_i
####        reduce-scatter gradients for layer_i


#### this is a runnble version, could be used in an interview
#### this is from chatgpt

import os
import math
import numpy as np

# -----------------------------
# Utilities
# -----------------------------
def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss(logits, y):
    """
    logits: (N, C)
    y: (N,) int labels in [0, C-1]
    """
    probs = softmax(logits)
    N = logits.shape[0]
    # Avoid log(0)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    logp = -np.log(probs[np.arange(N), y])
    return np.mean(logp), probs

def one_hot(y, num_classes):
    N = len(y)
    oh = np.zeros((N, num_classes), dtype=np.float64)
    oh[np.arange(N), y] = 1.0
    return oh

def relu(x):
    return np.maximum(x, 0.0)

def relu_backward(grad_out, x_cached):
    g = grad_out.copy()
    g[x_cached <= 0.0] = 0.0
    return g

def shard_splits(total_rows, world_size):
    """
    Row-wise sharding sizes that handle non-even splits robustly.
    Returns sizes and start indices.
    """
    # Use numpy split sizes similar to np.array_split logic
    base = total_rows // world_size
    rem = total_rows % world_size
    sizes = [base + (1 if i < rem else 0) for i in range(world_size)]
    starts = [0]
    for i in range(1, world_size):
        starts.append(starts[-1] + sizes[i-1])
    return sizes, starts

# -----------------------------
# Linear layer shards (row-wise)
# -----------------------------
class ShardedLinear:
    """
    Row-sharded Linear: y = x @ W^T + b
    We shard W along its row/output dimension across ranks.
    For simulation, we keep all ranks' shards in this object.
    """
    def __init__(self, in_features, out_features, world_size, seed=0):
        rng = np.random.default_rng(seed)
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size

        sizes, starts = shard_splits(out_features, world_size)
        self.sizes = sizes
        self.starts = starts

        # Initialize shards: list of W_shard (rows_i, in_features), b_shard (rows_i,)
        self.W_shards = [
            rng.normal(0, 0.02, size=(sizes[i], in_features)).astype(np.float64)
            for i in range(world_size)
        ]
        self.b_shards = [
            np.zeros((sizes[i],), dtype=np.float64) for i in range(world_size)
        ]

        # Accumulators for reduce-scatter gradients (per rank)
        self.dW_shards_accum = [np.zeros_like(self.W_shards[i]) for i in range(world_size)]
        self.db_shards_accum = [np.zeros_like(self.b_shards[i]) for i in range(world_size)]

    def all_gather_full_weights(self):
        """Reconstruct full W, b by concatenating shards along rows."""
        W_full = np.concatenate(self.W_shards, axis=0)
        b_full = np.concatenate(self.b_shards, axis=0)
        return W_full, b_full

    def reduce_scatter_grads(self, dW_full, db_full):
        """Split full grads row-wise and accumulate into shard grad buffers."""
        # Slice per shard using sizes & starts
        for i in range(self.world_size):
            s = self.starts[i]
            e = s + self.sizes[i]
            self.dW_shards_accum[i] += dW_full[s:e, :]
            self.db_shards_accum[i] += db_full[s:e]

    def zero_grad_accum(self):
        for i in range(self.world_size):
            self.dW_shards_accum[i].fill(0.0)
            self.db_shards_accum[i].fill(0.0)

    def step(self, lr, world_size):
        """
        Apply SGD update using averaged grads across ranks:
        W_i -= lr * (dW_i_accum / world_size), similarly for b.
        """
        for i in range(self.world_size):
            self.W_shards[i] -= lr * (self.dW_shards_accum[i] / world_size)
            self.b_shards[i] -= lr * (self.db_shards_accum[i] / world_size)

# -----------------------------
# MLP with two Linear layers
# -----------------------------
class ShardedMLP_FSDP:
    """
    Simulated FSDP MLP:
    Layers: Linear(in->hidden), ReLU, Linear(hidden->num_classes)
    Uses FSDP-like logic:
      Forward per layer: all-gather full weights -> compute -> discard full
      Backward per layer: all-gather full weights -> compute -> discard -> reduce-scatter grads
    """
    def __init__(self, in_features, hidden, num_classes, world_size, seed=0):
        self.l1 = ShardedLinear(in_features, hidden, world_size, seed=seed+1)
        self.l2 = ShardedLinear(hidden, num_classes, world_size, seed=seed+2)
        self.world_size = world_size

    def forward(self, x):
        """
        Returns:
            logits, cache for backward (to avoid storing full weights)
        Cache stores intermediate activations needed for backprop.
        """
        cache = {}

        # Layer 1: all-gather full weights -> forward -> discard
        W1, b1 = self.l1.all_gather_full_weights()
        z1 = x @ W1.T + b1  # (N, hidden)
        a1 = relu(z1)
        cache['x'] = x
        cache['z1'] = z1  # for relu backward
        cache['a1'] = a1

        # Layer 2: all-gather full weights -> forward -> discard
        W2, b2 = self.l2.all_gather_full_weights()
        logits = a1 @ W2.T + b2  # (N, C)
        cache['W1_shape'] = W1.shape
        cache['W2_shape'] = W2.shape
        # Note: We do NOT store W1 or W2 (simulate discard)

        return logits, cache

    def backward_and_reduce_scatter(self, cache, dlogits):
        """
        Given dL/dlogits, run backward layer-by-layer with FSDP logic and
        reduce-scatter grads into shard accumulators.
        """
        a1 = cache['a1']
        z1 = cache['z1']
        x  = cache['x']

        # ---- Layer 2 backward ----
        # all-gather full weights for L2
        W2_full, _b2_full = self.l2.all_gather_full_weights()

        # dW2_full = dlogits^T @ a1
        dW2_full = dlogits.T @ a1  # (C, hidden)
        db2_full = np.sum(dlogits, axis=0)  # (C,)
        # dx for next layer: dL/da1 = dlogits @ W2_full
        da1 = dlogits @ W2_full  # (N, hidden)

        # reduce-scatter grads for L2
        self.l2.reduce_scatter_grads(dW2_full, db2_full)

        # ---- ReLU backward ----
        dz1 = relu_backward(da1, z1)  # (N, hidden)

        # ---- Layer 1 backward ----
        W1_full, _b1_full = self.l1.all_gather_full_weights()
        dW1_full = dz1.T @ x  # (hidden, in_features)
        db1_full = np.sum(dz1, axis=0)  # (hidden,)
        # dx not needed for input

        # reduce-scatter grads for L1
        self.l1.reduce_scatter_grads(dW1_full, db1_full)

    def zero_grad(self):
        self.l1.zero_grad_accum()
        self.l2.zero_grad_accum()

    def step(self, lr, world_size):
        self.l1.step(lr, world_size)
        self.l2.step(lr, world_size)

    def gather_full_model(self):
        """Return full (W1,b1,W2,b2) for evaluation/inference."""
        W1, b1 = self.l1.all_gather_full_weights()
        W2, b2 = self.l2.all_gather_full_weights()
        return (W1, b1, W2, b2)

# -----------------------------
# Data generation
# -----------------------------
def make_synthetic_data(n_samples=600, n_features=16, n_classes=3, seed=0):
    """
    Multiclass Gaussian blobs.
    """
    rng = np.random.default_rng(seed)
    X = []
    y = []
    centers = rng.normal(0, 3.0, size=(n_classes, n_features))
    for c in range(n_classes):
        Xi = centers[c] + rng.normal(0, 1.0, size=(n_samples // n_classes, n_features))
        yi = np.full((n_samples // n_classes,), c, dtype=int)
        X.append(Xi)
        y.append(yi)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx].astype(np.float64), y[idx]

def train_val_split(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y)
    idx = rng.permutation(N)
    cut = int(N * (1 - val_ratio))
    tr = idx[:cut]
    va = idx[cut:]
    return X[tr], y[tr], X[va], y[va]

# -----------------------------
# Training (simulated multi-rank loop)
# -----------------------------
def run_training(
    input_dim=16,
    hidden=32,
    num_classes=3,
    epochs=5,
    batch_size=64,
    lr=0.1,
    seed=0
):
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    print(f"Simulated world_size = {world_size}")

    # Data
    X, y = make_synthetic_data(n_samples=600, n_features=input_dim, n_classes=num_classes, seed=seed)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_ratio=0.2, seed=seed+1)

    # Simple batching helper
    def iterate_minibatches(Xd, yd, bs):
        N = len(yd)
        for i in range(0, N, bs):
            yield Xd[i:i+bs], yd[i:i+bs]

    # Model
    model = ShardedMLP_FSDP(input_dim, hidden, num_classes, world_size, seed=seed)

    # Training
    for ep in range(1, epochs+1):
        # Shuffle training data each epoch
        perm = np.random.permutation(len(ytr))
        Xtr_shuf, ytr_shuf = Xtr[perm], ytr[perm]
        losses = []
        correct = 0
        seen = 0

        for xb, yb in iterate_minibatches(Xtr_shuf, ytr_shuf, batch_size):
            # Zero accumulators (per step)
            model.zero_grad()

            # ---- Simulate synchronous multi-rank step ----
            # Split the batch across ranks (data parallel input split)
            splits, starts = shard_splits(len(xb), world_size)
            for rank in range(world_size):
                s = starts[rank]
                e = s + splits[rank]
                if s >= e:
                    continue  # handle tiny batches

                xb_r = xb[s:e]
                yb_r = yb[s:e]

                # Forward (FSDP-style per layer all-gather)
                logits, cache = model.forward(xb_r)
                loss, probs = cross_entropy_loss(logits, yb_r)
                losses.append(loss)

                # Accuracy tracking (per rank)
                preds = np.argmax(probs, axis=1)
                correct += np.sum(preds == yb_r)
                seen += len(yb_r)

                # Backward
                # dL/dlogits = (probs - one_hot)/N_rank  (average over local rank batch)
                oh = one_hot(yb_r, num_classes)
                dlogits = (probs - oh) / max(1, len(yb_r))
                model.backward_and_reduce_scatter(cache, dlogits)

            # After all ranks processed their local splits, apply optimizer step
            model.step(lr, world_size)

        # Epoch summary
        tr_loss = float(np.mean(losses)) if losses else float('nan')
        tr_acc = 100.0 * correct / max(1, seen)
        va_acc = evaluate(model, Xva, yva)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} | train acc {tr_acc:.2f}% | val acc {va_acc:.2f}%")

    # Final evaluation
    final_acc = evaluate(model, Xva, yva)
    print(f"Final validation accuracy: {final_acc:.2f}%")
    return model

def evaluate(model, X, y):
    # Gather full model weights to do an ordinary forward on CPU
    W1, b1, W2, b2 = model.gather_full_model()
    a1 = relu(X @ W1.T + b1)
    logits = a1 @ W2.T + b2
    preds = np.argmax(logits, axis=1)
    return 100.0 * np.mean(preds == y)

# -----------------------------
# Run (train + eval)
# -----------------------------
if __name__ == "__main__":
    # You can change WORLD_SIZE via environment, e.g.:
    # os.environ["WORLD_SIZE"] = "2"
    model = run_training(
        input_dim=16,
        hidden=32,
        num_classes=3,
        epochs=5,
        batch_size=64,
        lr=0.2,
        seed=42
    )
