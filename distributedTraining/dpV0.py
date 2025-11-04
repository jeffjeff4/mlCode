import numpy as np
from multiprocessing import Pool

# ------------------------------
# Activation functions
# ------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ------------------------------
# Dataset generation
# ------------------------------
def generate_dataset(num_samples=1000, input_dim=10, num_classes=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(num_samples, input_dim)
    true_W = np.random.randn(input_dim, num_classes)
    logits = X @ true_W
    y = np.argmax(logits + 0.5 * np.random.randn(*logits.shape), axis=1)
    y_one_hot = np.eye(num_classes)[y]
    return X, y, y_one_hot

# ------------------------------
# Parallel gradient computation
# ------------------------------
def compute_gradients(args):
    X_batch, y_batch, W1, b1, W2, b2 = args

    # Forward
    Z1 = X_batch @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Backward
    dZ2 = A2 - y_batch
    dW2 = (A1.T @ dZ2) / len(X_batch)
    db2 = np.sum(dZ2, axis=0) / len(X_batch)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (X_batch.T @ dZ1) / len(X_batch)
    db1 = np.sum(dZ1, axis=0) / len(X_batch)
    return dW1, db1, dW2, db2

# ------------------------------
# Evaluation
# ------------------------------
def evaluate(X, y, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    preds = np.argmax(Z2, axis=1)
    acc = np.mean(preds == y)
    return acc

# ------------------------------
# Training loop
# ------------------------------
def train_parallel(X, y_one_hot, y, input_dim, hidden_dim, output_dim,
                   learning_rate=0.1, epochs=20, num_devices=4):
    np.random.seed(0)

    # Initialize weights
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)

    for epoch in range(epochs):
        # Split data across "devices"
        batch_indices = np.array_split(range(X.shape[0]), num_devices)
        batch_data = [(X[idx], y_one_hot[idx], W1, b1, W2, b2) for idx in batch_indices]

        # Parallel gradient computation
        with Pool(num_devices) as pool:
            gradients = pool.map(compute_gradients, batch_data)

        # Aggregate gradients
        dW1 = np.mean([g[0] for g in gradients], axis=0)
        db1 = np.mean([g[1] for g in gradients], axis=0)
        dW2 = np.mean([g[2] for g in gradients], axis=0)
        db2 = np.mean([g[3] for g in gradients], axis=0)

        # Update weights
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = evaluate(X, y, W1, b1, W2, b2)
            print(f"Epoch {epoch+1:02d}: accuracy = {acc:.4f}")

    return W1, b1, W2, b2

# ------------------------------
# Main test
# ------------------------------
if __name__ == "__main__":
    X, y, y_one_hot = generate_dataset(num_samples=1000, input_dim=10, num_classes=3)

    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    train_y_one_hot, test_y_one_hot = y_one_hot[:800], y_one_hot[800:]

    print("Training parallel 2-layer network...")
    W1, b1, W2, b2 = train_parallel(train_X, train_y_one_hot, train_y,
                                    input_dim=10, hidden_dim=16, output_dim=3,
                                    learning_rate=0.5, epochs=20, num_devices=4)

    test_acc = evaluate(test_X, test_y, W1, b1, W2, b2)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
