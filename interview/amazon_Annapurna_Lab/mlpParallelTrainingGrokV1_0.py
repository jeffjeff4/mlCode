####grok, simplified version

import numpy as np
from multiprocessing import Pool
import time

# Hyperparameters
input_size = 784  # e.g., flattened MNIST images
hidden_size = 256
output_size = 10  # 10 classes
learning_rate = 0.01
batch_size = 64
epochs = 2
num_devices = 2  # For parallel implementations


# Activation functions
def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    m = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)] + 1e-15)
    return np.mean(log_likelihood)


# Single-device MLP training (for reference)
def train_single_device(X, Y_one_hot, W1, b1, W2, b2):
    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]
        running_loss = 0.0
        num_batches = X.shape[0] // batch_size

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Forward pass
            Z1 = X_batch @ W1 + b1
            A1 = relu(Z1)
            Z2 = A1 @ W2 + b2
            A2 = softmax(Z2)

            # Compute loss
            loss = cross_entropy_loss(A2, Y_batch)
            running_loss += loss

            # Backward pass
            m = X_batch.shape[0]
            dZ2 = A2 - Y_batch
            dW2 = (A1.T @ dZ2) / m
            db2 = np.sum(dZ2, axis=0) / m
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * relu_deriv(Z1)
            dW1 = (X_batch.T @ dZ1) / m
            db1 = np.sum(dZ1, axis=0) / m

            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        avg_loss = running_loss / num_batches
        print(f"[Single] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2


# Data parallel training
def compute_gradients(args):
    X_batch, Y_batch, W1, b1, W2, b2 = args
    # Forward pass
    Z1 = X_batch @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Compute loss
    loss = cross_entropy_loss(A2, Y_batch)

    # Backward pass
    m = X_batch.shape[0]
    dZ2 = A2 - Y_batch
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (X_batch.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m

    return dW1, db1, dW2, db2, loss


def train_data_parallel(X, Y_one_hot, W1, b1, W2, b2):
    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]
        running_loss = 0.0
        num_batches = X.shape[0] // batch_size

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Split batch across "devices"
            batch_indices = np.array_split(range(X_batch.shape[0]), num_devices)
            batch_data = [(X_batch[indices], Y_batch[indices], W1, b1, W2, b2)
                          for indices in batch_indices]

            # Compute gradients in parallel
            with Pool(num_devices) as pool:
                gradients = pool.map(compute_gradients, batch_data)

            # Aggregate gradients
            dW1 = np.mean([g[0] for g in gradients], axis=0)
            db1 = np.mean([g[1] for g in gradients], axis=0)
            dW2 = np.mean([g[2] for g in gradients], axis=0)
            db2 = np.mean([g[3] for g in gradients], axis=0)
            running_loss += np.mean([g[4] for g in gradients])

            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        avg_loss = running_loss / num_batches
        print(f"[Data Parallel] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2


# Tensor parallel training
def tensor_parallel_forward_backward(X_batch, Y_batch, W1_parts, b1_parts, W2, b2, device_id, hidden_split):
    # Forward pass (split hidden layer)
    W1_part = W1_parts[device_id]
    b1_part = b1_parts[device_id]
    Z1_part = X_batch @ W1_part + b1_part
    A1_part = relu(Z1_part)

    # Return A1_part and other gradients
    m = X_batch.shape[0]
    if device_id == 0:
        # Placeholder for full A1 (will be set in main loop)
        A1 = np.zeros((X_batch.shape[0], hidden_size))
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Backward pass
        dZ2 = A2 - Y_batch
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        dA1 = dZ2 @ W2.T
        loss = cross_entropy_loss(A2, Y_batch)
    else:
        dA1 = np.zeros((X_batch.shape[0], hidden_size))
        dW2 = np.zeros_like(W2)
        db2 = np.zeros_like(b2)
        loss = 0

    # Split dA1 for each device
    dA1_part = dA1[:, hidden_split * device_id: hidden_split * (device_id + 1)]
    dZ1_part = dA1_part * relu_deriv(Z1_part)
    dW1_part = (X_batch.T @ dZ1_part) / m
    db1_part = np.sum(dZ1_part, axis=0) / m

    return A1_part, dW1_part, db1_part, dW2, db2, loss


def train_tensor_parallel(X, Y_one_hot, W1, b1, W2, b2):
    # Split weights across devices
    hidden_split = hidden_size // num_devices
    W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
    b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]
        running_loss = 0.0
        num_batches = X.shape[0] // batch_size

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Compute gradients in parallel
            with Pool(num_devices) as pool:
                args = [(X_batch, Y_batch, W1_parts, b1_parts, W2, b2, device_id, hidden_split)
                        for device_id in range(num_devices)]
                results = pool.starmap(tensor_parallel_forward_backward, args)

            # Collect A1_parts from all devices
            A1_parts = [r[0] for r in results]
            A1 = np.concatenate(A1_parts, axis=1)  # Shape: (batch_size, hidden_size)

            # Recompute forward/backward for device 0 to get correct Z2, A2, and gradients
            m = X_batch.shape[0]
            Z2 = A1 @ W2 + b2
            A2 = softmax(Z2)
            dZ2 = A2 - Y_batch
            dW2 = (A1.T @ dZ2) / m
            db2 = np.sum(dZ2, axis=0) / m
            dA1 = dZ2 @ W2.T
            loss = cross_entropy_loss(A2, Y_batch)

            # Aggregate gradients
            dW1 = np.concatenate([r[1] for r in results], axis=1)
            db1 = np.concatenate([r[2] for r in results])
            # Use recomputed dW2, db2 from full A1
            running_loss += loss

            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

            # Update weight parts for next iteration
            W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
            b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]

        avg_loss = running_loss / num_batches
        print(f"[Tensor Parallel] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2


# Evaluate accuracy
def evaluate(X, Y, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    predictions = np.argmax(A2, axis=1)
    return np.mean(predictions == Y)


if __name__ == '__main__':
    # Generate synthetic dataset
    np.random.seed(42)
    X = np.random.randn(1000, input_size)  # 1000 samples x 784 features
    Y = np.random.randint(0, output_size, 1000)  # Integer labels (0-9)
    Y_one_hot = np.zeros((1000, output_size))
    Y_one_hot[np.arange(1000), Y] = 1  # One-hot encoded labels

    # Initialize weights
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)

    # Run training
    print("Training with Single Device")
    #W1_s, b1_s, W2_s, b2_s = train_single_device(X, Y_one_hot, W1.copy(), b1.copy(), W2.copy(), b2.copy())

    print("\nTraining with Data Parallel")
    #W1_dp, b1_dp, W2_dp, b2_dp = train_data_parallel(X, Y_one_hot, W1.copy(), b1.copy(), W2.copy(), b2.copy())

    print("\nTraining with Tensor Parallel")
    W1_tp, b1_tp, W2_tp, b2_tp = train_tensor_parallel(X, Y_one_hot, W1.copy(), b1.copy(), W2.copy(), b2.copy())

    # Evaluate accuracies
    #print(f"\nSingle Device Accuracy: {evaluate(X, Y, W1_s, b1_s, W2_s, b2_s):.4f}")
    #print(f"Data Parallel Accuracy: {evaluate(X, Y, W1_dp, b1_dp, W2_dp, b2_dp):.4f}")
    print(f"Tensor Parallel Accuracy: {evaluate(X, Y, W1_tp, b1_tp, W2_tp, b2_tp):.4f}")