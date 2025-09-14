import numpy as np
from multiprocessing import Pool

def compute_gradients(batch_data):
    X_batch, y_batch = batch_data
    # Forward and backward pass (same as above)
    Z1 = X_batch @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    dZ2 = A2 - y_batch
    dW2 = (A1.T @ dZ2) / len(X_batch)
    db2 = np.sum(dZ2, axis=0) / len(X_batch)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (X_batch.T @ dZ1) / len(X_batch)
    db1 = np.sum(dZ1, axis=0) / len(X_batch)
    return dW1, db1, dW2, db2

# Split data across "devices"
num_devices = 4
batch_indices = np.array_split(range(X.shape[0]), num_devices)
batch_data = [(X[indices], y_one_hot[indices]) for indices in batch_indices]

# Compute gradients in parallel
with Pool(num_devices) as pool:
    gradients = pool.map(compute_gradients, batch_data)

# Aggregate gradients
dW1 = np.mean([g[0] for g in gradients], axis=0)
db1 = np.mean([g[1] for g in gradients], axis=0)
dW2 = np.mean([g[2] for g in gradients], axis=0)
db2 = np.mean([g[3] for g in gradients], axis=0)

# Update weights (same as single-device)
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2