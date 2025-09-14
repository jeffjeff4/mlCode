####grok

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
def tensor_parallel_forward_backward(X_batch, Y_batch, W1_parts, b1_parts, W2_parts, b2_parts, A1, device_id,
                                     hidden_split, output_split):
    # Forward pass (first layer)
    W1_part = W1_parts[device_id]
    b1_part = b1_parts[device_id]
    Z1_part = X_batch @ W1_part + b1_part  # Shape: (batch_size, hidden_split), e.g., (64, 128)
    A1_part = relu(Z1_part)

    # Forward pass (second layer)
    W2_part = W2_parts[device_id]  # Shape: (hidden_size, output_split), e.g., (256, 5)
    b2_part = b2_parts[device_id]  # Shape: (output_split,), e.g., (5,)
    Z2_part = A1 @ W2_part + b2_part  # Shape: (batch_size, output_split), e.g., (64, 5)

    # Backward pass
    m = X_batch.shape[0]
    # Note: A2 and dZ2 are computed in main loop to avoid redundant computation
    dW2_part = np.zeros_like(W2_part)  # Placeholder
    db2_part = np.zeros_like(b2_part)  # Placeholder
    dA1 = np.zeros((X_batch.shape[0], hidden_size))  # Placeholder
    loss = 0

    # Compute gradients for first layer
    dA1_part = dA1[:, hidden_split * device_id: hidden_split * (device_id + 1)]
    dZ1_part = dA1_part * relu_deriv(Z1_part)
    dW1_part = (X_batch.T @ dZ1_part) / m
    db1_part = np.sum(dZ1_part, axis=0) / m

    return A1_part, Z2_part, dW1_part, db1_part, dW2_part, db2_part, loss


def train_tensor_parallel(X, Y_one_hot, W1, b1, W2, b2):
    hidden_split = hidden_size // num_devices  # e.g., 256 / 2 = 128
    output_split = output_size // num_devices  # e.g., 10 / 2 = 5
    W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
    b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
    W2_parts = [W2[:, i * output_split:(i + 1) * output_split] for i in range(num_devices)]  # Shape: (256, 5)
    b2_parts = [b2[i * output_split:(i + 1) * output_split] for i in range(num_devices)]

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]
        running_loss = 0.0
        num_batches = X.shape[0] // batch_size

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # First pass: compute A1_parts
            with Pool(num_devices) as pool:
                args = [(X_batch, Y_batch, W1_parts, b1_parts, W2_parts, b2_parts,
                         np.zeros((X_batch.shape[0], hidden_size)), device_id, hidden_split, output_split)
                        for device_id in range(num_devices)]
                results = pool.starmap(tensor_parallel_forward_backward, args)

            # Collect A1_parts and form A1
            A1_parts = [r[0] for r in results]
            A1 = np.concatenate(A1_parts, axis=1)  # Shape: (batch_size, hidden_size), e.g., (64, 256)

            # Second pass: compute Z2_parts and gradients with full A1
            with Pool(num_devices) as pool:
                args = [(X_batch, Y_batch, W1_parts, b1_parts, W2_parts, b2_parts, A1, device_id, hidden_split,
                         output_split)
                        for device_id in range(num_devices)]
                results = pool.starmap(tensor_parallel_forward_backward, args)

            # Collect Z2_parts and gradients
            Z2_parts = [r[1] for r in results]
            Z2 = np.concatenate(Z2_parts, axis=1)  # Shape: (batch_size, output_size), e.g., (64, 10)
            A2 = softmax(Z2)

            # Backward pass
            m = X_batch.shape[0]
            dZ2 = A2 - Y_batch
            dW2 = (A1.T @ dZ2) / m
            db2 = np.sum(dZ2, axis=0) / m
            dA1 = dZ2 @ W2.T
            loss = cross_entropy_loss(A2, Y_batch)

            # Aggregate gradients
            dW1 = np.concatenate([r[2] for r in results], axis=1)
            db1 = np.concatenate([r[3] for r in results])
            # Use centrally computed dW2, db2
            running_loss += loss

            # Update weights
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

            # Update weight parts
            W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
            b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
            W2_parts = [W2[:, i * output_split:(i + 1) * output_split] for i in range(num_devices)]
            b2_parts = [b2[i * output_split:(i + 1) * output_split] for i in range(num_devices)]

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



####-------------------------------------------------------------------------

####"However, in this simplified tensor parallel implementation, only device 0 computes the full A1, Z2, A2, and dZ2. Non-zero devices don’t have access to the full A1 or Z2 at this stage, so they cannot compute dZ2 or dA1."
####what about the real tensor parallel implementation?
####
####In the provided code, the tensor parallel implementation is simplified, as it assumes only device 0 computes the full A1, Z2, A2, and dZ2, while non-zero devices (e.g., device 1) return placeholder zeros for dA1, dW2, db2, and loss. This simplification avoids complex inter-device communication in raw Python, which is challenging without frameworks like PyTorch or DeepSpeed. You’ve asked about how a real tensor parallel implementation would differ, particularly regarding how non-zero devices would handle these computations. Below, I’ll explain the key differences between the simplified implementation and a real tensor parallel implementation, focusing on how all devices, including non-zero devices, would compute and share A1, Z2, A2, and dZ2. I’ll also provide a conceptual outline of how this could be implemented, while keeping the explanation grounded in the context of the MLP and the provided code.
####Key Differences in a Real Tensor Parallel Implementation
####In a real tensor parallel implementation, the goal is to distribute the computation and memory of the MLP across multiple devices (e.g., GPUs) more evenly, ensuring all devices contribute to both layers’ computations, not just the first layer. This requires splitting both W1 (first layer weights) and W2 (second layer weights) across devices and implementing efficient communication (e.g., via all-gather or all-reduce operations) to combine intermediate results. Here’s how the handling of A1, Z2, A2, and dZ2 differs:
####
####Splitting Both Layers’ Weights:
####
####Simplified Implementation: Only W1 (shape (784, 256)) and b1 (shape (256,)) are split across devices (e.g., two (784, 128) and (128,) for num_devices=2). W2 (shape (256, 10)) and b2 (shape (10,)) are kept whole, and only device 0 computes the second layer’s forward and backward passes (Z2, A2, dZ2, dW2, db2).
####Real Implementation: Both W1 and W2 are split. For example:
####
####W1 is split column-wise: each device gets a (784, 128) matrix.
####W2 is split row-wise: each device gets a (128, 10) matrix (since W2’s rows correspond to hidden_size=256).
####b1 and b2 are also split: b1 into two (128,), and b2 into two (5,) (assuming output_size=10 is split evenly).
####
####
####Impact: All devices compute partial outputs for both layers, reducing memory per device and distributing the computation more evenly.
####
####
####Computing A1:
####
####Simplified Implementation: Each device computes its A1_part (e.g., (batch_size, 128)), and the main loop collects all A1_parts to form the full A1 (e.g., (batch_size, 256)). Non-zero devices don’t use A1 locally and rely on the main loop to provide it.
####Real Implementation: Each device computes its A1_part (e.g., (batch_size, 128)), and an all-gather operation is performed to share A1_parts across all devices. After the all-gather, every device has the full A1 (e.g., (batch_size, 256)). This allows all devices to proceed with the second layer’s computations locally.
####Communication: The all-gather operation (e.g., via NCCL in a GPU framework or MPI in raw Python) ensures each device receives all A1_parts, concatenating them into A1.
####
####
####Computing Z2 and A2:
####
####Simplified Implementation: Only device 0 attempts to compute Z2 = A1 @ W2 + b2 and A2 = softmax(Z2), using a placeholder A1 (which is incorrect until the main loop provides the full A1). Non-zero devices skip this step and return loss=0.
####Real Implementation: Each device computes a partial Z2 using its portion of W2. For example:
####
####Device 0 has W2_0 (shape (128, 10)), computes Z2_0 = A1 @ W2_0 + b2_0, where b2_0 is its portion of the bias (e.g., (5,)).
####Device 1 has W2_1, computes Z2_1 = A1 @ W2_1 + b2_1.
####The partial outputs (Z2_0, Z2_1) are concatenated (via all-gather) to form the full Z2 (e.g., (batch_size, 10)), and then A2 = softmax(Z2) is computed on all devices.
####
####
####Communication: An all-gather is needed to combine Z2_0 and Z2_1 into Z2. Alternatively, each device could compute A2 locally after receiving the full Z2, or the softmax could be applied after aggregation.
####
####
####Computing dZ2 and dA1:
####
####Simplified Implementation: Only device 0 computes dZ2 = A2 - Y_batch and dA1 = dZ2 @ W2.T, while non-zero devices return dA1 = np.zeros(...), dW2 = np.zeros_like(W2), and db2 = np.zeros_like(b2). The main loop recomputes these correctly using the full A1.
####Real Implementation: All devices compute dZ2 and a portion of dA1:
####
####Each device has A2 (after the all-gather for Z2) and computes dZ2 = A2 - Y_batch (shape (batch_size, 10)).
####For dA1, each device computes a partial gradient using its portion of W2.T. For example:
####
####Device 0: dA1_0 = dZ2 @ W2_0.T (where W2_0.T is (10, 128)), yielding dA1_0 (shape (batch_size, 128)).
####Device 1: dA1_1 = dZ2 @ W2_1.T, yielding dA1_1 (shape (batch_size, 128)).
####
####
####The full dA1 (shape (batch_size, 256)) is implicitly formed by using dA1_0 and dA1_1 for each device’s portion of the hidden layer.
####
####
####Communication: No additional communication is needed for dA1, as each device computes its own portion directly. However, dZ2 requires the full A2, which depends on the prior all-gather for Z2.
####
####
####Gradient Computation (dW2, db2):
####
####Simplified Implementation: Non-zero devices return dW2 = np.zeros_like(W2) and db2 = np.zeros_like(b2), and the main loop computes the correct dW2 and db2 using the full A1 and dZ2.
####Real Implementation: Each device computes gradients for its portion of W2 and b2:
####
####dW2_0 = (A1_0.T @ dZ2) / m (where A1_0 is device 0’s portion of A1, shape (batch_size, 128)), yielding dW2_0 (shape (128, 10)).
####db2_0 = np.sum(dZ2, axis=0)[:5] / m (for device 0’s portion of b2).
####Similarly for device 1.
####An all-reduce operation aggregates dW2_0 and dW2_1 into the full dW2 (if needed for weight updates).
####
####
####Communication: An all-reduce is typically used to sum gradients across devices for dW2 and db2, ensuring consistent weight updates.
####
####
####Loss Computation:
####
####Simplified Implementation: Non-zero devices return loss = 0, and the main loop computes the correct loss using the full A2.
####Real Implementation: All devices compute the loss locally after receiving the full A2 (post all-gather for Z2). The loss is identical across devices, so no aggregation is needed, or a single device reports it.
####
####
####
####Why the Simplified Implementation Uses Placeholders
####In the provided code, non-zero devices return zeros for dA1, dW2, db2, and loss because:
####
####The implementation avoids complex inter-device communication in raw Python, which lacks efficient primitives like NCCL or MPI.
####Device 0’s computations for Z2, A2, dZ2, etc., are placeholders (using an incorrect A1), and the main loop recomputes them correctly after collecting A1_parts.
####Non-zero devices focus on computing first-layer gradients (dW1_part, db1_part), which are valid and aggregated correctly.
####
####In a real implementation, non-zero devices would actively participate in all computations, requiring communication to share A1_parts, partial Z2, and gradients.
####Real Tensor Parallel Implementation Outline
####Here’s how tensor_parallel_forward_backward and train_tensor_parallel would look in a real tensor parallel setup, assuming we have a communication mechanism (e.g., MPI or a simplified multiprocessing.Pipe for raw Python). I’ll provide a conceptual outline without implementing full communication, as raw Python makes this complex, but I’ll show the key changes.

####Key Changes in Real Implementation
####
####W2 Splitting: W2 is split row-wise, and each device computes a partial Z2_part.
####All-Gather for A1 and Z2: Devices share A1_part and Z2_part to form full A1 and Z2, enabling all devices to compute A2, dZ2, dA1, dW2_part, and db2_part.
####No Placeholders: Non-zero devices compute valid gradients and loss, eliminating the need for zero placeholders.
####Communication Overhead: Requires two all-gather operations (for A1 and Z2) and an all-reduce for dW2 and db2.
####
####Challenges in Raw Python
####Implementing this in raw Python is complex due to:
####
####Communication: multiprocessing.Pool doesn’t support efficient inter-process communication. You’d need multiprocessing.Pipe, sockets, or MPI, which adds significant complexity.
####Performance: Copying large arrays (e.g., A1_parts) between processes is slow compared to GPU frameworks using NCCL.
####Synchronization: Ensuring all devices have consistent data requires careful management.
####
####In practice, frameworks like PyTorch with DeepSpeed or Megatron-LM handle these operations efficiently using optimized communication primitives.
####Why the Simplified Version Uses Zeros
####The simplified version uses zeros for non-zero devices because:
####
####It avoids implementing complex communication in raw Python.
####It centralizes second-layer computations in the main loop, reducing the need for non-zero devices to compute A2, dZ2, etc.
####It’s sufficient for the small MLP and synthetic dataset, where the overhead of proper tensor parallelism might outweigh benefits.
####
####Conclusion
####In a real tensor parallel implementation, all devices compute partial outputs and gradients for both layers, using all-gather to share A1 and Z2 and all-reduce for gradients. This distributes memory and computation more evenly but requires efficient communication, which is impractical in raw Python. The simplified version’s else block with zeros is a workaround to maintain consistency while offloading second-layer computations to the main loop.
####If you want a full implementation with simulated communication (e.g., using multiprocessing.Pipe) or further details on integrating this with a framework like PyTorch, let me know!