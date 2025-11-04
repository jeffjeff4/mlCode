import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-10), axis=1))

def cross_entropy_grad(pred, target):
    return pred - target

def adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * np.square(grad)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

def initialize_adam_params(W1, b1, W2, b2):
    return {
        'm_W1': np.zeros_like(W1), 'v_W1': np.zeros_like(W1),
        'm_b1': np.zeros_like(b1), 'v_b1': np.zeros_like(b1),
        'm_W2': np.zeros_like(W2), 'v_W2': np.zeros_like(W2),
        'm_b2': np.zeros_like(b2), 'v_b2': np.zeros_like(b2)
    }

def print_ascii_matrix(rank, title, matrix, max_cols=10, max_rows=5):
    print(f"Device {rank}: {title}")
    rows, cols = matrix.shape
    if rows > max_rows:
        print(f"(Showing first {max_rows} rows of {rows})")
        matrix = matrix[:max_rows]
        cols = min(cols, max_cols)
    if cols > max_cols:
        print(f"(Showing first {max_cols} cols of {cols})")
        matrix = matrix[:, :max_cols]
        cols = max_cols
    print("+" + "-" * (cols * 6 + cols - 1) + "+")
    for row in matrix:
        s = "| " + " | ".join(f"{x: .2f}" for x in row) + " |"
        print(s)
    print("+" + "-" * (cols * 6 + cols - 1) + "+")

def print_flow(rank, message):
    print(f"Device {rank}: {message}")

def print_tp_visualization(batch_size, input_dim, hidden_dim, output_dim, num_devices):
    local_hidden = hidden_dim // num_devices
    print("=== Tensor Parallel Training with Adam ===")
    print("\nTP Visualization - W1 Split (Column-Wise):\n")
    print(f"  X ({batch_size} x {input_dim})")
    print("    |")
    print("    v")
    print("+-----------------+")
    print(f"| W1              |")
    print(f"| {input_dim} x {hidden_dim}   |")
    print("+-----------------+")
    print("Split column-wise:")
    print("+---------+---------+")
    print("| Device 0| Device 1|")
    print(f"| {input_dim} x {local_hidden}| {input_dim} x {local_hidden}|")
    print("+---------+---------+")
    print("    |         |")
    print("    v         v")
    print(f" h_part0   h_part1 ({batch_size} x {local_hidden})")
    print(f" ({batch_size} x {local_hidden})")
    print(" (No All-Gather for h; used locally for logit_local)")

    print("\nTP Visualization - W2 Split (Row-Wise):\n")
    print(f" activated_part ({batch_size} x {local_hidden})")
    print("    |")
    print("    v")
    print("+-----------------+")
    print(f"| W2              |")
    print(f"| {hidden_dim} x {output_dim}   |")
    print("+-----------------+")
    print("Split row-wise:")
    print("+---------+")
    print("| Device 0|")
    print(f"| {local_hidden} x {output_dim}|")
    print("+---------+")
    print("+---------+")
    print("| Device 1|")
    print(f"| {local_hidden} x {output_dim}|")
    print("+---------+")
    print("    |       ")
    print("    v       ")
    print(f" logit_local0 ({batch_size} x {output_dim})")
    print("    |")
    print("    v")
    print(f" logit_local1 ({batch_size} x {output_dim})")
    print("    |")
    print("    +--------+")
    print("      All-Reduce (sum)")
    print("        |")
    print("        v")
    print(f" logit ({batch_size} x {output_dim})")

    print("\nTP Visualization - Backward Flow:\n")
    print(f" d_logit ({batch_size} x {output_dim})")
    print("    |")
    print("    v")
    print(f"d_activated_local = d_logit @ W2_part.T ({batch_size} x {local_hidden})")
    print("    |")
    print("    v")
    print(f" d_h_part = d_activated_local * relu_deriv(h_local) ({batch_size} x {local_hidden})")
    print("    |")
    print("    v")
    print(f" grad_W1_part = X.T @ d_h_part / {batch_size} ({input_dim} x {local_hidden})")
    print(f" grad_W2_part = activated_part.T @ d_logit / {batch_size} ({local_hidden} x {output_dim})")

def tensor_parallel_worker(device_id, num_devices, hidden_split, X_batch, Y_batch, W1_part, b1_part, W2_part, b2, queue_in, queue_out):
    # Adam states
    m_W1 = np.zeros_like(W1_part)
    v_W1 = np.zeros_like(W1_part)
    m_b1 = np.zeros_like(b1_part)
    v_b1 = np.zeros_like(b1_part)
    m_W2 = np.zeros_like(W2_part)
    v_W2 = np.zeros_like(W2_part)
    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)

    # Forward pass
    print_flow(device_id, f"Forward: h_local = X_batch @ W1_part + b1_part | Shapes: {X_batch.shape} @ {W1_part.shape} -> {X_batch.shape[0], hidden_split}")
    h_local = X_batch @ W1_part + b1_part
    print_ascii_matrix(device_id, "h_local (sharded hidden)", h_local)

    print_flow(device_id, "Forward: activated = relu(h_local)")
    activated = relu(h_local)
    print_ascii_matrix(device_id, "activated (sharded)", activated)

    print_flow(device_id, f"Forward: logit_local = activated @ W2_part + b2 | Shapes: {activated.shape} @ {W2_part.shape} -> {activated.shape[0], W2_part.shape[1]}")
    logit_local = activated @ W2_part + b2
    print_ascii_matrix(device_id, "logit_local (partial sum)", logit_local)

    # Send logit_local for all-reduce sum
    queue_out.put(logit_local)

    # Receive full logit
    logit = queue_in.get()
    print_flow(device_id, "Forward: Received full logit after all-reduce")
    print_ascii_matrix(device_id, "full logit", logit)

    # Loss (for logging)
    pred = softmax(logit)
    loss = cross_entropy_loss(pred, Y_batch)
    print_flow(device_id, f"Loss: {loss}")

    # Backward pass
    d_logit = cross_entropy_grad(pred, Y_batch)
    print_flow(device_id, "Backward: d_logit (full)")
    print_ascii_matrix(device_id, "d_logit", d_logit)

    # Grad for b2 (replicated)
    grad_b2 = np.sum(d_logit, axis=0) / X_batch.shape[0]

    # Grad for W2
    print_flow(device_id, "Backward: grad_W2 = activated.T @ d_logit")
    grad_W2 = activated.T @ d_logit / X_batch.shape[0]

    # d_activated (sharded directly, no all-gather needed)
    print_flow(device_id, "Backward: d_activated = d_logit @ W2_part.T")
    d_activated = d_logit @ W2_part.T
    print_ascii_matrix(device_id, "d_activated (sharded)", d_activated)

    # Activation deriv
    d_h_local = d_activated * relu_deriv(h_local)

    # Grad for b1
    grad_b1 = np.sum(d_h_local, axis=0) / X_batch.shape[0]

    # Grad for W1
    print_flow(device_id, "Backward: grad_W1 = X_batch.T @ d_h_local")
    grad_W1 = X_batch.T @ d_h_local / X_batch.shape[0]

    # Send gradients and loss
    queue_out.put((grad_W1, grad_b1, grad_W2, grad_b2, loss))

def train_tensor_parallel(X, Y_one_hot, num_devices=2):
    input_size = X.shape[1]
    hidden_size = 4  # Divisible by num_devices
    output_size = Y_one_hot.shape[1]
    batch_size = 2
    epochs = 3
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Print visualization
    print_tp_visualization(batch_size, input_size, hidden_size, output_size, num_devices)

    # Initialize weights
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    adam_params = initialize_adam_params(W1, b1, W2, b2)
    t = 1

    hidden_split = hidden_size // num_devices

    # Split weights
    W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
    b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
    W2_parts = [W2[i * hidden_split:(i + 1) * hidden_split, :] for i in range(num_devices)]
    b2 = b2  # Replicated

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]
        running_loss = 0.0
        num_batches = X.shape[0] // batch_size

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Start processes
            processes = []
            queue_ins = [Queue() for _ in range(num_devices)]
            queue_outs = [Queue() for _ in range(num_devices)]
            for d in range(num_devices):
                p = Process(target=tensor_parallel_worker, args=(
                    d, num_devices, hidden_split, X_batch, Y_batch, W1_parts[d], b1_parts[d], W2_parts[d], b2,
                    queue_ins[d], queue_outs[d]))
                processes.append(p)
                p.start()

            # Collect logit_local, sum to logit (all-reduce)
            logit_locals = [queue_outs[d].get() for d in range(num_devices)]
            logit = np.sum(logit_locals, axis=0)

            # Send full logit to all devices
            for q in queue_ins:
                q.put(logit)

            # Collect gradients and partial loss
            dW1_parts_new = [None] * num_devices
            db1_parts_new = [None] * num_devices
            dW2_parts_new = [None] * num_devices
            db2_parts_new = [None] * num_devices
            partial_losses = [None] * num_devices
            for d in range(num_devices):
                grad_W1, grad_b1, grad_W2, grad_b2, partial_loss = queue_outs[d].get()
                dW1_parts_new[d] = grad_W1
                db1_parts_new[d] = grad_b1
                dW2_parts_new[d] = grad_W2
                db2_parts_new[d] = grad_b2
                partial_losses[d] = partial_loss

            # Aggregate gradients
            dW1 = np.concatenate(dW1_parts_new, axis=1)  # Shape: (input_size, hidden_size), e.g., (4, 4)
            db1 = np.concatenate(db1_parts_new)  # Shape: (hidden_size,), e.g., (4,)
            dW2 = np.concatenate(dW2_parts_new, axis=0)  # Shape: (hidden_size, output_size), e.g., (4, 2)
            db2 = np.sum(db2_parts_new, axis=0)  # Shape: (output_size,), e.g., (2,)

            running_loss += sum(partial_losses) / num_devices

            # Update with Adam
            W1, adam_params['m_W1'], adam_params['v_W1'] = adam_update(
                W1, dW1, adam_params['m_W1'], adam_params['v_W1'], lr, beta1, beta2, eps, t)
            b1, adam_params['m_b1'], adam_params['v_b1'] = adam_update(
                b1, db1, adam_params['m_b1'], adam_params['v_b1'], lr, beta1, beta2, eps, t)
            W2, adam_params['m_W2'], adam_params['v_W2'] = adam_update(
                W2, dW2, adam_params['m_W2'], adam_params['v_W2'], lr, beta1, beta2, eps, t)
            b2, adam_params['m_b2'], adam_params['v_b2'] = adam_update(
                b2, db2, adam_params['m_b2'], adam_params['v_b2'], lr, beta1, beta2, eps, t)
            t += 1

            # Update weight parts
            W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
            b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]
            W2_parts = [W2[i * hidden_split:(i + 1) * hidden_split, :] for i in range(num_devices)]

            for p in processes:
                p.join()

        avg_loss = running_loss / num_batches
        print(f"[Tensor Parallel] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    N = 8  # Total samples
    input_size = 4
    output_size = 2
    X = np.random.randn(N, input_size)
    Y = np.random.randint(0, output_size, N)
    Y_one_hot = np.zeros((N, output_size))
    Y_one_hot[np.arange(N), Y] = 1

    # Run training
    W1, b1, W2, b2 = train_tensor_parallel(X, Y_one_hot, num_devices=2)