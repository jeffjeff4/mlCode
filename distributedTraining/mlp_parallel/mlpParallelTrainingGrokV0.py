####runnable, use for interview
####grok
####runnable code, using queue
####
####python raw implementation, implement raw python code, simple MLP 的training loop，with data parallel, tensor parallel.
####please generate data X, Y, and code to use the above DP and tensor parallel code
####
####请：
####1. 把这个代码 改成支持 Adam 优化器、
####2. TP 尽量与真实的切分方式一致， 真实环境中是W1按列切，W2按行切
####第一层权重矩阵 W1 按列切分（column-wise partitioning）。
####第二层权重矩阵 W2 按行切分（row-wise partitioning）。
####3. 这里用 multiprocessing.Process + Queue 进行进程间通信（适合教学）。
####4. 可以不使用真实框架中会用更高效的通信（NCCL / AllGather / AllReduce）。
####   可以不使用真实部署在多 GPU 上会使用 GPU 通信库、共享内存、梯度压缩等技巧。
####6. 请模拟使用2块GPU
####7. 请把这个 可视化图解（用 ascii block）集成到之前写的教学代码里，这样跑的时候就能看到矩阵怎么切分、数据怎么流动。请包括forward propagation, backward propagation
####-------------------------------------------------------------------------------------------------------------
####python raw implementation, implement raw python code, simple MLP 的training loop，with data parallel, tensor parallel.
####please generate data X, Y, and code to use the above DP and tensor parallel code
####
####请：
####1. 把这个代码 改成支持 Adam 优化器、
####2. TP 尽量与真实的切分方式一致， 真实环境中是W1按列切，W2按行切
####第一层权重矩阵 W1 按列切分（column-wise partitioning）。
####第二层权重矩阵 W2 按行切分（row-wise partitioning）。
####3. 这里用 multiprocessing.Process + Queue 进行进程间通信（适合教学）。
####4. 可以不使用真实框架中会用更高效的通信（NCCL / AllGather / AllReduce）。
####   可以不使用真实部署在多 GPU 上会使用 GPU 通信库、共享内存、梯度压缩等技巧。
####5. 请不要使用queue，使用简单的方法来模拟gpu之间的数据交互
####6. 请模拟使用2块GPU
####7. 请把这个 可视化图解（用 ascii block）集成到之前写的教学代码里，这样跑的时候就能看到矩阵怎么切分、数据怎么流动。请包括forward propagation, backward propagation
####8. 请产生类似如下的流程图：
####=== Tensor Parallel Training with Adam ===
####
####TP Visualization - W1 Split (Column-Wise):
####
####  X (64 x 784)
####    |
####    v
####+---------+
####| W1      |
####| 784 x 256|
####+---------+
####Split column-wise:
####+---------+---------+
####| Device 0| Device 1|
####| 784 x 128| 784 x 128|
####+---------+---------+
####    |         |
####    v         v
#### A1_part0  A1_part1 (64 x 128)
#### (64 x 128)
#### (No All-Gather for A1; used locally for Z2_local)
####
####
####TP Visualization - W2 Split (Row-Wise):
####
#### A1_part (64 x 128)
####    |
####    v
####+---------+
####| W2      |
####| 256 x 10|
####+---------+
####Split row-wise:
####+---------+
####| Device 0|
####| 128 x 10|
####+---------+
####+---------+
####| Device 1|
####| 128 x 10|
####+---------+
####    |
####    v
#### Z2_local0 (64 x 10)
####    |
####    v
#### Z2_local1 (64 x 10)
####    |
####    +--------+
####      All-Reduce (sum)
####        |
####        v
#### Z2 (64 x 10)
####
####
####TP Visualization - Backward Flow:
####
#### dZ2 (64 x 10)
####    |
####    v
####dA1_local = dZ2 @ W2_part.T (64 x 128)
####    |
####    +--------+
####      All-Gather (concat)
####        |
####        v
#### dA1 (64 x 256)
####Split to dA1_slice (64 x 128)
####    |
####    v
#### dZ1_part = dA1_slice * relu_deriv (64 x 128)
####    |
####    v
#### dW1_part = X.T @ dZ1_part / m (784 x 128)
#### dW2_part = A1_part.T @ dZ2 / m (128 x 10)
####
####
####真实环境中不是W1按列切，W2按行切吗？这个code是按照W1按列切，W2按行切的方式来的吗？
####
####torch.chunk(W1, tp_size, dim=1)
####
####请把这个 可视化图解（用 ascii block）集成到之前写的教学代码里，这样跑的时候就能看到矩阵怎么切分、数据怎么流动？

####-------------------------------------------------------------------------------------
####真实环境中不是W1按列切，W2按行切吗？上面这个code是按照W1按列切，W2按行切的方式来的吗？
####
####
####对，你的理解是对的 ✅。 在 真实的 Tensor Parallel (TP) 实现里（比如 Megatron-LM），常见的切分方式是：
####W1（输入层 → 隐层）：
####W1 的 列切分（column parallelism）
####每个 rank 负责一部分 输出维度。
####优点：输入 X 可以广播到所有 rank，各自算 X @ W1_shard，得到的是对应 shard 的部分 hidden。最后再拼起来。
####W2（隐层 → 输出层）：
####W2 的 行切分（row parallelism）
####每个 rank 负责一部分 输入维度。
####优点：每个 rank 只存储部分 hidden 的参数，forward 时各自处理 shard，对输出做 AllReduce（因为最终输出要整合所有 shard 的贡献）。
####
####🔎 你提到的我的示例代码
####我之前给你的教学版代码，确实是这样切的：
####W1 是 torch.randn(D_in, D_hidden)，在构造时我用了 列切 → torch.chunk(W1, tp_size, dim=1)。 ✔ 这就是 “W1 按列切”。
####W2 是 torch.randn(D_hidden, D_out)，在构造时我用了 行切 → torch.chunk(W2, tp_size, dim=0)。 ✔ 这就是 “W2 按行切”。
####所以和你说的 真实环境常见做法是一致的。
####
####要不要我帮你在代码里加一段 图示（矩阵切分可视化），这样可以清楚看到：
####W1 如何按列切分
####W2 如何按行切分 以及 forward/backward 中各个 shard 的数据流？
####
####
####
####好 👌 我来帮你把 Tensor Parallel 的矩阵切分方式 用图示直观展示一下。
####
####🔹 1. W1 按列切分（Column Parallel）
####设输入层维度是 D_in，隐层维度是 D_hidden。 权重矩阵 W1 的形状是 (D_in, D_hidden)。
####正常情况：
####
####X (N, D_in)  @  W1 (D_in, D_hidden)  →  H (N, D_hidden)
####列切分：
####
####W1 被按列切分 → 每个 rank 拿一部分 hidden
####例子： 假设 D_in = 4, D_hidden = 6, 切成 2 份：
####
####         W1 (4x6)
####   ┌───────────────┐
####   │      Rank0    │      Rank1
####   │   (4x3 cols)  │   (4x3 cols)
####   └───────────────┘
####每个 rank 算：
####
####H0 = X @ W1_shard0   → (N, 3)
####H1 = X @ W1_shard1   → (N, 3)
####最后拼接：
####
####H = [H0 | H1]  → (N, 6)
####
####🔹 2. W2 按行切分（Row Parallel）
####设隐层维度是 D_hidden，输出层维度是 D_out。 权重矩阵 W2 的形状是 (D_hidden, D_out)。
####正常情况：
####
####H (N, D_hidden)  @  W2 (D_hidden, D_out)  →  Y (N, D_out)
####行切分：
####
####W2 被按行切分 → 每个 rank 拿一部分输入 hidden
####例子： 假设 D_hidden = 6, D_out = 2, 切成 2 份：
####
####         W2 (6x2)
####   ┌───────┐
####   │ Rank0 │  (3x2 rows)
####   │ Rank1 │  (3x2 rows)
####   └───────┘
####每个 rank 算：
####
####Y0 = H[:, :3] @ W2_shard0   → (N, 2)
####Y1 = H[:, 3:] @ W2_shard1   → (N, 2)
####最后 加和 (AllReduce)：
####
####Y = Y0 + Y1   → (N, 2)
####
####🔹 3. 总结数据流
####
####X ──► W1 (col split) ──► 拼接 H ──► W2 (row split) ──► AllReduce ──► Y
####W1：列切 → 输出拼接（Concat）
####W2：行切 → 输出求和（Reduce）
####
####要不要我直接把这个 可视化图解（用 matplotlib 或 ascii block）集成到我之前写的教学代码里，这样你跑的时候就能看到矩阵怎么切分、数据怎么流动？
####
####
####
####请加反向传播的 ascii block，显示 grad_W1 和 grad_W2 是怎么合并的
####
####
####
####好 ✅ 在前向我们已经画了 W1 列切 / W2 行切，现在我来帮你补上 反向传播的 ascii block，并且把它集成到代码里。这样一跑，不光能看到 forward 的数据流，还能看到 backward 的梯度流。
####
####🔹 反向传播直观逻辑
####Loss 对输出的梯度
####
####dL/dY  (N, D_out)
####W2（行切分）的反传
####forward 时 W2 是 行切 → 每个 rank 负责一部分 hidden → 反传时每个 rank 只收到对应 shard 的 grad_H_local。
####
####grad_W2_rank = H_localᵀ @ dL/dY
####grad_H_local = dL/dY @ W2_shardᵀ
####最后：
####
####grad_W2 = sum(grad_W2_rank)   (行拼起来)
####grad_H  = concat(grad_H_local)  (列拼起来)
####W1（列切分）的反传
####forward 时 W1 是 列切 → 每个 rank 只计算部分 H → backward 时要 sum 各个 shard 对输入的梯度贡献：
####
####grad_W1_rank = Xᵀ @ grad_H_local
####grad_X       = sum_over_ranks(grad_H_local @ W1_shardᵀ)
####最后：
####
####grad_W1 = sum(grad_W1_rank)


import numpy as np
from multiprocessing import Process, Queue
import time

# Hyperparameters
input_size = 784  # e.g., flattened MNIST images
hidden_size = 256
output_size = 10  # 10 classes
learning_rate = 0.001  # For Adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
batch_size = 64
epochs = 2  # Reduced for demo
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


# Adam optimizer parameters initialization for full params
def initialize_adam_params(W1, b1, W2, b2):
    return {
        'm_W1': np.zeros_like(W1),
        'v_W1': np.zeros_like(W1),
        'm_b1': np.zeros_like(b1),
        'v_b1': np.zeros_like(b1),
        'm_W2': np.zeros_like(W2),
        'v_W2': np.zeros_like(W2),
        'm_b2': np.zeros_like(b2),
        'v_b2': np.zeros_like(b2)
    }


# Adam update for full params
def update_with_adam(param, dparam, m, v, t, key):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return m, v


# Single-device MLP training with Adam
def train_single_device(X, Y_one_hot):
    print("\n=== Single Device Training with Adam ===")
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    adam_params = initialize_adam_params(W1, b1, W2, b2)
    t = 1

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
            m_batch = X_batch.shape[0]
            dZ2 = A2 - Y_batch
            dW2 = (A1.T @ dZ2) / m_batch
            db2 = np.sum(dZ2, axis=0) / m_batch
            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * relu_deriv(Z1)
            dW1 = (X_batch.T @ dZ1) / m_batch
            db1 = np.sum(dZ1, axis=0) / m_batch

            # Update with Adam
            adam_params['m_W1'], adam_params['v_W1'] = update_with_adam(W1, dW1, adam_params['m_W1'],
                                                                        adam_params['v_W1'], t, 'W1')
            adam_params['m_b1'], adam_params['v_b1'] = update_with_adam(b1, db1, adam_params['m_b1'],
                                                                        adam_params['v_b1'], t, 'b1')
            adam_params['m_W2'], adam_params['v_W2'] = update_with_adam(W2, dW2, adam_params['m_W2'],
                                                                        adam_params['v_W2'], t, 'W2')
            adam_params['m_b2'], adam_params['v_b2'] = update_with_adam(b2, db2, adam_params['m_b2'],
                                                                        adam_params['v_b2'], t, 'b2')
            t += 1

        avg_loss = running_loss / num_batches
        print(f"[Single] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2


# Data parallel training with Adam
def data_parallel_worker(X_batch, Y_batch, W1, b1, W2, b2, queue):
    # Forward pass
    Z1 = X_batch @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Compute loss
    m_batch = X_batch.shape[0]
    loss = cross_entropy_loss(A2, Y_batch)

    # Backward pass
    dZ2 = A2 - Y_batch
    dW2 = (A1.T @ dZ2) / m_batch
    db2 = np.sum(dZ2, axis=0) / m_batch
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (X_batch.T @ dZ1) / m_batch
    db1 = np.sum(dZ1, axis=0) / m_batch

    queue.put((dW1, db1, dW2, db2, loss))


def train_data_parallel(X, Y_one_hot, num_devices=2):
    print("\n=== Data Parallel Training with Adam ===")
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    adam_params = initialize_adam_params(W1, b1, W2, b2)
    t = 1

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        Y_shuffled = Y_one_hot[indices]
        running_loss = 0.0
        num_batches = X.shape[0] // batch_size

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # Split batch across devices
            batch_indices = np.array_split(range(X_batch.shape[0]), num_devices)
            processes = []
            queues = [Queue() for _ in range(num_devices)]
            for d in range(num_devices):
                p = Process(target=data_parallel_worker,
                            args=(X_batch[batch_indices[d]], Y_batch[batch_indices[d]], W1, b1, W2, b2, queues[d]))
                processes.append(p)
                p.start()

            # Collect gradients
            dW1_sum = np.zeros_like(W1)
            db1_sum = np.zeros_like(b1)
            dW2_sum = np.zeros_like(W2)
            db2_sum = np.zeros_like(b2)
            batch_loss = 0.0
            for q in queues:
                dW1, db1, dW2, db2, loss = q.get()
                dW1_sum += dW1
                db1_sum += db1
                dW2_sum += dW2
                db2_sum += db2
                batch_loss += loss
            dW1 = dW1_sum / num_devices
            db1 = db1_sum / num_devices
            dW2 = dW2_sum / num_devices
            db2 = db2_sum / num_devices
            running_loss += batch_loss / num_devices

            # Update with Adam
            adam_params['m_W1'], adam_params['v_W1'] = update_with_adam(W1, dW1, adam_params['m_W1'],
                                                                        adam_params['v_W1'], t, 'W1')
            adam_params['m_b1'], adam_params['v_b1'] = update_with_adam(b1, db1, adam_params['m_b1'],
                                                                        adam_params['v_b1'], t, 'b1')
            adam_params['m_W2'], adam_params['v_W2'] = update_with_adam(W2, dW2, adam_params['m_W2'],
                                                                        adam_params['v_W2'], t, 'W2')
            adam_params['m_b2'], adam_params['v_b2'] = update_with_adam(b2, db2, adam_params['m_b2'],
                                                                        adam_params['v_b2'], t, 'b2')
            t += 1

            for p in processes:
                p.join()

        avg_loss = running_loss / num_batches
        print(f"[Data Parallel] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return W1, b1, W2, b2


# Tensor parallel worker
def tensor_parallel_worker(device_id, num_devices, hidden_split, X_batch, Y_batch, W1_part, b1_part, W2_part, b2,
                           queue_in, queue_out):
    m_batch = X_batch.shape[0]

    # Forward first layer: column-parallel
    Z1_part = X_batch @ W1_part + b1_part  # Shape: (batch_size, hidden_split), e.g., (64, 128)
    A1_part = relu(Z1_part)

    # Forward second layer: row-parallel
    Z2_local = A1_part @ W2_part + b2 / num_devices  # Shape: (batch_size, output_size), e.g., (64, 10)

    # Send Z2_local for all-reduce (sum)
    queue_out.put(Z2_local)

    # Receive full Z2
    Z2 = queue_in.get()

    # Compute A2, dZ2, partial loss
    A2 = softmax(Z2)
    dZ2 = A2 - Y_batch
    partial_loss = cross_entropy_loss(A2, Y_batch)

    # Backward second layer: row-parallel
    dA1_local = dZ2 @ W2_part.T  # Shape: (batch_size, hidden_split), e.g., (64, 128)
    dW2_part = (A1_part.T @ dZ2) / m_batch  # Shape: (hidden_split, output_size), e.g., (128, 10)
    # db2 computed in main

    # Send dA1_local for all-gather (concat)
    queue_out.put(dA1_local)

    # Receive full dA1
    dA1 = queue_in.get()

    # Backward first layer: column-parallel
    dA1_slice = dA1[:, device_id * hidden_split: (device_id + 1) * hidden_split]  # Shape: (batch_size, hidden_split)
    dZ1_part = dA1_slice * relu_deriv(Z1_part)
    dW1_part = (X_batch.T @ dZ1_part) / m_batch  # Shape: (input_size, hidden_split), e.g., (784, 128)
    db1_part = np.sum(dZ1_part, axis=0) / m_batch  # Shape: (hidden_split,), e.g., (128,)

    # Send grads and partial loss
    queue_out.put((dW1_part, db1_part, dW2_part, partial_loss))


def train_tensor_parallel(X, Y_one_hot, num_devices=2):
    print("\n=== Tensor Parallel Training with Adam ===")
    # Print visualizations
    print("\nTP Visualization - W1 Split (Column-Wise):")
    print("""
  X (64 x 784)
    |
    v
+---------+
| W1      |
| 784 x 256|
+---------+
Split column-wise:
+---------+---------+
| Device 0| Device 1|
| 784 x 128| 784 x 128|
+---------+---------+
    |         |
    v         v
 A1_part0  A1_part1 (64 x 128)
 (64 x 128)
 (No All-Gather for A1; used locally for Z2_local)
""")

    print("\nTP Visualization - W2 Split (Row-Wise):")
    print("""
 A1_part (64 x 128)
    |
    v
+---------+
| W2      |
| 256 x 10|
+---------+
Split row-wise:
+---------+
| Device 0|
| 128 x 10|
+---------+
+---------+
| Device 1|
| 128 x 10|
+---------+
    |       
    v       
 Z2_local0 (64 x 10)
    |
    v
 Z2_local1 (64 x 10)
    |
    +--------+
      All-Reduce (sum)
        |
        v
 Z2 (64 x 10)
""")

    print("\nTP Visualization - Backward Flow:")
    print("""
 dZ2 (64 x 10)
    |
    v
dA1_local = dZ2 @ W2_part.T (64 x 128)
    |
    +--------+
      All-Gather (concat)
        |
        v
 dA1 (64 x 256)
Split to dA1_slice (64 x 128)
    |
    v
 dZ1_part = dA1_slice * relu_deriv (64 x 128)
    |
    v
 dW1_part = X.T @ dZ1_part / m (784 x 128)
 dW2_part = A1_part.T @ dZ2 / m (128 x 10)
""")

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    adam_params = initialize_adam_params(W1, b1, W2, b2)
    t = 1

    hidden_split = hidden_size // num_devices  # e.g., 256 / 2 = 128

    # W1: column-wise split, W2: row-wise split
    W1_parts = [W1[:, i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]  # Shape: (784, 128)
    b1_parts = [b1[i * hidden_split:(i + 1) * hidden_split] for i in range(num_devices)]  # Shape: (128,)
    W2_parts = [W2[i * hidden_split:(i + 1) * hidden_split, :] for i in range(num_devices)]  # Shape: (128, 10)
    b2 = b2  # Replicated across devices

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
                d, num_devices, hidden_split, X_batch, Y_batch, W1_parts[d], b1_parts[d], W2_parts[d], b2, queue_ins[d],
                queue_outs[d]))
                processes.append(p)
                p.start()

            # Collect Z2_local, sum to Z2 (all-reduce)
            Z2_locals = [queue_outs[d].get() for d in range(num_devices)]
            Z2 = np.sum(Z2_locals, axis=0)  # Shape: (batch_size, output_size), e.g., (64, 10)

            # Send Z2 to all devices
            for q in queue_ins:
                q.put(Z2)

            # Collect dA1_local, concatenate to dA1 (all-gather)
            dA1_locals = [queue_outs[d].get() for d in range(num_devices)]
            dA1 = np.concatenate(dA1_locals, axis=1)  # Shape: (batch_size, hidden_size), e.g., (64, 256)

            # Send dA1 to all devices
            for q in queue_ins:
                q.put(dA1)

            # Collect gradients and partial loss
            dW1_parts_new = [None] * num_devices
            db1_parts_new = [None] * num_devices
            dW2_parts_new = [None] * num_devices
            partial_losses = [None] * num_devices
            for d in range(num_devices):
                dW1_part, db1_part, dW2_part, partial_loss = queue_outs[d].get()
                dW1_parts_new[d] = dW1_part
                db1_parts_new[d] = db1_part
                dW2_parts_new[d] = dW2_part
                partial_losses[d] = partial_loss

            # Aggregate gradients
            dW1 = np.concatenate(dW1_parts_new, axis=1)  # Shape: (input_size, hidden_size), e.g., (784, 256)
            db1 = np.concatenate(db1_parts_new)  # Shape: (hidden_size,), e.g., (256,)
            dW2 = np.concatenate(dW2_parts_new, axis=0)  # Shape: (hidden_size, output_size), e.g., (256, 10)

            # Compute db2 in main
            A2 = softmax(Z2)
            dZ2 = A2 - Y_batch
            db2 = np.sum(dZ2, axis=0) / batch_size  # Shape: (output_size,), e.g., (10,)

            running_loss += sum(partial_losses) / num_devices

            # Update with Adam
            adam_params['m_W1'], adam_params['v_W1'] = update_with_adam(W1, dW1, adam_params['m_W1'],
                                                                        adam_params['v_W1'], t, 'W1')
            adam_params['m_b1'], adam_params['v_b1'] = update_with_adam(b1, db1, adam_params['m_b1'],
                                                                        adam_params['v_b1'], t, 'b1')
            adam_params['m_W2'], adam_params['v_W2'] = update_with_adam(W2, dW2, adam_params['m_W2'],
                                                                        adam_params['v_W2'], t, 'W2')
            adam_params['m_b2'], adam_params['v_b2'] = update_with_adam(b2, db2, adam_params['m_b2'],
                                                                        adam_params['v_b2'], t, 'b2')
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

    # Run training
    print("Training with Single Device")
    #W1_s, b1_s, W2_s, b2_s = train_single_device(X, Y_one_hot)

    print("\nTraining with Data Parallel")
    #W1_dp, b1_dp, W2_dp, b2_dp = train_data_parallel(X, Y_one_hot)

    print("\nTraining with Tensor Parallel")
    W1_tp, b1_tp, W2_tp, b2_tp = train_tensor_parallel(X, Y_one_hot)

    # Evaluate accuracies
    #print(f"\nSingle Device Accuracy: {evaluate(X, Y, W1_s, b1_s, W2_s, b2_s):.4f}")
    #print(f"Data Parallel Accuracy: {evaluate(X, Y, W1_dp, b1_dp, W2_dp, b2_dp):.4f}")
    print(f"Tensor Parallel Accuracy: {evaluate(X, Y, W1_tp, b1_tp, W2_tp, b2_tp):.4f}")