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
import multiprocessing as mp
from multiprocessing import Process, Barrier, Lock, Array

def create_shared_array(shape):
    size = int(np.prod(shape))
    arr = Array('d', size)
    return np.frombuffer(arr.get_obj(), dtype=np.float64).reshape(shape)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

def mse_loss(pred, target):
    return np.mean((pred - target)**2)

def mse_grad(pred, target):
    return 2 * (pred - target) / pred.shape[0]

def adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * np.square(grad)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v

def print_ascii_matrix(rank, title, matrix, max_cols=10, max_rows=5):
    # Limit print size for large matrices, but since small dims, ok
    print(f"Rank {rank}: {title}")
    rows, cols = matrix.shape
    if rows > max_rows:
        print(f"(Showing first {max_rows} rows of {rows})")
        matrix = matrix[:max_rows]
        rows = max_rows
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
    print(f"Rank {rank}: {message}")

def print_tp_visualization(batch_size, input_dim, hidden_dim, output_dim, TP_size):
    local_hidden = hidden_dim // TP_size
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
    print(f"d_activated_local = d_logit @ W2_part.T ({batch_size} x {local_hidden})  # sharded directly")
    print("    |")
    print("    v")
    print(f" d_h_part = d_activated_local * relu_deriv(h_local) ({batch_size} x {local_hidden})")
    print("    |")
    print("    v")
    print(f" grad_W1_part = X.T @ d_h_part / {batch_size} ({input_dim} x {local_hidden})")
    print(f" grad_W2_part = activated_part.T @ d_logit / {batch_size} ({local_hidden} x {output_dim})")

def worker(rank, TP_size, DP_size, world_size, input_dim, hidden_dim, output_dim, N, micro_batch_size, epochs, lr, beta1, beta2, eps,
           shared_X, shared_Y, shared_W1_shards, shared_W2_shards, shared_b1_shards, shared_b2,
           shared_logit_buffers, shared_grad_W1_buffers, shared_grad_W2_buffers, shared_grad_b1_buffers, shared_grad_b2_buffers,
           barrier, lock):
    local_hidden = hidden_dim // TP_size
    tp_rank = rank % TP_size
    dp_rank = rank // TP_size
    local_N = N // DP_size
    start_idx = dp_rank * local_N
    end_idx = start_idx + local_N

    # Copy local data
    X_local = np.copy(shared_X[start_idx:end_idx])
    Y_local = np.copy(shared_Y[start_idx:end_idx])

    # Copy local model shards
    W1_local = np.copy(shared_W1_shards[tp_rank])
    W2_local = np.copy(shared_W2_shards[tp_rank])
    b1_local = np.copy(shared_b1_shards[tp_rank])
    b2_local = np.copy(shared_b2)

    # Visualize initial shards
    print_ascii_matrix(rank, f"Initial W1 shard (column-wise split)", W1_local)
    print_ascii_matrix(rank, f"Initial W2 shard (row-wise split)", W2_local)
    print_ascii_matrix(rank, f"Initial b1 shard", b1_local.reshape(1, -1))
    print_ascii_matrix(rank, f"Initial b2 (replicated)", b2_local.reshape(1, -1))

    # Adam states
    m_W1 = np.zeros_like(W1_local)
    v_W1 = np.zeros_like(W1_local)
    m_W2 = np.zeros_like(W2_local)
    v_W2 = np.zeros_like(W2_local)
    m_b1 = np.zeros_like(b1_local)
    v_b1 = np.zeros_like(b1_local)
    m_b2 = np.zeros_like(b2_local)
    v_b2 = np.zeros_like(b2_local)

    num_batches = local_N // micro_batch_size

    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            batch_start = batch_idx * micro_batch_size
            batch_end = batch_start + micro_batch_size
            X_batch = X_local[batch_start:batch_end]
            Y_batch = Y_local[batch_start:batch_end]

            # Forward pass
            print_flow(rank, f"Forward: local_h = X_batch @ W1_local + b1_local | Shapes: {X_batch.shape} @ {W1_local.shape} -> {X_batch.shape[0], local_hidden}")
            local_h = X_batch @ W1_local + b1_local
            print_ascii_matrix(rank, "local_h (sharded hidden)", local_h)

            print_flow(rank, "Forward: activated = relu(local_h)")
            activated = relu(local_h)
            print_ascii_matrix(rank, "activated (sharded)", activated)

            print_flow(rank, f"Forward: local_logit = activated @ W2_local + b2_local | Shapes: {activated.shape} @ {W2_local.shape} -> {activated.shape[0], output_dim}")
            local_logit = activated @ W2_local + b2_local
            print_ascii_matrix(rank, "local_logit (partial sum)", local_logit)

            # All-reduce sum for logit across TP (simulate with shared memory)
            shared_logit = shared_logit_buffers[dp_rank]
            with lock:
                if tp_rank == 0:
                    shared_logit.fill(0.0)
            barrier.wait()
            with lock:
                shared_logit += local_logit
            barrier.wait()
            logit = np.copy(shared_logit)
            print_flow(rank, "Forward: After all-reduce sum across TP -> full logit")
            print_ascii_matrix(rank, "full logit", logit)

            # Loss (for logging)
            loss = mse_loss(logit, Y_batch)
            print_flow(rank, f"Loss: {loss}")

            # Backward pass
            d_logit = mse_grad(logit, Y_batch)
            print_flow(rank, "Backward: d_logit (full)")
            print_ascii_matrix(rank, "d_logit", d_logit)

            # Grad for b2 (replicated)
            grad_b2 = np.sum(d_logit, axis=0)

            # Grad for W2
            print_flow(rank, "Backward: grad_W2 = activated.T @ d_logit")
            grad_W2 = activated.T @ d_logit

            # d_activated (sharded)
            print_flow(rank, "Backward: d_activated = d_logit @ W2_local.T")
            d_activated = d_logit @ W2_local.T
            print_ascii_matrix(rank, "d_activated (sharded)", d_activated)

            # Activation deriv
            d_local_h = d_activated * relu_deriv(local_h)

            # Grad for b1 (sharded)
            grad_b1 = np.sum(d_local_h, axis=0)

            # Grad for W1
            print_flow(rank, "Backward: grad_W1 = X_batch.T @ d_local_h")
            grad_W1 = X_batch.T @ d_local_h

            # All-reduce grads across DP (sum then avg)
            # For W1 grad
            shared_grad_W1 = shared_grad_W1_buffers[tp_rank]
            with lock:
                if dp_rank == 0:
                    shared_grad_W1.fill(0.0)
            barrier.wait()
            with lock:
                shared_grad_W1 += grad_W1
            barrier.wait()
            avg_grad_W1 = shared_grad_W1 / DP_size

            # For W2
            shared_grad_W2 = shared_grad_W2_buffers[tp_rank]
            with lock:
                if dp_rank == 0:
                    shared_grad_W2.fill(0.0)
            barrier.wait()
            with lock:
                shared_grad_W2 += grad_W2
            barrier.wait()
            avg_grad_W2 = shared_grad_W2 / DP_size

            # For b1
            shared_grad_b1 = shared_grad_b1_buffers[tp_rank]
            with lock:
                if dp_rank == 0:
                    shared_grad_b1.fill(0.0)
            barrier.wait()
            with lock:
                shared_grad_b1 += grad_b1
            barrier.wait()
            avg_grad_b1 = shared_grad_b1 / DP_size

            # For b2 (replicated, avg across DP, add only once per DP)
            shared_grad_b2 = shared_grad_b2_buffers[0]
            with lock:
                if dp_rank == 0 and tp_rank == 0:
                    shared_grad_b2.fill(0.0)
            barrier.wait()
            if tp_rank == 0:
                with lock:
                    shared_grad_b2 += grad_b2
            barrier.wait()
            avg_grad_b2 = shared_grad_b2 / DP_size

            # Adam update
            t = epoch * num_batches + batch_idx + 1
            W1_local, m_W1, v_W1 = adam_update(W1_local, avg_grad_W1, m_W1, v_W1, lr, beta1, beta2, eps, t)
            W2_local, m_W2, v_W2 = adam_update(W2_local, avg_grad_W2, m_W2, v_W2, lr, beta1, beta2, eps, t)
            b1_local, m_b1, v_b1 = adam_update(b1_local, avg_grad_b1, m_b1, v_b1, lr, beta1, beta2, eps, t)
            b2_local, m_b2, v_b2 = adam_update(b2_local, avg_grad_b2, m_b2, v_b2, lr, beta1, beta2, eps, t)

            # Visualize updated shards (less frequent)
            if batch_idx % 2 == 0:
                print_ascii_matrix(rank, f"Updated W1 shard after batch {batch_idx}", W1_local)
                print_ascii_matrix(rank, f"Updated W2 shard after batch {batch_idx}", W2_local)

    print(f"Rank {rank}: Training complete.")

if __name__ == "__main__":
    TP_size = 2
    DP_size = 1  # Set to 1 for simulating 2 GPUs with TP=2
    world_size = TP_size * DP_size
    input_dim = 4
    hidden_dim = 16  # Divisible by TP_size
    output_dim = 2
    local_hidden = hidden_dim // TP_size
    N = 4  # Total samples
    micro_batch_size = 2
    epochs = 3
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Print visualization flowchart
    print_tp_visualization(micro_batch_size, input_dim, hidden_dim, output_dim, TP_size)

    # Generate data
    np.random.seed(42)
    shared_X = create_shared_array((N, input_dim))
    shared_Y = create_shared_array((N, output_dim))
    shared_X[:] = np.random.randn(N, input_dim)
    shared_Y[:] = np.random.randn(N, output_dim)

    # Initialize model shards
    shared_W1_shards = [create_shared_array((input_dim, local_hidden)) for _ in range(TP_size)]
    shared_W2_shards = [create_shared_array((local_hidden, output_dim)) for _ in range(TP_size)]
    shared_b1_shards = [create_shared_array((local_hidden,)) for _ in range(TP_size)]
    shared_b2 = create_shared_array((output_dim,))

    for i in range(TP_size):
        shared_W1_shards[i][:] = np.random.randn(input_dim, local_hidden) * 0.1
        shared_W2_shards[i][:] = np.random.randn(local_hidden, output_dim) * 0.1
        shared_b1_shards[i][:] = np.random.randn(local_hidden) * 0.1
    shared_b2[:] = np.random.randn(output_dim) * 0.1

    # Shared buffers
    shared_logit_buffers = [create_shared_array((micro_batch_size, output_dim)) for _ in range(DP_size)]
    shared_grad_W1_buffers = [create_shared_array((input_dim, local_hidden)) for _ in range(TP_size)]
    shared_grad_W2_buffers = [create_shared_array((local_hidden, output_dim)) for _ in range(TP_size)]
    shared_grad_b1_buffers = [create_shared_array((local_hidden,)) for _ in range(TP_size)]
    shared_grad_b2_buffers = [create_shared_array((output_dim,)) for _ in range(TP_size)]  # Use [0]

    barrier = Barrier(world_size)
    lock = Lock()

    processes = []
    for rank in range(world_size):
        p = Process(target=worker, args=(rank, TP_size, DP_size, world_size, input_dim, hidden_dim, output_dim, N, micro_batch_size, epochs, lr, beta1, beta2, eps,
                                         shared_X, shared_Y, shared_W1_shards, shared_W2_shards, shared_b1_shards, shared_b2,
                                         shared_logit_buffers, shared_grad_W1_buffers, shared_grad_W2_buffers, shared_grad_b1_buffers, shared_grad_b2_buffers,
                                         barrier, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("shared_W1_shards = ", shared_W1_shards)
    print("shared_grad_W1_buffers = ", shared_grad_W1_buffers)
    print("shared_W2_shards = ", shared_W2_shards)
    print("shared_grad_W2_buffers = ", shared_grad_W2_buffers)
    print("shared_b1_shards = ", shared_b1_shards)
    print("shared_grad_b1_buffers = ", shared_grad_b1_buffers)
    print("shared_b2 = ", shared_b2)
    print("shared_grad_b2_buffers = ", shared_grad_b2_buffers)



