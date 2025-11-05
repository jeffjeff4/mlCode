####sometimes runnable, sometimes stuck
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


import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import numpy as np
import copy
from typing import Tuple

# Global variable to simulate shared memory for communication.
# Use this with caution, as it is a simplified model.
shared_data_list = None


def generate_data(num_samples: int = 256,
                  input_dim: int = 128,
                  output_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data for a simple MLP.
    """
    print("Generating synthetic data...")
    X = torch.randn(num_samples, input_dim)
    # A simple linear relationship with some noise.
    Y = torch.randn(num_samples, output_dim) * 0.1 + torch.sum(X, dim=1, keepdim=True) * 0.5
    print(f"Data generated. X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron model for single-device and DP.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def single_device_train(X: torch.Tensor, Y: torch.Tensor, device: torch.device):
    """
    Implements a single-device training loop.
    """
    print("\n--- Start Single-Device Training ---")
    model = MLP(X.shape[1], 128, Y.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        print(f"Single-Device | Epoch {epoch + 1}/5 | Loss: {loss.item():.4f}")

    print("--- Single-Device Training Finished ---")


def data_parallel_train(X_data: torch.Tensor, Y_data: torch.Tensor, rank: int, world_size: int, shared_data_list):
    """
    Implements a Data Parallel training loop for a single process.
    """
    if rank == 0:
        print("""
            --------------------
            | Data Parallelism |
            --------------------

            +-----------+            +-----------+
            |  Data (X) |            |   Model   |
            |   [1..N]  |            |  (Copy)   |
            +-----------+            +-----+-----+
                  |                        |
                  | Split Data             | Replicate Model
                  V                        V
      +-----------+           +------------+------------+
      |  Data (X1) |  ->  | Model (Copy 1) | -> Gradients (G1)
      +-----------+           +------------+------------+
      |  Data (X2) |  ->  | Model (Copy 2) | -> Gradients (G2)
      +-----------+           +------------+------------+
      |  ...      |           |   ...      | -> ...
      +-----------+           +------------+------------+
                                     |
                                     | All-Reduce & Average Gradients
                                     V
                           +------------------+
                           |  Updated Model   |
                           +------------------+
        """)

    print(f"\n--- Process {rank}: Running Data Parallel Training ---")

    batch_size = X_data.shape[0] // world_size
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    X_part = X_data[start_idx:end_idx]
    Y_part = Y_data[start_idx:end_idx]

    print(f"Process {rank} handles data from index {start_idx} to {end_idx - 1}, shape {X_part.shape}")

    model = MLP(X_part.shape[1], 128, Y_part.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_part)
        loss = criterion(outputs, Y_part)
        loss.backward()

        # Simulate All-Reduce for Gradients
        shared_data_list.append({name: p.grad.detach() for name, p in model.named_parameters()})

        # Wait for all processes to finish putting gradients
        while len(shared_data_list) < world_size * (epoch + 1):
            pass

        # Process 0 acts as the master to average gradients
        if rank == 0:
            sum_grads = copy.deepcopy(shared_data_list[epoch * world_size])
            for i in range(1, world_size):
                other_grads = shared_data_list[epoch * world_size + i]
                for name, grad in other_grads.items():
                    sum_grads[name] += grad

            for name, param in model.named_parameters():
                param.grad = sum_grads[name] / world_size

            optimizer.step()
            print(f"Data Parallel | Epoch {epoch + 1}/5 | Total Loss: {loss.item():.4f}")
        else:
            # For this simple demo, other processes just wait.
            pass


def tensor_parallel_train(X_data: torch.Tensor, Y_data: torch.Tensor, rank: int, world_size: int, shared_data_list):
    """
    Implements a Tensor Parallel training loop for a single process using a Master-Worker pattern.
    """
    print(f"\n--- Process {rank}: Running Tensor Parallel Training ---")

    input_dim = X_data.shape[1]
    hidden_dim = 128
    output_dim = Y_data.shape[1]

    assert hidden_dim % world_size == 0, "Hidden dim must be divisible by world size."
    hidden_dim_part = hidden_dim // world_size

    # Manually create model weights for each partition.
    W1_part = torch.randn(input_dim, hidden_dim_part, requires_grad=True)
    b1_part = torch.zeros(hidden_dim_part, requires_grad=True)

    W2_part = torch.randn(hidden_dim_part, output_dim, requires_grad=True)
    b2 = torch.zeros(output_dim, requires_grad=True)

    optimizer = optim.Adam([W1_part, W2_part, b1_part, b2], lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):
        optimizer.zero_grad()

        # --- Forward Pass (local computation) ---
        Y_part = torch.matmul(X_data, W1_part) + b1_part
        Y_part = nn.functional.relu(Y_part)
        Z_part = torch.matmul(Y_part, W2_part) + b2

        # --- Master-Worker Communication ---
        # Master process (rank 0) collects all Z_parts.
        if rank == 0:
            # Master gets its own Z_part
            all_Z_parts = [Z_part]
            # Master waits for and collects Z_parts from all workers.
            while len(shared_data_list) < world_size - 1:
                pass
            all_Z_parts.extend(shared_data_list[:])
            shared_data_list[:] = []  # Clear the list for next epoch.

            # Master sums all Z_parts to get Z_full.
            Z_full = sum(all_Z_parts)
            loss = criterion(Z_full, Y_data)

            # Master computes the gradient of the loss and sends it to workers.
            # IMPORTANT: Detach the gradient tensor before sending!
            dZ_full = (2 * (Z_full - Y_data) / X_data.shape[0]).detach()
            for _ in range(world_size - 1):
                shared_data_list.append(dZ_full)

        else:  # Worker process
            # Worker sends its detached Z_part to the master.
            shared_data_list.append(Z_part.detach())

            # Worker waits for master to send back the dZ_full.
            while len(shared_data_list) == 0:
                pass
            dZ_full = shared_data_list.pop(0)

        # --- Manual Backward Pass (local computation) ---
        # Each process uses the shared dZ_full to compute local gradients.

        # 1. dW2_part and db2 (local computation)
        dW2_part = Y_part.T @ dZ_full
        db2 = torch.sum(dZ_full, dim=0)

        # 2. dY_part (local computation)
        dY_part = dZ_full @ W2_part.T

        # 3. dW1_part and db1_part (local computation)
        dW1_part = X_data.T @ dY_part
        db1_part = torch.sum(dY_part, dim=0)

        # --- Optimization Step ---
        # Apply the gradients manually and update weights.
        W1_part.grad = dW1_part
        b1_part.grad = db1_part
        W2_part.grad = dW2_part
        b2.grad = db2

        optimizer.step()

        if rank == 0:
            print(f"Tensor Parallel | Epoch {epoch + 1}/5 | Total Loss: {loss.item():.4f}")

        # --- Backward Pass Visualization ---
        if rank == 0:
            print("\nTP Visualization - Backward Pass:")
            print(f"""
            ========================================================
            |       Backward Pass with Gradients All-Reduce      |
            ========================================================
            Loss: {loss.item():.4f}
                    |
                    v
            dZ_full (Full Gradient): {tuple(dZ_full.shape)}
                    |
                    v
            dW2_part0 = Y_part0.T @ dZ_full
            dW2_part1 = Y_part1.T @ dZ_full

            +--------------------+--------------------+
            | dW2_part0: {tuple(W2_part.shape)} | dW2_part1: {tuple(W2_part.shape)} |
            +--------------------+--------------------+


            ========================================================
            |   All-Gather for dY (Hidden Layer Gradient)      |
            ========================================================

            dY_part0 = dZ_full @ W2_part0.T
            dY_part1 = dZ_full @ W2_part1.T

            +--------------------+--------------------+
            | dY_part0: {tuple(dY_part.shape)} | dY_part1: {tuple(dY_part.shape)} |
            +--------------------+--------------------+
                    |                     |
                    +---- All-Gather (Concat) ----+
                                |
                                v
                    dY_full: {tuple(dY_part.shape)}


            ========================================================
            |     dW1 Gradient Calculation (Local)           |
            ========================================================

            dW1_part0 = X.T @ dY_part0
            dW1_part1 = X.T @ dY_part1

            +--------------------+--------------------+
            | dW1_part0: {tuple(W1_part.shape)} | dW1_part1: {tuple(W1_part.shape)} |
            +--------------------+--------------------+
            """)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    manager = multiprocessing.Manager()
    shared_data_list = manager.list()

    INPUT_DIM = 128
    OUTPUT_DIM = 1
    X, Y = generate_data(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)

    device = torch.device('cpu')
    X, Y = X.to(device), Y.to(device)

    single_device_train(X, Y, device)

    world_size_dp = 2
    print(f"\n{'=' * 50}\nBeginning Data Parallel Training on {world_size_dp} processes.\n{'=' * 50}")
    processes_dp = []
    for rank in range(world_size_dp):
        p = multiprocessing.Process(target=data_parallel_train,
                                    args=(X, Y, rank, world_size_dp, shared_data_list))
        processes_dp.append(p)
        p.start()
    for p in processes_dp:
        p.join()

    shared_data_list[:] = []

    world_size_tp = 2
    print(f"\n{'=' * 50}\nBeginning Tensor Parallel Training on {world_size_tp} processes.\n{'=' * 50}")
    processes_tp = []
    for rank in range(world_size_tp):
        p = multiprocessing.Process(target=tensor_parallel_train,
                                    args=(X, Y, rank, world_size_tp, shared_data_list))
        processes_tp.append(p)
        p.start()
    for p in processes_tp:
        p.join()
