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
####   可以不使用真实部署在多 GPU 上会使用 GPU 通信库、共享内存、梯度压缩等技巧。
####5. 请不要使用queue，使用简单的方法来模拟gpu之间的数据交互
####6. 请模拟使用2块GPU
####
####7. in this code, in data parallel simulation, please do not use rank0 gpu calculated sum_grads 
####in simulation, please code similar as in real world is data parallel implemention 
####
####
####8. in this code, in tensor parallel simulation, please do not use rank0 gpu calculated 1) loss function, 2) dZ_full.
####in simulation, please code similar as in real world is tensor parallel implemention 
####
####9. 请把这个 可视化图解（用 ascii block）集成到之前写的教学代码里，这样跑的时候就能看到矩阵怎么切分、数据怎么流动。请包括forward propagation, backward propagation
####
####10. 请产生类似如下的流程图：
####=== Tensor Parallel Training with Adam ===
####
####TP Visualization - W1 Split (Column-Wise):
####
####  X (64 x 784)
####    |
####    v
####+---------+
####| W1      |
####| 784 x 256|
####+---------+
####Split column-wise:
####+---------+---------+
####| Device 0| Device 1|
####| 784 x 128| 784 x 128|
####+---------+---------+
####    |         |
####    v         v
#### A1_part0  A1_part1 (64 x 128)
#### (64 x 128)
#### (No All-Gather for A1; used locally for Z2_local)
####
####
####TP Visualization - W2 Split (Row-Wise):
####
#### A1_part (64 x 128)
####    |
####    v
####+---------+
####| W2      |
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
####    |
####    v
#### Z2_local0 (64 x 10)
####    |
####    v
#### Z2_local1 (64 x 10)
####    |
####    +--------+
####      All-Reduce (sum)
####        |
####        v
#### Z2 (64 x 10)
####
####
####TP Visualization - Backward Flow:
####
#### dZ2 (64 x 10)
####    |
####    v
####dA1_local = dZ2 @ W2_part.T (64 x 128)
####    |
####    +--------+
####      All-Gather (concat)
####        |
####        v
#### dA1 (64 x 256)
####Split to dA1_slice (64 x 128)
####    |
####    v
#### dZ1_part = dA1_slice * relu_deriv (64 x 128)
####    |
####    v
#### dW1_part = X.T @ dZ1_part / m (784 x 128)
#### dW2_part = A1_part.T @ dZ2 / m (128 x 10)
####
####
####真实环境中不是W1按列切，W2按行切吗？这个code是按照W1按列切，W2按行切的方式来的吗？
####
####torch.chunk(W1, tp_size, dim=1)
####
####请把这个 可视化图解（用 ascii block）集成到之前写的教学代码里，这样跑的时候就能看到矩阵怎么切分、数据怎么流动？
####
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
####X (N, D_in)  @  W1 (D_in, D_hidden)  →  H (N, D_hidden)
####列切分：
####
####W1 被按列切分 → 每个 rank 拿一部分 hidden
####例子： 假设 D_in = 4, D_hidden = 6, 切成 2 份：
####
####         W1 (4x6)
####   ┌───────────────┐
####   │      Rank0    │      Rank1
####   │   (4x3 cols)  │   (4x3 cols)
####   └───────────────┘
####每个 rank 算：
####
####H0 = X @ W1_shard0   → (N, 3)
####H1 = X @ W1_shard1   → (N, 3)
####最后拼接：
####
####H = [H0 | H1]  → (N, 6)
####
####🔹 2. W2 按行切分（Row Parallel）
####设隐层维度是 D_hidden，输出层维度是 D_out。 权重矩阵 W2 的形状是 (D_hidden, D_out)。
####正常情况：
####
####H (N, D_hidden)  @  W2 (D_hidden, D_out)  →  Y (N, D_out)
####行切分：
####
####W2 被按行切分 → 每个 rank 拿一部分输入 hidden
####例子： 假设 D_hidden = 6, D_out = 2, 切成 2 份：
####
####         W2 (6x2)
####   ┌───────┐
####   │ Rank0 │  (3x2 rows)
####   │ Rank1 │  (3x2 rows)
####   └───────┘
####每个 rank 算：
####
####Y0 = H[:, :3] @ W2_shard0   → (N, 2)
####Y1 = H[:, 3:] @ W2_shard1   → (N, 2)
####最后 加和 (AllReduce)：
####
####Y = Y0 + Y1   → (N, 2)
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
####dL/dY  (N, D_out)
####W2（行切分）的反传
####forward 时 W2 是 行切 → 每个 rank 负责一部分 hidden → 反传时每个 rank 只收到对应 shard 的 grad_H_local。
####
####grad_W2_rank = H_localᵀ @ dL/dY
####grad_H_local = dL/dY @ W2_shardᵀ
####最后：
####
####grad_W2 = sum(grad_W2_rank)   (行拼起来)
####grad_H  = concat(grad_H_local)  (列拼起来)
####W1（列切分）的反传
####forward 时 W1 是 列切 → 每个 rank 只计算部分 H → backward 时要 sum 各个 shard 对输入的梯度贡献：
####
####grad_W1_rank = Xᵀ @ grad_H_local
####grad_X       = sum_over_ranks(grad_H_local @ W1_shardᵀ)
####最后：
####
####grad_W1 = sum(grad_W1_rank)




import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import time
from typing import Dict, List, Tuple

NUM_EPOCHS = 1

# ----------------------------------------------------------------------------
# 模拟 GPU 间通信的简化工具 (已修复)
# ----------------------------------------------------------------------------
def communicate(
        rank: int,
        barrier: multiprocessing.Barrier,
        shared_data: Dict[str, torch.Tensor],
        key_prefix: str,
        data_to_send: torch.Tensor
) -> List[torch.Tensor]:
    """
    一个修复后、更健壮的通信函数，用于模拟同步操作。
    它使用一个 Barrier 来确保所有进程在读写操作之间正确同步，从而解决死锁问题。
    """
    world_size = barrier.parties
    my_key = f"{key_prefix}_{rank}"
    shared_data[my_key] = data_to_send.clone().detach()
    barrier.wait()
    all_data = [shared_data[f"{key_prefix}_{r}"] for r in range(world_size)]
    barrier.wait()
    if rank == 0:
        for r in range(world_size):
            del shared_data[f"{key_prefix}_{r}"]
    barrier.wait()
    return all_data


# ----------------------------------------------------------------------------
# 数据生成与模型定义
# ----------------------------------------------------------------------------
def generate_data(num_samples: int = 64, input_dim: int = 784, output_dim: int = 10) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """生成用于分类的随机数据。"""
    print("正在生成模拟数据...")
    X = torch.randn(num_samples, input_dim)
    labels = torch.randint(0, output_dim, (num_samples,))
    Y = nn.functional.one_hot(labels, num_classes=output_dim).float()
    print(f"数据生成完毕。X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y


class SimpleMLP(nn.Module):
    """一个用于数据并行的简单 MLP 模型。"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ----------------------------------------------------------------------------
# 数据并行 (Data Parallel) 训练实现
# ----------------------------------------------------------------------------
def data_parallel_worker(rank: int, world_size: int, X: torch.Tensor, Y: torch.Tensor, barrier: multiprocessing.Barrier,
                         shared_data: Dict):
    """数据并行工作进程。"""
    batch_size_per_gpu = X.shape[0] // world_size
    start, end = rank * batch_size_per_gpu, (rank + 1) * batch_size_per_gpu
    X_part, Y_part = X[start:end], Y[start:end]

    torch.manual_seed(42)
    model = SimpleMLP(X.shape[1], 256, Y.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_part)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs, Y_part)
        loss.backward()

        for i, param in enumerate(model.parameters()):
            key = f"epoch_{epoch}_dp_grad_{i}"
            all_grads = communicate(rank, barrier, shared_data, key, param.grad)
            param.grad = torch.sum(torch.stack(all_grads), dim=0) / world_size

        optimizer.step()

        if rank == 0:
            print(f"DP | Epoch {epoch + 1}/3 | Rank {rank} | Loss: {loss.item():.4f}")


# ----------------------------------------------------------------------------
# 张量并行 (Tensor Parallel) 训练实现 (已修复)
# ----------------------------------------------------------------------------
def tensor_parallel_worker(rank: int, world_size: int, X: torch.Tensor, Y: torch.Tensor,
                           barrier: multiprocessing.Barrier, shared_data: Dict):
    """张量并行工作进程 (已修复)。"""
    D_in, D_hidden, D_out = X.shape[1], 256, Y.shape[1]
    D_hidden_part = D_hidden // world_size

    torch.manual_seed(42)
    W1_part = torch.chunk(torch.randn(D_in, D_hidden), world_size, dim=1)[rank].clone().requires_grad_()
    W2_part = torch.chunk(torch.randn(D_hidden, D_out), world_size, dim=0)[rank].clone().requires_grad_()
    b1_part = torch.chunk(torch.randn(D_hidden), world_size, dim=0)[rank].clone().requires_grad_()
    b2 = torch.randn(D_out, requires_grad=True)

    optimizer = optim.Adam([W1_part, W2_part, b1_part, b2], lr=0.001)

    for epoch in range(NUM_EPOCHS):
        # ========== 前向传播 (Forward Pass) ==========
        A1_part = X @ W1_part + b1_part
        Z1_part = torch.relu(A1_part)  # Shape: (batch, D_hidden_part)

        all_Z1_parts = communicate(rank, barrier, shared_data, f"epoch_{epoch}_tp_Z1_gather", Z1_part)
        Z1_full = torch.cat(all_Z1_parts, dim=1)  # Shape: (batch, D_hidden)

        Z1_slice_for_W2 = torch.chunk(Z1_full, world_size, dim=1)[rank]
        Z2_local = Z1_slice_for_W2 @ W2_part

        all_Z2_locals = communicate(rank, barrier, shared_data, f"epoch_{epoch}_tp_Z2_reduce", Z2_local)
        Z2_full = torch.sum(torch.stack(all_Z2_locals), dim=0) + b2

        # ========== 损失和初始梯度 (在每个GPU上独立计算) ==========
        A2 = torch.softmax(Z2_full, dim=1)
        loss = -torch.sum(Y * torch.log(A2 + 1e-9)) / X.shape[0]
        dZ2_full = (A2 - Y) / X.shape[0]

        if rank == 0:
            print(f"TP | Epoch {epoch + 1}/3 | Rank {rank} | Loss: {loss.item():.4f}")

        # ========== 反向传播 (Backward Pass) - 已修复 ==========
        # 1. g'(Z1) 的梯度 [行并行部分]
        # 关键修复：使用 Z1_slice_for_W2 (即 Z1_part) 来计算 dW2_part
        dW2_part = Z1_slice_for_W2.T @ dZ2_full  # Shape: (D_hidden_part, D_out)

        # dZ1 = dZ2 @ W2.T -> 每个rank计算部分梯度 dZ1_local
        dZ1_local = dZ2_full @ W2_part.T  # Shape: (batch, D_hidden_part)

        # 2. [通信]: All-Gather dZ1_local -> dZ1_full (关键修复)
        all_dZ1_locals = communicate(rank, barrier, shared_data, f"epoch_{epoch}_tp_dZ1_gather", dZ1_local)
        dZ1_full = torch.cat(all_dZ1_locals, dim=1)  # Shape: (batch, D_hidden)

        # 3. f'(X) 的梯度 [列并行部分]
        dZ1_slice = torch.chunk(dZ1_full, world_size, dim=1)[rank]  # Shape: (batch, D_hidden_part)
        dA1_part = dZ1_slice * (A1_part > 0).float()

        dW1_part = X.T @ dA1_part
        db1_part = dA1_part.sum(dim=0)
        db2 = dZ2_full.sum(dim=0)

        W1_part.grad, W2_part.grad, b1_part.grad, b2.grad = dW1_part, dW2_part, db1_part, db2
        optimizer.step()


# ----------------------------------------------------------------------------
# 可视化图解函数
# ----------------------------------------------------------------------------
def print_dp_visualization(batch_size, world_size):
    print("\n" + "=" * 70)
    print("=== 开始数据并行训练 (Data Parallel Training with Adam) ===")
    print("=" * 70)
    print(f"""
DP Visualization - Forward & Backward Flow (All-Reduce):

+-----------------+ +-----------------+
|   GPU 0         | |   GPU 1         |
| Model Replica M | | Model Replica M |
+-----------------+ +-----------------+
        ^                   ^
        | (Sync Weights)    | (Sync Weights)
        +-------------------+
                  |
+------------------------------------------+
| Data Batch (Size: {batch_size})                  |
+------------------------------------------+
                  | (Split Batch)
      +-------------------+
      |                   |
      v                   v
+-----------------+ +-----------------+
| Data Shard 0    | | Data Shard 1    |
| (Size: {batch_size // world_size})     | | (Size: {batch_size // world_size})     |
+-----------------+ +-----------------+
      |                   |
      v (Fwd & Bwd)       v (Fwd & Bwd)
+-----------------+ +-----------------+
| Local Grads G0  | | Local Grads G1  |
+-----------------+ +-----------------+
      |                   |
      +---- All-Reduce (Sum & Average) ----+
      |                   |
      v                   v
+-----------------+ +-----------------+
| Averaged Grads  | | Averaged Grads  |
| G_avg = (G0+G1)/2 | | G_avg = (G0+G1)/2 |
+-----------------+ +-----------------+
      | (Update M)        | (Update M)
      v                   v
[End of Step, Models are in Sync]
""")


def print_tp_visualization(m, d_in, d_hidden, d_hidden_part, d_out, w1_part_shape, w2_part_shape):
    print("\n" + "=" * 70)
    print("=== 开始张量并行训练 (Tensor Parallel Training with Adam) ===")
    print("=" * 70)
    print(f"""
TP Visualization - W1 Split (Column-Wise):

  X ({m} x {d_in})
    |
    v
+-----------------+-----------------+
| GPU 0 (W1_part) | GPU 1 (W1_part) |
| {w1_part_shape[0]} x {w1_part_shape[1]}      | {w1_part_shape[0]} x {w1_part_shape[1]}      |
+-----------------+-----------------+
    | (X @ W1_p0)     | (X @ W1_p1)
    v                 v
 Z1_part0 ({m}x{d_hidden_part})   Z1_part1 ({m}x{d_hidden_part})
    |                 |
    +---- All-Gather (concatenate) ----+
              |
              v
         Z1_full (after ReLU) ({m} x {d_hidden})

TP Visualization - W2 Split (Row-Wise):

  Z1_full ({m} x {d_hidden}) -> [Split]-> Z1_p0 ({m}x{d_hidden_part}), Z1_p1 ({m}x{d_hidden_part})
    |                                     |
    v (Z1_p0 @ W2_p0)                     v (Z1_p1 @ W2_p1)
+-----------------+                     +-----------------+
| W2_part0        |                     | W2_part1        |
| {w2_part_shape[0]} x {w2_part_shape[1]}       |                     | {w2_part_shape[0]} x {w2_part_shape[1]}       |
+-----------------+                     +-----------------+
    |                                     |
    v                                     v
 Z2_local0 ({m} x {d_out})                  Z2_local1 ({m} x {d_out})
    |                                     |
    +----------- All-Reduce (sum) -----------+
                       |
                       v
                   Z2_full ({m} x {d_out})

TP Visualization - Backward Flow (Corrected):

  dZ2 ({m} x {d_out}) [All GPUs compute this independently]
    |
    +------------------------------------------+
    | (Z1_slice.T @ dZ2)                       | (dZ2 @ W2_part.T)
    v                                          v
 dW2_part ({w2_part_shape[0]}x{w2_part_shape[1]})            dZ1_local ({m}x{d_hidden_part})
 [Local Grad for W2]                         |
                                             +-- All-Gather (cat) --+
                                                       |
                                                       v
                                                  dZ1_full ({m}x{d_hidden}) -> [Split] -> dZ1_slice
                                                       |
                                                       v (X.T @ (dZ1_slice * relu'))
                                                  dW1_part ({w1_part_shape[0]}x{w1_part_shape[1]})
                                                  [Local Grad for W1]
""")


# ----------------------------------------------------------------------------
# 主执行函数 (已修复)
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    D_IN, D_OUT, D_HIDDEN = 784, 10, 256
    X_main, Y_main = generate_data(input_dim=D_IN, output_dim=D_OUT)

    manager = multiprocessing.Manager()

    # --- 数据并行模拟 ---
    world_size_dp = 2
    shared_data_dp = manager.dict()
    barrier_dp = multiprocessing.Barrier(world_size_dp)

    print_dp_visualization(X_main.shape[0], world_size_dp)
    processes_dp = []
    for rank in range(world_size_dp):
        p = multiprocessing.Process(
            target=data_parallel_worker,
            args=(rank, world_size_dp, X_main, Y_main, barrier_dp, shared_data_dp)
        )
        processes_dp.append(p)
        p.start()
    for p in processes_dp: p.join()

    # --- 张量并行模拟 ---
    world_size_tp = 2
    shared_data_tp = manager.dict()
    barrier_tp = multiprocessing.Barrier(world_size_tp)

    print_tp_visualization(X_main.shape[0], D_IN, D_HIDDEN, D_HIDDEN // world_size_tp, D_OUT,
                           (D_IN, D_HIDDEN // world_size_tp), (D_HIDDEN // world_size_tp, D_OUT))

    processes_tp = []
    for rank in range(world_size_tp):
        p = multiprocessing.Process(
            target=tensor_parallel_worker,
            args=(rank, world_size_tp, X_main, Y_main, barrier_tp, shared_data_tp)
        )
        processes_tp.append(p)
        p.start()
    for p in processes_tp: p.join()

    print("\n" + "=" * 70)
    print("所有训练模拟已完成。")
    print("=" * 70)

