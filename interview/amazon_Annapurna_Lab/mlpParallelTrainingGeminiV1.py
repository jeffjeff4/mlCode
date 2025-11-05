####gemini
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import queue
import time
import copy
from typing import Tuple, List, Dict


# -----------------
# 1. 資料生成
# -----------------
def generate_data(num_samples: int = 256,
                  input_dim: int = 128,
                  output_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data for a simple MLP.
    """
    print("Generating synthetic data...")
    X = torch.randn(num_samples, input_dim)
    Y = torch.randn(num_samples, output_dim) * 0.1 + torch.sum(X, dim=1, keepdim=True) * 0.5
    print(f"Data generated. X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y


# -----------------
# 2. 模型定義
# -----------------
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model for single-device and DP."""

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


def tensor_parallel_train(X_data: torch.Tensor, Y_data: torch.Tensor, rank: int, world_size: int,
                          queue_in: multiprocessing.Queue, queue_out: multiprocessing.Queue):
    """
    Implements a Tensor Parallel training loop for a single process.

    This implementation follows the standard TP method:
    - The first linear layer's weight (W1) is split column-wise.
    - The second linear layer's weight (W2) is split row-wise.
    """
    print(f"\n--- Process {rank}: Running Tensor Parallel Training ---")

    # ASCII Art for Tensor Parallelism
    if rank == 0:
        print("""
            --------------------
            | Tensor Parallelism |
            --------------------

            Model Part 1:              Model Part 2:
            W1 (col-split)             W1 (col-split)
            +----------+               +----------+
            |  W1_part |               |  W1_part |
            +----------+               +----------+
                  /                        /
              /                            /
          X(full)                       X(full)
          (batch_size, input_dim)        (batch_size, input_dim)
               \                        /
                \                      /
     +-----------+-----------+        +------------+-------------+
     |   Y1_part (local)     |  ...   |    Y2_part (local)      |
     +-----------------------+        +-------------------------+
                  |                           |
                  | All-Reduce & Sum          | All-Reduce & Sum
                  V                           V
            +----------------+
            | Full Output (Z) |
            +----------------+
        """)

    # Model parameters for each process.
    input_dim = X_data.shape[1]
    hidden_dim = 128
    output_dim = Y_data.shape[1]

    assert hidden_dim % world_size == 0, "Hidden dim must be divisible by world size."
    hidden_dim_part = hidden_dim // world_size

    # Manually create model weights.
    # W1 is column-wise partitioned.
    W1_full = torch.randn(input_dim, hidden_dim)
    W1_part = W1_full[:, rank * hidden_dim_part:(rank + 1) * hidden_dim_part].clone().detach().requires_grad_(True)
    b1_full = torch.zeros(hidden_dim)
    b1_part = b1_full[rank * hidden_dim_part:(rank + 1) * hidden_dim_part].clone().detach().requires_grad_(True)

    # W2 is row-wise partitioned.
    W2_full = torch.randn(hidden_dim, output_dim)
    W2_part = W2_full[rank * hidden_dim_part:(rank + 1) * hidden_dim_part, :].clone().detach().requires_grad_(True)
    b2 = torch.zeros(output_dim, requires_grad=True)

    # Use Adam optimizer on local parameters.
    optimizer = optim.Adam([W1_part, W2_part, b1_part, b2], lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):
        optimizer.zero_grad()

        # --- Forward Pass ---
        # Each process computes its local part of the hidden layer output.
        Y_part = torch.matmul(X_data, W1_part) + b1_part

        # Each process computes its local part of the final output.
        Z_part = torch.matmul(Y_part, W2_part) + b2

        # === 核心修正部分 ===
        # 在將 Z_part 放入隊列之前，將其與計算圖分離。
        queue_out.put(Z_part.detach())

        # --- Backward Pass (main process-centric) ---
        if rank == 0:
            # Main process sums all partial outputs.
            Z_full = Z_part.clone()
            for _ in range(world_size - 1):
                Z_full += queue_in.get()

            # Calculate total loss on the full output.
            loss = criterion(Z_full, Y_data)
            loss.backward()

            # Collect gradients from all processes.
            # === 核心修正部分 ===
            # 同樣，在傳輸梯度時也需要 detach。
            grad_W1_part = W1_part.grad.detach()
            grad_W2_part = W2_part.grad.detach()
            grad_b1_part = b1_part.grad.detach()
            grad_b2 = b2.grad.detach()

            # Simulate gradient All-Reduce.
            queue_out.put({'grad_W1_part': grad_W1_part, 'grad_W2_part': grad_W2_part, 'grad_b1_part': grad_b1_part,
                           'grad_b2': grad_b2})

            # Apply gradients.
            optimizer.step()

            print(f"Tensor Parallel | Epoch {epoch + 1}/5 | Total Loss: {loss.item():.4f}")
        else:
            # Other processes wait to receive gradients from the main process
            # to complete the backward pass and update their local parameters.
            received_grads = queue_in.get()

            W1_part.grad = received_grads['grad_W1_part']
            W2_part.grad = received_grads['grad_W2_part']
            b1_part.grad = received_grads['grad_b1_part']
            b2.grad = received_grads['grad_b2']

            optimizer.step()


def data_parallel_train(X_data: torch.Tensor, Y_data: torch.Tensor, rank: int, world_size: int,
                        queue_in: multiprocessing.Queue, queue_out: multiprocessing.Queue):
    """
    Implements a Data Parallel training loop for a single process.
    """
    print(f"\n--- Process {rank}: Running Data Parallel Training ---")

    # ASCII Art for Data Parallelism
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

    # Each process gets a slice of the data.
    batch_size = X_data.shape[0] // world_size
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    X_part = X_data[start_idx:end_idx]
    Y_part = Y_data[start_idx:end_idx]

    print(f"Process {rank} handles data from index {start_idx} to {end_idx - 1}")

    model = MLP(X_part.shape[1], 128, Y_part.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_part)
        loss = criterion(outputs, Y_part)
        loss.backward()

        # Manually gather gradients from all processes.
        # This simulates the All-Reduce operation.
        # === 核心修正部分 ===
        # 將梯度從計算圖中分離，以避免序列化問題。
        detached_grads = {name: p.grad.detach() for name, p in model.named_parameters()}
        queue_out.put(detached_grads)

        # Main process (rank 0) waits for and sums all gradients.
        if rank == 0:
            sum_grads = copy.deepcopy(detached_grads)
            for _ in range(world_size - 1):
                other_grads = queue_in.get()
                for name, grad in other_grads.items():
                    sum_grads[name] += grad

            # Apply the averaged gradients.
            for name, param in model.named_parameters():
                param.grad = sum_grads[name] / world_size

            optimizer.step()

            # Print loss from the main process.
            print(f"Data Parallel | Epoch {epoch + 1}/5 | Total Loss: {loss.item():.4f}")
        else:
            # Other processes wait for the main process to complete the step
            # before continuing. In a real-world scenario, this is not needed
            # as All-Reduce is synchronized. Here, it is to ensure order.
            time.sleep(1)


if __name__ == '__main__':
    # 設置多進程的啟動方法為 'spawn'，以確保所有進程都是獨立的。
    multiprocessing.set_start_method('spawn', force=True)

    # 數據維度
    INPUT_DIM = 128
    OUTPUT_DIM = 1

    # 生成數據
    X, Y = generate_data(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)

    # 確保所有數據都在 CPU 上，因為 multiprocessing.Process 之間不能直接共享 GPU 內存。
    device = torch.device('cpu')
    X, Y = X.to(device), Y.to(device)

    # 1. 運行單一設備訓練
    # print("\n--- Start Single-Device Training ---")
    # single_device_train(X, Y, device)

    # 2. 運行資料並行訓練
####    world_size_dp = 2
####    print(f"\n{'=' * 50}\nBeginning Data Parallel Training on {world_size_dp} processes.\n{'=' * 50}")
####
####    queue_out_dp = multiprocessing.Queue()
####    queue_in_dp = multiprocessing.Queue()
####
####    processes_dp = []
####    for rank in range(world_size_dp):
####        p = multiprocessing.Process(target=data_parallel_train,
####                                    args=(X, Y, rank, world_size_dp, queue_in_dp, queue_out_dp))
####        processes_dp.append(p)
####        p.start()
####
####    # 主進程負責協調，這裡讓主進程等待其他進程完成。
####    # 在實際應用中，會用更複雜的同步機制。
####    for p in processes_dp:
####        p.join()

    # 3. 運行張量並行訓練
    world_size_tp = 2
    print(f"\n{'=' * 50}\nBeginning Tensor Parallel Training on {world_size_tp} processes.\n{'=' * 50}")

    queue_out_tp = multiprocessing.Queue()
    queue_in_tp = multiprocessing.Queue()

    processes_tp = []
    for rank in range(world_size_tp):
        p = multiprocessing.Process(target=tensor_parallel_train,
                                    args=(X, Y, rank, world_size_tp, queue_in_tp, queue_out_tp))
        processes_tp.append(p)
        p.start()

    # 等待所有子進程完成
    for p in processes_tp:
        p.join()
