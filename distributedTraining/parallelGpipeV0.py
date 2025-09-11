####实现 GPipe 算法而不直接调用 torchgpipe 库是一项复杂的任务，因为它涉及手动管理模型分区、微批次调度、跨 GPU/CPU 通信以及梯度检查点（Checkpointing）以节省内存。
####
####下面的代码提供了一个简化的 GPipe 实现。这个版本旨在展示 GPipe 的核心思想，即模型并行与微批次管道化。
####重要简化说明：
####
####无实际梯度检查点：为了代码的简洁性，此示例不包含完整的梯度检查点（recomputing intermediate activations）。在真实的 GPipe 中，中间激活值会在反向传播时重新计算以节省内存，而此示例会存储它们。
####
####点对点通信：使用 dist.send 和 dist.recv 进行微批次数据的传递。
####dist.barrier() 用于同步：在关键点使用 dist.barrier() 来强制同步，模拟 GPipe 的严格分阶段（fill-and-drain）执行模式。
####
####灵活的设备支持：通过 device_type 参数，代码可以在 CPU 或 CUDA（GPU）上运行。
####
####GPipe 核心逻辑
####GPipe 算法的两个主要阶段是：
####前向传播：将一个 Mini-Batch 划分为多个 Micro-Batch。每个 Micro-Batch 依次通过模型在不同设备上的分区。
####反向传播：当最后一个 Micro-Batch 完成前向传播后，反向传播开始，并以相反的顺序通过分区。梯度在所有 Micro-Batch 上累积，最后一步更新权重。



import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import functools
import torch.optim as optim

# --- 1. 定义一个简单的多层模型 ---
class SimpleSequentialModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # No final activation for MSELoss target
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# --- 2. 设置分布式环境 ---
def setup_distributed(rank, world_size, device_type):
    """根据设备类型初始化分布式环境。"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # 确保端口未被占用
    backend = 'gloo' if device_type == 'cpu' else 'nccl'
    if device_type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot use 'cuda' device type.")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device_type == 'cuda':
        torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized on {device_type}.")


def cleanup_distributed():
    """销毁分布式进程组。"""
    dist.destroy_process_group()


# --- 3. GPipe 训练逻辑 ---
def run_gpipe_process(rank, world_size, device_type, model_config):
    setup_distributed(rank, world_size, device_type)
    current_device = torch.device(device_type if device_type == 'cpu' else f'cuda:{rank}')

    in_dim, hidden_dim, out_dim = model_config

    # 3.1. 手动划分模型 (简化示例，实际应动态划分)
    # 假设模型被平均分成3个逻辑部分，部署在世界大小为3的GPU上。
    # 如果world_size != 3，这里会出问题，实际应用需要更灵活的划分策略。
    if world_size < 1:
        raise ValueError("world_size must be at least 1 for this example.")

    if rank == 0:
        model_part = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
    elif rank == 1:
        model_part = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
    elif rank == 2 and world_size >= 3:
        model_part = nn.Sequential(nn.Linear(hidden_dim, out_dim))
    else:  # Fallback for world_size < 3 or if rank is outside assumed partitions
        # This part handles cases where world_size doesn't match predefined partitions,
        # or if a rank has no specific part. In a real GPipe, the model would be split dynamically.
        # For simplicity, we just make it an identity or empty module for ranks not in the 3-stage example.
        model_part = nn.Identity()

    model_part.to(current_device)
    optimizer = optim.SGD(model_part.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    MINI_BATCH_SIZE = 32
    MICRO_BATCH_SIZE = 8
    num_micro_batches = MINI_BATCH_SIZE // MICRO_BATCH_SIZE

    # 用于存储前向传播的激活值（用于反向传播）和损失
    # 真实 GPipe 会用梯度检查点节省内存，这里简化为存储
    forward_outputs = [None] * num_micro_batches
    micro_batch_inputs = [None] * num_micro_batches

    losses = []

    # --- 前向传播阶段 ---
    if rank == 0:
        print(f"\n--- Rank {rank} (GPU {rank if device_type == 'cuda' else 'CPU'}) 前向传播开始 ---")

    for i in range(num_micro_batches):
        # 1. 获取输入微批次
        if rank == 0:
            # 根进程生成数据
            micro_batch_input = torch.randn(MICRO_BATCH_SIZE, in_dim, requires_grad=True).to(current_device)
            micro_batch_inputs[i] = micro_batch_input  # 存储原始输入用于反向
            # print(f"Rank {rank}: 生成微批次 {i} 输入")
        else:
            # 其他进程接收上一个进程的输出作为输入
            # wrong???
            #micro_batch_input = torch.empty(MICRO_BATCH_SIZE, in_dim if rank == 1 else hidden_dim,
            #                                requires_grad=True).to(current_device)
            micro_batch_input = torch.empty(MICRO_BATCH_SIZE, hidden_dim,
                                            requires_grad=True).to(current_device)
            dist.recv(tensor=micro_batch_input, src=rank - 1)
            # print(f"Rank {rank}: 接收微批次 {i} 输入")
            micro_batch_inputs[i] = micro_batch_input  # 存储接收到的输入用于反向

        # 2. 执行本地前向计算
        micro_batch_output = model_part(micro_batch_input)

        # 3. 存储输出（用于反向）并发送给下一个进程
        if rank < world_size - 1:
            # 确保输出可梯度追踪，因为下一个阶段会依赖它
            if not micro_batch_output.requires_grad:
                micro_batch_output.requires_grad_(True)
            forward_outputs[i] = micro_batch_output  # 存储中间输出
            dist.send(tensor=micro_batch_output, dst=rank + 1)
            # print(f"Rank {rank}: 发送微批次 {i} 输出")
        else:
            # 最后一个进程计算损失
            target = torch.randn(MICRO_BATCH_SIZE, out_dim).to(current_device)
            loss = criterion(micro_batch_output, target)
            losses.append(loss)
            forward_outputs[i] = loss  # 存储损失以便反向传播
            # print(f"Rank {rank}: 计算微批次 {i} 损失")

        # 确保所有前向传播完成后再开始反向
        dist.barrier()

    if rank == 0:
        print(f"\n--- Rank {rank} (GPU {rank if device_type == 'cuda' else 'CPU'}) 反向传播开始 ---")

    # --- 反向传播阶段 ---
    # 反向传播从最后一个微批次开始，并计算梯度
    for i in reversed(range(num_micro_batches)):
        # 1. 获取损失或上游梯度
        if rank == world_size - 1:  # 最后一个进程
            current_loss = forward_outputs[i]  # 获取之前存储的损失
            current_loss.backward()
            # print(f"Rank {rank}: 微批次 {i} 反向传播完成")
        else:  # 中间进程
            # 接收来自下一个进程的梯度
            grad_from_next_stage = torch.empty_like(forward_outputs[i])
            dist.recv(tensor=grad_from_next_stage, src=rank + 1)
            forward_outputs[i].backward(grad_from_next_stage)
            # print(f"Rank {rank}: 接收到微批次 {i} 梯度并完成反向传播")

        # 2. 发送梯度给上一个进程 (如果不是第一个进程)
        if rank > 0:
            # 获取当前分区输入张量的梯度，并发送给上一个进程
            grad_to_prev_stage = micro_batch_inputs[i].grad
            if grad_to_prev_stage is not None:
                dist.send(tensor=grad_to_prev_stage, dst=rank - 1)
                # print(f"Rank {rank}: 发送微批次 {i} 梯度")
            else:
                # 如果这个输入不依赖于上一个stage的梯度，可以发送一个dummy或者跳过
                pass  # This is a simplification; a real GPipe would handle this more robustly

        dist.barrier()  # 确保所有反向传播完成后再进行下一步

    # --- 优化器更新 ---
    optimizer.step()
    optimizer.zero_grad()

    # 仅在计算了损失的进程上执行求和和打印
    if rank == world_size - 1:
        if len(losses) > 0:
            total_loss = sum(l.item() for l in losses) / len(losses)
            print(f"\nRank {rank}: 训练完成。平均损失: {total_loss:.4f}")
        else:
            print(f"\nRank {rank}: 训练完成。未计算损失。")

    cleanup_distributed()


# --- 主函数，用于启动多进程 ---
if __name__ == "__main__":
    # 配置模型参数
    # in_dim -> hidden_dim -> hidden_dim -> out_dim
    model_config = (1024, 1024, 1024)

    # 选择设备类型: 'cpu' 或 'cuda'
    # 注意: 如果选择 'cuda' 但没有 GPU，会抛出 RuntimeError
    # world_size 决定了模型的划分数量和并发进程数
    # 此示例设计为 3 个分区。如果您有更多 GPU，请调整 model_config 和分区逻辑。

    # 尝试在 CPU 上运行
    # desired_device_type = 'cpu'
    # world_size = 3 # 3个进程在CPU上模拟3个分区

    # 尝试在 GPU 上运行 (需要至少 3 个 GPU)
    desired_device_type = 'cuda'
    if desired_device_type == 'cuda' and torch.cuda.is_available():
        world_size = min(3, torch.cuda.device_count())  # 最多使用3个GPU
        if world_size < 3:
            print(
                f"Warning: Only {world_size} GPUs available. This example assumes 3 model partitions for full functionality. Adjusting world_size to {world_size}.")
    elif desired_device_type == 'cuda':
        print("CUDA not available. Falling back to CPU.")
        desired_device_type = 'cpu'
        world_size = 3
    else:
        world_size = 3

    print(f"Running GPipe example with {world_size} processes on {desired_device_type}...")

    mp.spawn(run_gpipe_process,
             args=(world_size, desired_device_type, model_config),
             nprocs=world_size,
             join=True)