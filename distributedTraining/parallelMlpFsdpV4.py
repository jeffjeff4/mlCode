####runnable, but can not get any results, can not be used in an interview

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    BackwardPrefetch,
)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 50)
        self.layer2 = torch.nn.Linear(50, 2)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def fsdp_training(rank, world_size):
    setup(rank, world_size)

    # 初始化模型
    model = SimpleModel().to(rank)

    # 使用FSDP包装模型
    model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 模拟数据加载器
    def get_batch(rank):
        torch.manual_seed(42 + rank)  # 每个GPU有不同的数据
        data = torch.randn(32, 10).to(rank)
        target = torch.randint(0, 2, (32,)).to(rank)
        return data, target

    # 训练循环
    for epoch in range(3):
        # 获取当前GPU的批次数据
        data, target = get_batch(rank)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)

        # 方法1: 获取平均损失（推荐）
        reduced_loss = reduce_loss(loss)

        # 方法2: 获取每个GPU的损失（如果需要）
        if epoch == 0:  # 只需要在第一个epoch检查一次
            per_gpu_losses = get_loss_per_gpu(loss)
            if rank == 0:
                print(f"Epoch {epoch}: Per-GPU losses = {per_gpu_losses}")

        # 反向传播
        loss.backward()  # 使用原始损失进行反向传播
        optimizer.step()

        # 只在主进程打印
        if rank == 0:
            print(f"Epoch {epoch}: Average loss = {reduced_loss.item():.4f}")

    cleanup()


def reduce_loss(loss):
    """同步损失 across all GPUs"""
    world_size = dist.get_world_size()
    reduced_loss = loss.clone().detach()
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss /= world_size
    return reduced_loss


def get_loss_per_gpu(loss):
    """获取每个GPU的损失值"""
    world_size = dist.get_world_size()
    loss_tensor = torch.tensor([loss.item()], device=loss.device)
    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss_tensor)
    return [loss.item() for loss in gathered_losses]


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(fsdp_training, args=(world_size,), nprocs=world_size, join=True)