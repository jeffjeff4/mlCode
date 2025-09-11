import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parameter import Parameter
import os


class FSDPParameter(Parameter):
    """FSDP参数包装类"""

    def __new__(cls, data, device=None, requires_grad=True):
        return super().__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, data, device=None):
        self.original_device = device
        self.is_sharded = False
        self.shard = None


class FSDPLinear(nn.Module):
    """FSDP版本的线性层"""

    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        # 计算分片大小 - 处理不能整除的情况
        self.shard_size = out_features // world_size
        self.remainder = out_features % world_size

        # 调整分片大小，确保每个rank都有合理的分片
        if self.remainder > 0 and rank < self.remainder:
            self.shard_size += 1

        # 计算当前rank的分片起始和结束位置
        self.start_idx = 0
        for r in range(rank):
            if r < self.remainder:
                self.start_idx += (out_features // world_size + 1)
            else:
                self.start_idx += (out_features // world_size)

        self.end_idx = self.start_idx + self.shard_size

        # 初始化权重和偏置
        self.weight = FSDPParameter(torch.randn(out_features, in_features))
        self.bias = FSDPParameter(torch.randn(out_features))

        self.weight.is_sharded = True
        self.bias.is_sharded = True

    def shard_parameters(self):
        """分片参数到各个设备"""
        # 权重分片 [out_features, in_features] -> [shard_size, in_features]
        self.weight.shard = self.weight.data[self.start_idx:self.end_idx].detach().clone()
        self.bias.shard = self.bias.data[self.start_idx:self.end_idx].detach().clone()

        # 释放原始参数内存
        #self.weight.data = None
        #self.bias.data = None

    def all_gather_parameters(self):
        """从所有设备收集参数"""
        if not self.weight.is_sharded:
            return

        # 收集所有权重分片
        weight_shards = [None] * self.world_size
        bias_shards = [None] * self.world_size

        # 为每个rank准备正确大小的张量
        for r in range(self.world_size):
            shard_size_r = self.out_features // self.world_size
            remainder_r = self.out_features % self.world_size
            if remainder_r > 0 and r < remainder_r:
                shard_size_r += 1

            weight_shards[r] = torch.zeros(shard_size_r, self.in_features)
            bias_shards[r] = torch.zeros(shard_size_r)

        # 收集分片
        dist.all_gather(weight_shards, self.weight.shard)
        dist.all_gather(bias_shards, self.bias.shard)

        # 重建完整参数
        self.weight.data = torch.cat(weight_shards, dim=0)
        self.bias.data = torch.cat(bias_shards, dim=0)

        self.weight.is_sharded = False
        self.bias.is_sharded = False

    def forward(self, x):
        """前向传播 - 需要时收集参数"""
        if self.weight.is_sharded:
            self.all_gather_parameters()

        return torch.nn.functional.linear(x, self.weight, self.bias)

    def backward_hook(self):
        """反向传播后重新分片参数"""
        if not self.weight.is_sharded and self.weight.grad is not None:
            # 分片梯度
            self.shard_parameters()

            # 释放梯度内存
            self.weight.grad = None
            self.bias.grad = None


class RobustFSDPLinear(nn.Module):
    """能处理任意维度的FSDP线性层"""

    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        # 计算每个rank的分片大小
        base_size = out_features // world_size
        remainder = out_features % world_size

        self.shard_sizes = [base_size + 1 if r < remainder else base_size
                            for r in range(world_size)]

        # 计算当前rank的起始位置
        self.start_idx = sum(self.shard_sizes[:rank])
        self.end_idx = self.start_idx + self.shard_sizes[rank]

        # 初始化参数
        self.weight = Parameter(torch.randn(out_features, in_features))
        self.bias = Parameter(torch.randn(out_features))

        # 立即分片
        self._shard_parameters()

    def _shard_parameters(self):
        """分片参数"""
        self.weight_shard = self.weight.data[self.start_idx:self.end_idx].clone().detach()
        self.bias_shard = self.bias.data[self.start_idx:self.end_idx].clone().detach()

        # 释放原始参数内存
        # my change
        #self.weight.data = None
        #self.bias.data = None
        self.is_sharded = True

    def _gather_parameters(self):
        """收集所有分片"""
        if not self.is_sharded:
            return

        # 收集所有分片
        weight_shards = [torch.zeros(s, self.in_features) for s in self.shard_sizes]
        bias_shards = [torch.zeros(s) for s in self.shard_sizes]

        dist.all_gather(weight_shards, self.weight_shard)
        dist.all_gather(bias_shards, self.bias_shard)

        # 重建完整参数
        self.weight.data = torch.cat(weight_shards, dim=0)
        self.bias.data = torch.cat(bias_shards, dim=0)
        self.is_sharded = False

    def forward(self, x):
        self._gather_parameters()
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        self._shard_parameters()  # 立即重新分片以减少内存占用
        return output


# 使用示例
class FlexibleModel(nn.Module):
    def __init__(self, world_size, rank):
        super().__init__()
        # 现在可以使用任意维度了
        self.layer1 = RobustFSDPLinear(10, 20, world_size, rank)
        self.layer2 = RobustFSDPLinear(20, 5, world_size, rank)  # 5不能被2整除，但现在可以处理了
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class SimpleFSDPModel(nn.Module):
    """简单的FSDP模型 - 修正维度问题"""

    def __init__(self, world_size, rank):
        super().__init__()
        # 确保维度能被world_size整除，或者使用能处理余数的FSDPLinear
        self.layer1 = FSDPLinear(10, 20, world_size, rank)  # 20能被2整除
        self.layer2 = FSDPLinear(20, 6, world_size, rank)  # 改为6，能被2整除

        #self.layer1 = RobustFSDPLinear(10, 20, world_size, rank)  # 20能被2整除
        #self.layer2 = RobustFSDPLinear(20, 6, world_size, rank)  # 改为6，能被2整除

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

    def apply_sharding(self):
        """应用参数分片"""
        self.layer1.shard_parameters()
        self.layer2.shard_parameters()


def setup(rank, world_size):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialized")


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def train_step(rank, world_size):
    """训练步骤示例"""
    try:
        setup(rank, world_size)

        # 创建模型并应用分片
        model = SimpleFSDPModel(world_size, rank)
        model.apply_sharding()

        # 模拟数据
        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 6)  # 匹配输出维度

        # 前向传播
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)

        print(f"Rank {rank}: Loss = {loss.item():.4f}, Output shape: {output.shape}")

        # 反向传播
        loss.backward()

        # 应用分片以减少内存使用
        model.layer1.backward_hook()
        model.layer2.backward_hook()

        print(f"Rank {rank}: Training step completed successfully")

    except Exception as e:
        print(f"Rank {rank}: Error - {e}")
    finally:
        cleanup()


def run_demo():
    """运行演示"""
    world_size = 2
    processes = []

    print(f"Starting FSDP demo with world_size={world_size}")

    for rank in range(world_size):
        p = mp.Process(target=train_step, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("FSDP demo completed")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    run_demo()