import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
import os
import sys
from typing import Dict, Any, List


# --- 0. Setup: Initialize Distributed Environment ---

def setup_distributed_environment(rank: int, world_size: int):
    """Initializes a mock distributed environment using the Gloo backend."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        print(f"[RANK {rank}] Distributed backend 'gloo' initialized (World Size: {world_size}).")


# --- 1. Define the Module ---

class SimpleMLP(nn.Module):
    def __init__(self, in_f: int = 2048, out_f: int = 10):
        super().__init__()
        self.target_layer = nn.Linear(in_f, out_f, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.target_layer(x))


# --- 2. Print Internal FSDP Structure (FIXED ACCESS) ---

def print_fsdp_internal_state(fsdp_layer: FSDP, rank: int, world_size: int):
    """
    打印 FSDP 包装器在当前进程上的核心状态。
    使用 getattr 尝试访问 _local_shard，并在失败时提供理论值。
    """

    # 访问扁平化参数
    flat_param = fsdp_layer._flat_param

    print(f"\n{'=' * 20} [RANK {rank}] FSDP 狀態開始 {'=' * 20}")

    # 1. 扁平化参数 (_flat_param) 状态
    print(f"--- 1. 扁平化參數 (_flat_param) ---")

    if flat_param is not None:
        total_logical_elements = flat_param.numel()
        print(f"  [RANK {rank}] 總邏輯元素 (Total): {total_logical_elements}")

    # 2. 局部分片數據 (Local Sharded Data)
    print("\n--- 2. 本地分片數據 (Local Shards) ---")

    local_shard_numel = 0
    local_shard_source = "理論計算值"

    try:
        # 尝试调用 _local_shard()。这是最直接的方法。
        # 如果 hasattr 检查失败，我们仍然尝试调用它，或者访问其他已知的内部属性。
        if hasattr(fsdp_layer, '_local_shard'):
            local_shard_data = fsdp_layer._local_shard()
            local_shard_numel = local_shard_data.numel()
            local_shard_source = "_local_shard() 實測值"
        elif hasattr(fsdp_layer, '_handles') and fsdp_layer._handles:
            # 如果 _local_shard 不可用，尝试通过 Param Handle 访问
            local_shard_data = fsdp_layer._handles[0].flat_param.local_sharded_data
            local_shard_numel = local_shard_data.numel()
            local_shard_source = "_handles[0].local_sharded_data"

        # 如果以上方法都失败或未返回任何值，则退回到理论计算值
        if local_shard_numel == 0 and total_logical_elements > 0:
            raise Exception("Access failed, reverting to theoretical.")

    except Exception as e:
        # 如果访问内部属性失败 (例如 AttributeError 或未初始化)，则使用理论值
        local_shard_numel = total_logical_elements // world_size
        local_shard_source = f"理論計算值 (基於錯誤: {type(e).__name__})"

    # 打印最终结果
    print(f"  [RANK {rank}] 獲取分片數據來源: {local_shard_source}")
    print(f"  [RANK {rank}] 總本地分片元素數: {local_shard_numel}")

    if total_logical_elements > 0 and world_size > 1:
        shard_ratio = local_shard_numel / total_logical_elements
        expected_ratio = 1 / world_size
        print(f"  [RANK {rank}] 分片比例 (Shard Ratio): {shard_ratio:.4f} (期望值: {expected_ratio:.4f})")

    print(f"\n--- 3. 原始 Linear 結構 ---")
    linear_module = fsdp_layer.module
    print(f"  [RANK {rank}] 被包装模块: {linear_module}")
    print(f"  [RANK {rank}] Weight Size (Logical): {linear_module.weight.size()}")

    print(f"{'=' * 20} [RANK {rank}] FSDP 狀態結束 {'=' * 20}")


# --- 3. FSDP Worker Function ---

def fsdp_worker(rank: int, world_size: int, device: torch.device):
    """
    每個進程執行的函數，負責初始化分佈式環境並運行 FSDP 邏輯。
    """
    try:
        setup_distributed_environment(rank, world_size)

        # 1. 創建基礎模塊
        base_module = SimpleMLP()
        target_layer = base_module.target_layer.to(device)

        # 2. FSDP 包装
        fsdp_layer = FSDP(
            target_layer,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=device,
        )

        # 3. 打印 FSDP 狀態
        print_fsdp_internal_state(fsdp_layer, rank, world_size)

    except Exception as e:
        print(f"[RANK {rank}] FATAL ERROR: {e}", file=sys.stderr)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# --- 4. Main Execution (Multi-Processing Spawner) ---

if __name__ == '__main__':
    WORLD_SIZE = 2

    print(f"Running simulation on CPU with WORLD_SIZE={WORLD_SIZE} processes (Gloo backend).")
    devices = [torch.device('cpu')] * WORLD_SIZE

    mp.spawn(
        fsdp_worker,
        args=(WORLD_SIZE, devices[0]),
        nprocs=WORLD_SIZE,
        join=True
    )