import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import os


# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def setup_fsdp(rank, world_size):
    # FSDP requires the NCCL backend for GPU communication
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_fsdp():
    dist.destroy_process_group()


def fsdp_main(rank, world_size):
    setup_fsdp(rank, world_size)

    # 1. Option to enable CPU offload
    # Setting offload_params=True will move sharded parameters to CPU when not in use.
    # This is a key feature for enabling training of larger models with limited GPU memory,
    # but it comes at the cost of slower training speed due to communication overhead.
    cpu_offload_policy = CPUOffload(offload_params=True)

    # 2. Define the model and wrap it with FSDP
    # The model is placed on the assigned GPU
    model = MyModel().to(rank)
    # The FSDP wrapper handles sharding and offloading
    fsdp_model = FSDP(model, cpu_offload=cpu_offload_policy)

    # 3. Create dummy data and a distributed sampler
    dummy_data = torch.randn(100, 10)
    dummy_labels = torch.randint(0, 5, (100,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)

    sampler = DistributedSampler(dummy_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dummy_dataset, sampler=sampler, batch_size=16)

    # 4. Define the optimizer
    # FSDP manages the optimizer, and it's created after wrapping the model
    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)

    # 5. Training loop
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for data, labels in dataloader:
            data, labels = data.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = fsdp_model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    cleanup_fsdp()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 0:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.multiprocessing.spawn(fsdp_main, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("No GPUs available for FSDP. FSDP requires GPU(s).")