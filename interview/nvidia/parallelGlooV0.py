import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


# 1. Define the model class OUTSIDE of the if __name__ == "__main__": block
# This ensures child processes can find and pickle it.
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 10)

    def forward(self, x):
        return self.linear(x)


# --------------------------------------------------------------------------

def setup(rank, world_size):
    """Initialize distributed training on CPU"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Use 'gloo' backend for CPU distributed training
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, model_class, dataset, **kwargs):
    """Main worker function for CPU distributed training"""
    setup(rank, world_size)

    # ... (rest of main_worker remains the same)
    device = torch.device('cpu')
    torch.set_num_threads(1)

    model = model_class().to(device)
    ddp_model = DDP(model)

    # Your training code here...
    print(f"Rank {rank}: Model setup complete on CPU")

    optimizer = torch.optim.Adam(ddp_model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    data = torch.randn(32, 1000)
    target = torch.randint(0, 10, (32,))

    output = ddp_model(data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Rank {rank}: Training step completed, Loss: {loss.item():.4f}")

    cleanup()


if __name__ == "__main__":
    world_size = 4

    # The SimpleModel class is now passed from the global scope
    mp.spawn(
        main_worker,
        args=(world_size, SimpleModel, None),  # Passed correctly
        nprocs=world_size,
        join=True
    )