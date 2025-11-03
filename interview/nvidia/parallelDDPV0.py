import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os

PYTORCH_ENABLE_MPS_FALLBACK = 1


# --- Dummy Model and Dataset for the sketch ---
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(10), torch.tensor(idx % 5).long()


# --- End of Dummies ---


def setup(rank, world_size):
    """Initialize distributed training with CPU backend"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='gloo',  # Correct for CPU
        init_method='tcp://localhost:12355',
        rank=rank,
        world_size=world_size
    )
    # DELETE THIS LINE: It is incorrect for CPU DDP setup.
    # torch.cpu.set_device(rank)


def cleanup():
    """ Utility to clean up the process group. """
    dist.destroy_process_group()


def main_worker(rank, world_size):
    """
    The main training function that will be run *by each process*.
    'rank' is the process ID (0, 1, 2, or 3).
    """
    print(f"Running DDP worker on rank {rank}.")
    setup(rank, world_size)

    # 1. Define the device as CPU for this worker
    # DELETE: torch.cpu.set_device(rank) in setup is also incorrect.
    # New: Set device explicitly to 'cpu'
    device = torch.device('cpu')

    # --- 3. DATA: Use DistributedSampler ---
    dataset = MyDataset()

    # sampler = None
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # AFTER (DDP):
    # This sampler ensures each process (GPU) gets a unique,
    # non-overlapping subset of the data.
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # Shuffle data partitioning
    )

    # Must set shuffle=False in DataLoader, as the Sampler handles it.
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        sampler=sampler
    )

    # --- 2. MODEL: Wrap the model with DDP ---

    # 2. Move model to the *CPU* device for this rank
    # New: Use .to(device) which is 'cpu'
    model = MyModel().to(device)

    # 3. Wrap the model: NO device_ids for CPU DDP (Gloo)
    # New: Remove device_ids parameter
    ddp_model = DDP(model)

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- 4. TRAINING LOOP: The "Magic" ---
    NUM_EPOCHS = 3
    for epoch in range(NUM_EPOCHS):
        # !! CRITICAL: Tell the sampler what epoch it is.
        # This is required for shuffling to work correctly
        # across epochs.
        sampler.set_epoch(epoch)

        for inputs, labels in dataloader:
            # 4. Move data to the *CPU* device
            # New: Use .to(device) which is 'cpu'
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass using the DDP-wrapped model
            outputs = ddp_model(inputs)

            loss = criterion(outputs, labels)

            # *** THIS IS THE MAGIC ***
            # When loss.backward() is called, DDP automatically
            # triggers an "all-reduce" operation.
            # 1. Each GPU computes its local gradients.
            # 2. All GPUs communicate and *average* their gradients.
            # 3. Each GPU's model receives the *same* averaged gradient.
            loss.backward()

            # optimizer.step() then applies this identical,
            # synchronized update to all models.
            optimizer.step()

        if rank == 0:  # Only print from one process
            print(f"Epoch {epoch} complete.")

    cleanup()


####def main_worker(rank, world_size):
####    """
####    The main training function that will be run *by each process*.
####    'rank' is the GPU ID (0, 1, 2, or 3).
####    """
####    print(f"Running DDP worker on rank {rank}.")
####    setup(rank, world_size)
####
####    # --- 3. DATA: Use DistributedSampler ---
####    dataset = MyDataset()
####
####    # BEFORE (Single GPU):
####    # sampler = None
####    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
####
####    # AFTER (DDP):
####    # This sampler ensures each process (GPU) gets a unique,
####    # non-overlapping subset of the data.
####    sampler = DistributedSampler(
####        dataset,
####        num_replicas=world_size,
####        rank=rank,
####        shuffle=True  # Shuffle data partitioning
####    )
####
####    # Must set shuffle=False in DataLoader, as the Sampler handles it.
####    dataloader = DataLoader(
####        dataset,
####        batch_size=32,
####        shuffle=False,
####        sampler=sampler
####    )
####
####    # --- 2. MODEL: Wrap the model with DDP ---
####    # BEFORE (Single GPU):
####    # model = MyModel().cuda()
####
####    # AFTER (DDP):
####    # 1. Move model to the *specific* GPU for this rank
####    model = MyModel().to(rank)
####    # 2. Wrap the model
####    ddp_model = DDP(model, device_ids=[rank])
####
####    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
####    criterion = nn.CrossEntropyLoss()
####
####    # --- 4. TRAINING LOOP: The "Magic" ---
####    NUM_EPOCHS = 3
####    for epoch in range(NUM_EPOCHS):
####
####        # !! CRITICAL: Tell the sampler what epoch it is.
####        # This is required for shuffling to work correctly
####        # across epochs.
####        sampler.set_epoch(epoch)
####
####        # Loop looks almost identical to single-GPU...
####        for inputs, labels in dataloader:
####            # Move data to the correct GPU
####            inputs = inputs.to(rank)
####            labels = labels.to(rank)
####
####            optimizer.zero_grad()
####
####            # Forward pass using the DDP-wrapped model
####            outputs = ddp_model(inputs)
####
####            loss = criterion(outputs, labels)
####
####            # *** THIS IS THE MAGIC ***
####            # When loss.backward() is called, DDP automatically
####            # triggers an "all-reduce" operation.
####            # 1. Each GPU computes its local gradients.
####            # 2. All GPUs communicate and *average* their gradients.
####            # 3. Each GPU's model receives the *same* averaged gradient.
####            loss.backward()
####
####            # optimizer.step() then applies this identical,
####            # synchronized update to all models.
####            optimizer.step()
####
####        if rank == 0:  # Only print from one process
####            print(f"Epoch {epoch} complete.")
####
####    cleanup()


if __name__ == "__main__":
    # We want to spawn 4 processes, one for each GPU.
    #world_size = torch.cuda.device_count()
    #world_size = torch.cpu.device_count()
    world_size = 4

    if world_size < 4:
        print(f"Only found {world_size} GPUs. This demo needs 4 GPUs.")
    else:
        world_size = 4  # Cap at 4 for this example
        print(f"Spawning {world_size} processes for DDP...")
        # mp.spawn will create 'world_size' processes
        # and run the 'main_worker' function in each,
        # passing in the 'rank' (0, 1, 2, or 3) as the first arg.
        mp.spawn(
            main_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
