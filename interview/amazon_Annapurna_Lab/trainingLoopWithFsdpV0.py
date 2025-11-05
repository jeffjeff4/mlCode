import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
import functools


# --- 1. Distributed Setup Functions ---

def setup_distributed():
    """
    Initializes the distributed process group. torchrun will set the necessary
    environment variables (RANK, LOCAL_RANK, WORLD_SIZE).
    """
    dist.init_process_group("nccl")
    # The device ID is the local rank.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


# --- 2. Model Definition ---

class SimpleModel(nn.Module):
    """A simple model to demonstrate FSDP."""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


# --- 3. The Main Training Loop ---

def train(model, rank, world_size, train_loader, optimizer, epochs, loss_fn):
    """
    The core FSDP training loop.
    """
    model.train()

    for epoch in range(epochs):
        # The DistributedSampler needs to know the current epoch to shuffle properly.
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to the correct device for the current rank
            data, target = data.to(rank), target.to(rank)

            # --- Forward Pass ---
            optimizer.zero_grad()
            output = model(data)

            # --- Define and Calculate Loss ---
            loss = loss_fn(output, target)
            total_loss += loss.item()

            # --- Backward Pass ---
            # Gradients are calculated and sharded automatically by FSDP.
            loss.backward()

            # --- Update Parameters ---
            # FSDP handles the gradient reduction (All-Reduce) and update.
            optimizer.step()

        # Print loss only from the main process (rank 0) to avoid clutter.
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            if (epoch % 10) == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")


# --- 4. Main Execution Block (Serves as Test Case) ---

def main():
    setup_distributed()

    # Get rank and world size from the environment
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # --- Generate a simple training dataset ---
    # The dataset is identical on all processes, but the sampler will give each
    # process a unique slice.
    if rank == 0:
        print("--- Generating a simple training dataset ---")
    # We want to learn the function y = 3x^2 + 2x - 1
    X = torch.linspace(-5, 5, 2048).view(-1, 1)
    Y = 3 * X ** 2 + 2 * X - 1 + torch.randn_like(X) * 2  # Add noise

    dataset = TensorDataset(X, Y)
    # The DistributedSampler is ESSENTIAL for FSDP/DDP.
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # --- Model Initialization and FSDP Wrapping ---
    model = SimpleModel(input_size=1, hidden_size=50, output_size=1).to(local_rank)

    # FSDP wrapping must happen before the optimizer is created.
    # This automatically shards the model's parameters across the available GPUs.
    fsdp_model = FSDP(model, device_id=local_rank)

    # --- Optimizer and Loss Function ---
    optimizer = optim.Adam(fsdp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # --- Run Training ---
    if rank == 0:
        print(f"--- Starting FSDP training on {world_size} GPU(s) ---")

    train(fsdp_model, local_rank, world_size, train_loader, optimizer, epochs=101, loss_fn=loss_fn)

    # --- Test the trained model on rank 0 ---
    if rank == 0:
        print("\n--- Testing Trained Model on Rank 0 ---")
        test_point = torch.tensor([[2.0]], device=local_rank)
        true_value = 3 * 2.0 ** 2 + 2 * 2.0 - 1

        # In FSDP, model parameters are sharded. To run inference on a single
        # rank, you need to gather them first. For simplicity in this example,
        # we can assume for a single test point the overhead is minimal, but
        # for proper evaluation, a gather operation would be needed.
        # This will work correctly as FSDP handles the gather for the forward pass.
        prediction = fsdp_model(test_point)

        print(f"Input: {test_point.item():.2f}")
        print(f"True Value (y=3x^2+2x-1): {true_value:.4f}")
        print(f"Model Prediction: {prediction.item():.4f}")
        assert abs(prediction.item() - true_value) < 5.0, "Model did not learn effectively."
        print("\nTest passed: Model learned the relationship reasonably well.")

    cleanup_distributed()


if __name__ == '__main__':
    # This script is intended to be launched with `torchrun`.
    # For example: torchrun --nproc_per_node=2 fsdp_training_loop.py
    main()

####```

### How to Run This Code

### How to Run This Code

####This script is designed to be run in a distributed environment. You **cannot** run it with `python fsdp_training_loop.py`. You must use the `torchrun` launcher.
####
####**To run on a machine with 2 GPUs:**
####```bash
####torchrun --nproc_per_node=2 fsdp_training_loop.py
####```
####
####**To run in a mock single-GPU session (for testing the logic):**
####```bash
####torchrun --nproc_per_node=1 fsdp_training_loop.py
