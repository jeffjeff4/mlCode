####gemini version

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import os
import copy
from typing import Tuple


def generate_data(num_samples: int = 1000,
                  input_dim: int = 128,
                  output_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic data for a simple MLP.

    Args:
        num_samples (int): Number of data points to generate.
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output target.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors for features (X) and labels (Y).
    """
    print("Generating synthetic data...")
    X = torch.randn(num_samples, input_dim)
    # A simple linear relationship with some noise.
    Y = torch.randn(num_samples, output_dim) * 0.1 + torch.sum(X, dim=1, keepdim=True) * 0.5
    print(f"Data generated. X shape: {X.shape}, Y shape: {Y.shape}")
    return X, Y


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model."""

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


class SimpleTensorParallelMLP(nn.Module):
    """
    A simplified MLP implementation with a manually-split hidden layer
    for Tensor Parallelism.

    This is a conceptual example. Real-world implementations are more
    complex and often use libraries like Megatron-LM.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rank: int, world_size: int):
        super(SimpleTensorParallelMLP, self).__init__()
        self.rank = rank
        self.world_size = world_size

        # Split the hidden layer dimension across devices.
        # This is where the model is partitioned.
        # The hidden_dim must be divisible by world_size.
        assert hidden_dim % world_size == 0, "Hidden dimension must be divisible by world size."
        self.hidden_dim_part = hidden_dim // world_size

        # The first layer is not parallelized. Its weights and biases are
        # present on all devices.
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # The second layer is split. Each device has a slice of the full
        # weight matrix and a slice of the bias vector.
        self.fc2_weight = nn.Parameter(torch.randn(self.hidden_dim_part, output_dim))
        self.fc2_bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass for the first layer (full layer on each device).
        x = self.fc1(x)
        x = nn.functional.relu(x)

        # In tensor parallelism, each device gets a slice of the input tensor
        # to its partitioned layer. Here, the first layer's output is this input.
        # Since the first layer is not split, its output is the same on all devices.
        # So we can just use the local 'x'.

        # Manual forward pass for the second layer using the partitioned weights.
        # The result 'x_part' is only a part of the final output.
        x_part = torch.matmul(x, self.fc2_weight.t())

        # We need to sum up the outputs from all devices to get the final result.
        # This is where communication happens.
        dist.all_reduce(x_part, op=dist.ReduceOp.SUM)

        # Add the bias locally (since the bias is not partitioned).
        x_final = x_part + self.fc2_bias

        return x_final


def train_model(rank: int, world_size: int, parallel_mode: str = 'single_device'):
    """
    Main training loop for a simple MLP.

    Args:
        rank (int): The rank of the current process in the distributed group.
        world_size (int): The total number of processes.
        parallel_mode (str): The parallelization strategy to use.
                             'single_device', 'data_parallel', or 'tensor_parallel'.
    """
    print(f"[{parallel_mode.upper()}]: Initializing process with rank {rank}...")

    # Initialize the distributed environment for DP and TP.
    if parallel_mode in ['data_parallel', 'tensor_parallel']:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Check for GPU availability.
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"[{parallel_mode.upper()}]: Using device: {device}")

    # Generate or load data.
    X, Y = generate_data()
    dataset = TensorDataset(X, Y)

    # Define model hyperparameters.
    INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = X.shape[1], 256, Y.shape[1]
    NUM_EPOCHS = 5
    BATCH_SIZE = 32

    # --- Model and Parallelism Setup ---
    if parallel_mode == 'single_device':
        model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)

    elif parallel_mode == 'data_parallel':
        # DDP is the standard way to do Data Parallelism in PyTorch.
        # It automatically handles data scattering and gradient aggregation.
        model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    elif parallel_mode == 'tensor_parallel':
        # Our custom, simplified Tensor Parallel model.
        # The hidden dimension must be divisible by the world size.
        model = SimpleTensorParallelMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, rank, world_size).to(device)

    # --- Training Loop ---
    # The training loop itself is nearly identical for all modes.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0

        # Create a DataLoader based on the parallel mode.
        # For DP, each process loads a unique part of the data.
        if parallel_mode == 'data_parallel':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            # For Single Device and TP, each process handles the full dataset.
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Reset gradients.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization.
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if rank == 0:  # Print results from a single process to avoid clutter.
            print(f"[{parallel_mode.upper()}] Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    if parallel_mode in ['data_parallel', 'tensor_parallel']:
        dist.destroy_process_group()


if __name__ == '__main__':
    # This example requires a GPU to run.
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU to demonstrate parallel training.")
        print("Falling back to single-device training on CPU.")
        train_model(rank=0, world_size=1, parallel_mode='single_device')
    else:
        # We'll use 2 GPUs to demonstrate parallelism.
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        world_size = 2

        print("--- Running Single-Device Training ---")
        train_model(rank=0, world_size=1, parallel_mode='single_device')
        print("\n--- Running Data Parallel Training (using 2 GPUs) ---")
        torch.multiprocessing.spawn(train_model, args=(world_size, 'data_parallel'), nprocs=world_size, join=True)
        print("\n--- Running Tensor Parallel Training (using 2 GPUs) ---")
        torch.multiprocessing.spawn(train_model, args=(world_size, 'tensor_parallel'), nprocs=world_size, join=True)
