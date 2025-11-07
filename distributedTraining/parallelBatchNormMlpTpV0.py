####python code, for machine learning problem, with test case
####1. the code MUST be CORRECT
####2. consider all cases, including edge cases
####3. make it runnable in this chat session
####4. provide time complexity and space complexity analysis
####5. make the code runnable on cpu
####6. generate simple test datasets
####7. generate train and evaluation code
####8. world_size = int(os.environ.get("WORLD_SIZE", "2"))
####Question:
####1. batch norm code
####2. make it tensor parallel

#### this is grok code
#### the code is correct, could be used in interview

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# --- Distributed Setup ---

def setup_distributed(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    # NOTE: Using 'gloo' for CPU-based multiprocessing.
    # Use 'nccl' for GPU.
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)  # Ensure consistent initialization


def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


# --- Tensor Parallel BatchNorm Implementation ---

class TensorParallelBatchNorm1d(nn.Module):
    """
    A BatchNorm1d layer that is parallel across the feature dimension.

    It is assumed that the input tensor `[N, D_local]` on each rank
    is a slice of a larger tensor `[N, D_total]`, where `D_total` is
    split along dimension 1.

    The gamma, beta, running_mean, and running_var parameters are
    also split, and each rank only stores and updates its local slice.

    This operation requires NO communication between ranks, as the
    statistics for each feature are computed independently.
    """

    def __init__(self,
                 num_features_total: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        super(TensorParallelBatchNorm1d, self).__init__()

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if num_features_total % self.world_size != 0:
            raise ValueError(f"Total number of features ({num_features_total}) "
                             f"must be divisible by world_size ({self.world_size})")

        self.num_features_local = num_features_total // self.world_size

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        factory_kwargs = {'device': device, 'dtype': dtype}

        if self.affine:
            self.weight = Parameter(torch.empty(self.num_features_local, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.num_features_local, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features_local, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(self.num_features_local, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=device))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the tensor-parallel BatchNorm1d.

        Args:
            input (torch.Tensor): The local slice of the input tensor,
                                  with shape [N, D_local].

        Returns:
            torch.Tensor: The output tensor, with shape [N, D_local].
        """
        if input.dim() != 2:
            raise ValueError(f"Expected 2D input (got {input.dim()}D input)")

        if input.shape[1] != self.num_features_local:
            raise ValueError(f"Input feature dimension ({input.shape[1]}) does not "
                             f"match local features ({self.num_features_local})")

        # Use the built-in F.batch_norm function, which is highly optimized.
        # We are simply applying it to our local slice of features,
        # which is exactly what tensor-parallel batchnorm implies.

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps
        )

    def extra_repr(self):
        return (
            f'{self.num_features_local} (local), '
            f'num_features_total={self.num_features_local * self.world_size}, '
            f'eps={self.eps}, momentum={self.momentum}, affine={self.affine}, '
            f'track_running_stats={self.track_running_stats}'
        )


# --- Test and Validation ---

def run_test(rank, world_size):
    """
    Main test function to be run by each process.

    It compares the output of TensorParallelBatchNorm1d against a
    standard nn.BatchNorm1d to verify correctness.
    """
    setup_distributed(rank, world_size)
    if rank == 0:
        print(f"--- Starting Test with world_size = {world_size} ---")

    # --- Test Parameters ---
    BATCH_SIZE = 16
    D_TOTAL = 8  # Total feature dimension
    D_LOCAL = D_TOTAL // world_size

    # We use a fixed seed, so every rank generates the *same*
    # full dataset. This avoids needing to scatter/broadcast.
    torch.manual_seed(123)

    # 1. Create the base dataset (no gradients)
    #    This is the source of truth for the data.
    full_data_src = torch.rand(BATCH_SIZE, D_TOTAL) * 10 - 5

    # 2. Create the input for the serial computation.
    #    We clone() and set requires_grad=True to make it a clean leaf tensor.
    full_data = full_data_src.clone().requires_grad_()

    # Standard BatchNorm layer (will run on rank 0 for comparison)
    serial_bn = nn.BatchNorm1d(D_TOTAL)
    serial_bn.train()  # Set to training mode

    # 3. Create the input for the parallel computation.
    #    We chunk the source, then clone() and set requires_grad=True.
    local_data_slice = full_data_src.chunk(world_size, dim=1)[rank].clone().requires_grad_()

    # 4. Create the Tensor Parallel BatchNorm layer
    #    We must synchronize the initial parameters to match the serial_bn
    tp_bn = TensorParallelBatchNorm1d(D_TOTAL)
    with torch.no_grad():
        # Get the corresponding slices from the serial_bn
        weight_slice = serial_bn.weight.chunk(world_size, dim=0)[rank].clone()
        bias_slice = serial_bn.bias.chunk(world_size, dim=0)[rank].clone()
        tp_bn.weight.copy_(weight_slice)
        tp_bn.bias.copy_(bias_slice)

    tp_bn.train()  # Set to training mode

    # --- Test 1: Training Mode (Forward Pass) ---
    if rank == 0:
        print("\n--- Test 1: Training Mode (Forward) ---")

    # Serial computation
    serial_output = serial_bn(full_data)

    # Parallel computation
    local_output = tp_bn(local_data_slice)

    # Gather results from all ranks
    gathered_output_list = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(gathered_output_list, local_output)
    gathered_output = torch.cat(gathered_output_list, dim=1)

    if rank == 0:
        is_close = torch.allclose(serial_output, gathered_output, atol=1e-6)
        print(f"Forward Pass Correctness: {is_close}")
        if not is_close:
            print("Serial Output:\n", serial_output)
            print("Gathered TP Output:\n", gathered_output)
            print("Difference:\n", serial_output - gathered_output)
        assert is_close, "Training mode forward pass failed!"
        print("Running Mean (Serial):", serial_bn.running_mean)
        print("Running Mean (TP Rank 0):", tp_bn.running_mean)

    # --- Test 2: Training Mode (Backward Pass) ---
    if rank == 0:
        print("\n--- Test 2: Training Mode (Backward) ---")

    # Serial backward
    serial_output.sum().backward()

    # Parallel backward
    local_output.sum().backward()

    # Gather gradients for comparison
    local_weight_grad = tp_bn.weight.grad
    gathered_weight_grad_list = [torch.empty_like(local_weight_grad) for _ in range(world_size)]
    dist.all_gather(gathered_weight_grad_list, local_weight_grad)
    gathered_weight_grad = torch.cat(gathered_weight_grad_list, dim=0)

    local_bias_grad = tp_bn.bias.grad
    gathered_bias_grad_list = [torch.empty_like(local_bias_grad) for _ in range(world_size)]
    dist.all_gather(gathered_bias_grad_list, local_bias_grad)
    gathered_bias_grad = torch.cat(gathered_bias_grad_list, dim=0)

    local_input_grad = local_data_slice.grad
    gathered_input_grad_list = [torch.empty_like(local_input_grad) for _ in range(world_size)]
    dist.all_gather(gathered_input_grad_list, local_input_grad)
    gathered_input_grad = torch.cat(gathered_input_grad_list, dim=1)

    if rank == 0:
        # Compare weight gradients
        is_close_w = torch.allclose(serial_bn.weight.grad, gathered_weight_grad, atol=1e-6)
        print(f"Weight Gradient Correctness: {is_close_w}")
        assert is_close_w, "Weight gradient check failed!"

        # Compare bias gradients
        is_close_b = torch.allclose(serial_bn.bias.grad, gathered_bias_grad, atol=1e-6)
        print(f"Bias Gradient Correctness: {is_close_b}")
        assert is_close_b, "Bias gradient check failed!"

        # This comparison will now work
        is_close_x = torch.allclose(full_data.grad, gathered_input_grad, atol=1e-6)
        print(f"Input Gradient Correctness: {is_close_x}")
        assert is_close_x, "Input gradient check failed!"

    # --- Test 3: Evaluation Mode (Forward Pass) ---
    # We use the running_mean and running_var populated from the
    # training pass above.
    if rank == 0:
        print("\n--- Test 3: Evaluation Mode (Forward) ---")

    serial_bn.eval()
    tp_bn.eval()

    # Create new eval data
    torch.manual_seed(456)
    # No .requires_grad_() needed here since we're in eval mode
    full_eval_data = torch.rand(BATCH_SIZE, D_TOTAL) * 5
    local_eval_data_slice = full_eval_data.chunk(world_size, dim=1)[rank]

    # Serial computation
    serial_output_eval = serial_bn(full_eval_data)

    # Parallel computation
    local_output_eval = tp_bn(local_eval_data_slice)

    # Gather results
    gathered_output_eval_list = [torch.empty_like(local_output_eval) for _ in range(world_size)]
    dist.all_gather(gathered_output_eval_list, local_output_eval)
    gathered_output_eval = torch.cat(gathered_output_eval_list, dim=1)

    if rank == 0:
        is_close_eval = torch.allclose(serial_output_eval, gathered_output_eval, atol=1e-6)
        print(f"Eval Mode Forward Pass Correctness: {is_close_eval}")
        if not is_close_eval:
            print("Serial Eval Output:\n", serial_output_eval)
            print("Gathered TP Eval Output:\n", gathered_output_eval)
            print("Difference:\n", serial_output_eval - gathered_output_eval)
        assert is_close_eval, "Eval mode forward pass failed!"
        print("\n--- All Tests Passed Successfully! ---")

    cleanup_distributed()


def main():
    # Get world size from environment variable, default to 2
    world_size = int(os.environ.get("WORLD_SIZE", "2"))

    # Spawn 'world_size' processes, each running the 'run_test' function
    mp.spawn(run_test,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    # To run this script:
    # 1. From the command line:
    #    python tensor_parallel_batchnorm.py
    # 2. Or, to change the world size (e.g., to 4):
    #    WORLD_SIZE=4 python tensor_parallel_batchnorm.py
    #
    # Note: Requires D_TOTAL (8) to be divisible by WORLD_SIZE.
    #       (e.g., WORLD_SIZE=4 works, WORLD_SIZE=3 will fail)

    main()