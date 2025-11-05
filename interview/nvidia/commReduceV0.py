import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def setup(rank, world_size):
    """Initializes the distributed process group (for CPU)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Any free port

    # Use 'gloo' backend for CPU communication
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Initialized process {rank}/{world_size} on 'gloo' backend.")


def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


def main_worker(rank, world_size):
    """
    The main worker function.
    'rank' and 'world_size' are passed by mp.spawn().
    """
    setup(rank, world_size)

    # --- 1. Create Local Data ---
    # Each process creates its own tensor.
    # We use a float for operations like SUM.
    local_tensor = torch.tensor([rank + 1], dtype=torch.float32)

    print(f"Rank {rank} | Local data: {local_tensor.item()}")

    # Define the destination rank for the reduce operation
    destination_rank = 0

    # Wait for all processes to print their local data
    dist.barrier()

    if rank == 0:
        print(f"\nPerforming 'reduce' to Rank {destination_rank} with SUM...\n")

    dist.barrier()

    # --- 2. Perform Reduce ---
    # This operation gathers the tensor from *all* processes,
    # performs the 'op' (SUM), and stores the result *only*
    # on the 'dst' (destination) rank.
    dist.reduce(
        tensor=local_tensor,
        dst=destination_rank,
        op=dist.ReduceOp.SUM
    )

    # --- 3. Print Results ---
    # Only the destination rank will have the updated tensor.
    # For all other ranks, the tensor is unchanged.

    if rank == destination_rank:
        print(f"Rank {rank} (Destination) | Final reduced sum: {local_tensor.item()}")
    else:
        print(f"Rank {rank} (Worker)        | My tensor is unchanged: {local_tensor.item()}")

    cleanup()


if __name__ == "__main__":
    # --- This is where you set world_size = 4 ---
    # This script can be run directly from PyCharm.

    world_size = 4  # <-- You can set this to 2, 4, 8, etc.

    print(f"Starting {world_size} processes using mp.spawn...")

    # mp.spawn is the "in-code" launcher.
    mp.spawn(
        main_worker,
        args=(world_size,),  # Arguments to pass to main_worker
        nprocs=world_size,  # Number of processes to spawn
        join=True  # Wait for all processes to finish
    )

    print("All processes finished.")
