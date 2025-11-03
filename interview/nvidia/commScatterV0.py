####1. python code implement scatter
####
####2. world_size = int(os.environ.get("WORLD_SIZE", "4"))

####Here is a breakdown of what dist.scatter does.
####
####What is dist.scatter?
####
####Motto: "The leader deals the data."
####
####Action: This is a one-to-many operation. One process (the src rank) starts with a list of tensors. It "scatters" this list, sending a single, unique item from the list to each process in the group.
####
####Analogy: A teacher (Source Rank) has a stack of 4 different worksheets: [Worksheet_A, Worksheet_B, Worksheet_C, Worksheet_D].
####
####The teacher gives Worksheet_A to Student 0 (Rank 0).
####
####The teacher gives Worksheet_B to Student 1 (Rank 1).
####
####The teacher gives Worksheet_C to Student 2 (Rank 2).
####
####The teacher gives Worksheet_D to Student 3 (Rank 3).
####At the end, each student has one unique worksheet.
####
####Visual Explanation (world_size=4, src=0)
####
####Start: Only Rank 0 has the data.
####
####Rank 0 has [ tensor([10]), tensor([20]), tensor([30]), tensor([40]) ]
####
####Rank 1 has None
####
####Rank 2 has None
####
####Rank 3 has None
####
####dist.scatter(tensor, scatter_list, src=0) is called.
####
####End: Each process has one piece of the original list.
####
####Rank 0's receiver_tensor is now tensor([10])
####
####Rank 1's receiver_tensor is now tensor([20])
####
####Rank 2's receiver_tensor is now tensor([30])
####
####Rank 3's receiver_tensor is now tensor([40])
####
####scatter vs. all_gather (Opposite Operations)
####
####scatter (One-to-Many): One process's list becomes many processes' single tensors.
####
####Rank 0: [A, B, C, D] -> Rank 0: A, Rank 1: B, Rank 2: C, Rank 3: D
####
####all_gather (Many-to-All): Many processes' single tensors become all processes' list.
####
####Rank 0: A, Rank 1: B, Rank 2: C, Rank 3: D -> Rank 0: [A, B, C, D], Rank 1: [A, B, C, D], etc.


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

    # --- 1. Create Data on the Source Rank ---
    source_rank = 0
    scatter_list = None

    if rank == source_rank:
        # Only the source rank needs the full list of data to send
        scatter_list = [
            torch.tensor([10, 11]),
            torch.tensor([20, 21]),
            torch.tensor([30, 31]),
            torch.tensor([40, 41])
        ]
        # Ensure the list is the same size as world_size
        assert len(scatter_list) == world_size
        print(f"Rank {rank} (Source) | Scattering this list: {scatter_list}\n")

    # --- 2. Create Receiver Tensor on All Ranks ---
    # All processes need a tensor to receive their piece of the data.
    # It must be the same shape as the items in the list.
    receiver_tensor = torch.empty(2, dtype=torch.int64)

    # Wait for Rank 0 to print its message
    dist.barrier()

    # --- 3. Perform Scatter ---
    # `scatter` takes the `scatter_list` from the `src` rank
    # and "deals" one item to each process (including itself).
    # The item is copied into the `tensor` argument.
    dist.scatter(
        tensor=receiver_tensor,
        scatter_list=scatter_list,
        src=source_rank
    )

    # --- 4. Print Results ---
    # Each process now has its own, unique piece of the data.
    print(f"Rank {rank} | I received: {receiver_tensor}")

    cleanup()


if __name__ == "__main__":
    # --- This is where you set world_size = 4 ---
    # This script can be run directly from PyCharm.

    world_size = 4  # <-- You can set this to 2, 4, etc.

    print(f"Starting {world_size} processes using mp.spawn...")

    # mp.spawn is the "in-code" launcher.
    mp.spawn(
        main_worker,
        args=(world_size,),  # Arguments to pass to main_worker
        nprocs=world_size,  # Number of processes to spawn
        join=True  # Wait for all processes to finish
    )

    print("All processes finished.")
