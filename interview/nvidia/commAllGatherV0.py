####1. python code implement all gather
####
####2. world_size = int(os.environ.get("WORLD_SIZE", "4"))



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

    try:
        # --- 1. Create Local Data ---
        # Each process creates its own tensor.
        local_tensor = torch.tensor([rank, rank * 2, rank * 3], dtype=torch.int32)

        print(f"Rank {rank} | Local data: {local_tensor}")

        # --- 2. Create Output List ---
        output_list = [torch.empty_like(local_tensor) for _ in range(world_size)]

        # --- 3. Perform All-Gather ---
        dist.all_gather(output_list, local_tensor)

        # --- 4. Print Results ---
        # Add a barrier to make the print statements cleaner
        dist.barrier()

        # We only print from Rank 0 to avoid a messy console
        #if rank == 0:
        #    print(f"\nAll-Gather result (seen from Rank 0): {output_list}\n")

        print(f"\nrank {rank}, All-Gather result (seen from Rank 0): {output_list}\n")

    finally:
        cleanup()


if __name__ == "__main__":
    # --- THIS IS THE CHANGE ---
    # Here, you can define your world_size
    # This code will now work when you press "Run" (`►`) in PyCharm.

    world_size = 4  # <-- You can set this to 2, 4, 8, etc.

    print(f"Starting {world_size} processes using mp.spawn...")

    # mp.spawn is the "in-code" launcher.
    # It will call `main_worker` for each rank from 0 to world_size-1.
    mp.spawn(
        main_worker,
        args=(world_size,),  # Arguments to pass to main_worker
        nprocs=world_size,  # Number of processes to spawn
        join=True  # Wait for all processes to finish
    )

    print("All processes finished.")
