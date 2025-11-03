####1. python code implement all-reduce
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

    # --- 1. Create Local Data (The Input) ---
    # Each process creates its own tensor.
    # Rank 0 has [1.]
    # Rank 1 has [2.]
    # Rank 2 has [3.]
    # Rank 3 has [4.]
    local_tensor = torch.tensor([float(rank + 1)], dtype=torch.float32)
    print(f"Rank {rank} | Before all-reduce | Has tensor: {local_tensor.item()}", flush=True)

    dist.barrier()
    ####if rank == 0:
    ####    print("\nPerforming 'all_reduce' with SUM...\n", flush=True)

    print(f"\nrank {rank}, Performing 'all_reduce' with SUM...\n", flush=True)

    dist.barrier()

    # --- 2. Perform All-Reduce ---
    # This operation is IN-PLACE.
    # It will gather all 4 tensors, sum them (1+2+3+4 = 10),
    # and then put that final sum [10.] back into the
    # 'local_tensor' variable on ALL processes.
    dist.all_reduce(
        tensor=local_tensor,
        op=dist.ReduceOp.SUM
    )

    # --- 3. Print Results ---
    # *Everyone* should now have the same final value.
    print(f"\nRank {rank} | After all-reduce  | Has tensor: {local_tensor.item()}", flush=True)

    cleanup()


if __name__ == "__main__":
    # --- This is where you set world_size = 4 ---
    # This script can be run directly from PyCharm's "Run" button.

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
