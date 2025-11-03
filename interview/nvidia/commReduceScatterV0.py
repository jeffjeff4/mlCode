####1. python code implement reduce-scatter
####
####2. world_size = int(os.environ.get("WORLD_SIZE", "4"))

####Here is a breakdown of what dist.reduce_scatter does. This is the key to FSDP's backward pass.
####
####What is dist.reduce_scatter?
####
####Motto: "Combine everyone's data, then give everyone their own piece of the result."
####
####Action: This is a many-to-many operation that combines reduce and scatter.
####
####Reduce: It gathers a list of tensors from all processes and applies an operation (e.g., SUM).
####
####Scatter: It gives a unique piece of that final sum to each process.
####
####Analogy: A group of 4 farmers (world_size=4) is working on a 4-acre farm (a model layer).
####
####Each farmer calculates the yield (the gradient) for the entire 4-acre farm, but in 4 separate baskets (the input_list).
####
####Farmer 0 has [A0, A1, A2, A3] (their gradient calculation for each acre)
####
####Farmer 1 has [B0, B1, B2, B3] (their gradient calculation)
####
####Farmer 2 has [C0, C1, C2, C3]
####
####Farmer 3 has [D0, D1, D2, D3]
####
####The reduce_scatter call happens:
####
####Reduce (Summing): The system first sums the baskets for each acre:
####
####Total_Acre_0 = A0 + B0 + C0 + D0
####
####Total_Acre_1 = A1 + B1 + C1 + D1
####
####Total_Acre_2 = A2 + B2 + C2 + D2
####
####Total_Acre_3 = A3 + B3 + C3 + D3
####
####Scatter (Distributing): The system then gives the final total for each acre to the farmer responsible for that acre:
####
####Farmer 0 receives Total_Acre_0.
####
####Farmer 1 receives Total_Acre_1.
####
####Farmer 2 receives Total_Acre_2.
####
####Farmer 3 receives Total_Acre_3.
####
####Why FSDP Needs This
####
####This is exactly how FSDP performs its backward pass:
####
####Each GPU (farmer) calculates the gradient for the full layer (the 4-acre farm), split into world_size shards (the baskets).
####
####reduce-scatter efficiently sums the gradients and delivers the final, correct gradient only for the shard that GPU is responsible for.
####
####This way, each GPU only has to store 1/4th of the final gradients, saving a massive amount of memory.



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


def manual_reduce_scatter(input_list, output_tensor, world_size, rank):
    """
    Manually implements reduce_scatter using a loop of 'reduce' calls.
    This is necessary because the 'gloo' backend does not support
    the built-in 'dist.reduce_scatter'.
    """

    # We will perform one 'reduce' operation for each rank.
    # The i-th 'reduce' will sum the i-th element from everyone's
    # input_list and send the result to rank 'i'.

    for i in range(world_size):
        # 1. Get the local data we want to reduce for this round
        # This is the i-th tensor from our local input_list
        data_to_reduce = input_list[i].clone()  # Use .clone() to be safe

        # 2. Perform the 'reduce' operation
        # The 'dst' (destination) is 'i'.
        # This will sum everyone's 'data_to_reduce' and put the
        # result on rank 'i'.
        dist.reduce(
            tensor=data_to_reduce,
            dst=i,
            op=dist.ReduceOp.SUM
        )

        # 3. If we are the destination rank, copy the result
        #    to our final output tensor.
        if rank == i:
            output_tensor.copy_(data_to_reduce)

    # After this loop, every process will have its correct,
    # reduced, and scattered piece in 'output_tensor'.


def main_worker(rank, world_size):
    """
    The main worker function.
    'rank' and 'world_size' are passed by mp.spawn().
    """
    setup(rank, world_size)

    # --- 1. Create Local Data (The Input) ---
    # Each process must create an input_list with 'world_size' elements.
    # The i-th element in the list is "destined" for the i-th rank.

    # Let's make unique data for each rank
    # Rank 0 creates [1., 10., 100., 1000.]
    # Rank 1 creates [2., 20., 200., 2000.]
    # Rank 2 creates [3., 30., 300., 3000.]
    # Rank 3 creates [4., 40., 400., 4000.]
    input_list = [
        torch.tensor([float(1 * (10 ** i) + rank)], dtype=torch.float32)
        for i in range(world_size)
    ]

    print(f"Rank {rank} | Input List: {[t.item() for t in input_list]}")

    # --- 2. Create Receiver Tensor (The Output) ---
    # Each process needs one tensor to receive its final, reduced piece.
    output_tensor = torch.empty(1, dtype=torch.float32)

    dist.barrier()
    if rank == 0:
        print("\nPerforming *MANUAL* 'reduce_scatter' with SUM...\n", flush=True)
    dist.barrier()

    # --- 3. Perform Manual Reduce-Scatter ---
    manual_reduce_scatter(input_list, output_tensor, world_size, rank)

    # --- 4. Print Results ---
    # Each process will have a *different* final value.
    # Rank 0 should have: 1 + 2 + 3 + 4 = 10
    # Rank 1 should have: 10 + 20 + 30 + 40 = 100
    # Rank 2 should have: 100 + 200 + 300 + 400 = 1000
    # Rank 3 should have: 1000 + 2000 + 3000 + 4000 = 10000

    dist.barrier()
    if rank == 0:
        print("\n--- Final Results ---", flush=True)
    dist.barrier()

    print(f"Rank {rank} | Final Output Tensor: {output_tensor.item()}", flush=True)

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
