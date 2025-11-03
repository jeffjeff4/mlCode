import multiprocessing as mp
import os
import json
import time
import numpy as np

# Input data: in-memory array with numbers 1 to 1000
data = list(range(1, 1001))

# Number of parallel trainers
N = 4

# Partition data into shards
shard_size = len(data) // N
shards = [data[i * shard_size: (i + 1) * shard_size] for i in range(N)]


def worker(proc_id, shard, results, fail=False, fail_after=-1):
    checkpoint_file = f"checkpoint_{proc_id}.json"
    partial_sum = 0
    start_idx = 0

    # Check for existing checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            cp = json.load(f)
            partial_sum = cp['sum']
            start_idx = cp['index']
        print(f"Trainer {proc_id} resuming from index {start_idx} with sum {partial_sum}")
    else:
        print(f"Trainer {proc_id} starting fresh")

    # Process shard from start_idx
    for i in range(start_idx, len(shard)):
        partial_sum += shard[i]
        time.sleep(0.001)  # Simulate computational work

        # Simulate failure if enabled
        if fail and i == fail_after:
            raise ValueError(f"Simulated failure in trainer {proc_id} at index {i}")

        # Checkpoint every 50 items
        if (i + 1) % 50 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump({'sum': partial_sum, 'index': i + 1}, f)
            print(f"Trainer {proc_id} checkpointed at index {i + 1}")

    # Save final checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({'sum': partial_sum, 'index': len(shard)}, f)

    print(f"Trainer {proc_id} completed with sum {partial_sum}")
    results[proc_id] = partial_sum


def clean_checkpoints():
    for i in range(N):
        file = f"checkpoint_{i}.json"
        if os.path.exists(file):
            os.remove(file)
    print("Checkpoints cleaned")


def main():
    manager = mp.Manager()
    results = manager.list([None] * N)

    # Normal case test
    print("=== Normal Case (No Failure) ===")
    clean_checkpoints()
    processes = []
    for i in range(N):
        p = mp.Process(target=worker, args=(i, shards[i], results, False, -1))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_sum = sum([r for r in results if r is not None])
    expected_sum = sum(data)  # 500500
    print(f"Total sum: {total_sum}")
    assert total_sum == expected_sum, f"Error: Expected {expected_sum}, got {total_sum}"
    print("Normal case passed")

    # Failure-recovery case test
    print("\n=== Failure-Recovery Case ===")
    clean_checkpoints()
    results = manager.list([None] * N)
    processes = []

    # Simulate failure in trainer 2 at index 123
    print("Simulating failure...")
    for i in range(N):
        fail = (i == 2)
        fail_after = 123 if fail else -1
        p = mp.Process(target=worker, args=(i, shards[i], results, fail, fail_after))
        processes.append(p)
        p.start()

    try:
        for p in processes:
            p.join()
    except Exception as e:
        print(f"Caught simulated failure: {e}")

    # Resume from checkpoints
    print("Resuming from checkpoints...")
    results = manager.list([None] * N)
    processes = []
    for i in range(N):
        p = mp.Process(target=worker, args=(i, shards[i], results, False, -1))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    total_sum = sum([r for r in results if r is not None])
    print(f"Total sum after resume: {total_sum}")
    assert total_sum == expected_sum, f"Error: Expected {expected_sum}, got {total_sum}"
    print("Failure-recovery case passed")


if __name__ == "__main__":
    main()