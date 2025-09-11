#geminiV0.py

import multiprocessing as mp
import os
import time
import sys

# Define a directory for storing checkpoints
CHECKPOINT_DIR = "./checkpoints"


def worker(proc_id, shard, results, simulate_failure_flag=False):
    """
    Worker process to simulate a training job on a data shard.

    The worker computes the sum of its assigned data shard and saves its
    intermediate progress as a checkpoint. It includes a simulated
    failure condition to demonstrate fault tolerance.
    """
    # Check if a checkpoint exists for this worker
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{proc_id}.txt")
    partial_sum = 0
    start_index = 0

    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                data = f.read().split(',')
                partial_sum = float(data[0])  # Use float to prevent type errors
                start_index = int(data[1])
            print(
                f"Process {proc_id}: Resuming from checkpoint. Partial sum: {partial_sum}, starting from index {start_index}")
        else:
            print(f"Process {proc_id}: Starting new training job.")

        # Iterate through the shard, starting from the last checkpointed index
        for i in range(start_index, len(shard)):
            num = shard[i]
            partial_sum += num

            # Simulate a crash. This will be triggered on a specific worker for the failure test.
            if simulate_failure_flag and i == len(shard) // 2:
                print(f"Process {proc_id}: Simulating a failure and exiting.")
                # Force an exit, preventing the result from being written
                sys.exit(1)

            # Periodically save a checkpoint
            if (i + 1) % 100 == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    f.write(f"{partial_sum},{i + 1}")
                # print(f"Process {proc_id}: Checkpoint saved at step {i + 1}.")

        results[proc_id] = partial_sum
        print(f"Process {proc_id}: Finished. Partial sum is {partial_sum}.")

    except Exception as e:
        print(f"Process {proc_id} failed with an error: {e}")
        # The result will not be written, leaving a None in the shared list.


def create_data_shards(data, num_processes):
    """
    Splits the data into N non-overlapping shards.
    """
    avg_shard_size = len(data) // num_processes
    shards = []
    for i in range(num_processes):
        start = i * avg_shard_size
        end = (i + 1) * avg_shard_size if i < num_processes - 1 else len(data)
        shards.append(data[start:end])
    return shards


def clean_checkpoints():
    """
    Removes all checkpoint files.
    """
    print("Cleaning up old checkpoints...")
    if os.path.exists(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            os.remove(os.path.join(CHECKPOINT_DIR, f))
        os.rmdir(CHECKPOINT_DIR)
    print("Checkpoints cleaned.")


def run_simulation(data, num_processes, simulate_failure=False):
    """
    Runs a single pass of the distributed training simulation.
    """
    if not simulate_failure:
        print("\n--- Running Simulation ---")

    # Create data shards
    shards = create_data_shards(data, num_processes)

    # Create a Manager to share data between processes
    manager = mp.Manager()
    results = manager.list([None] * num_processes)

    processes = []
    for i in range(num_processes):
        p_args = (i, shards[i], results)

        # Pass a flag to the worker process to simulate failure
        if simulate_failure and i == 1:
            p = mp.Process(target=worker, args=p_args + (True,))
        else:
            p = mp.Process(target=worker, args=p_args)

        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Aggregate results, handling potential failures (None values)
    final_sum = 0
    incomplete_run = False
    for res in results:
        if res is not None:
            final_sum += res
        else:
            incomplete_run = True

    print("\n--- Aggregated Results ---")
    print(f"Calculated Sum: {final_sum}")
    return final_sum, incomplete_run


def main():
    """
    Main function to orchestrate the simulation and run tests.
    """
    DATA_SIZE = 1000
    data = list(range(1, DATA_SIZE + 1))
    EXPECTED_SUM = sum(data)
    NUM_PROCESSES = 4

    # --- Case 1: Normal execution ---
    clean_checkpoints()
    final_sum, _ = run_simulation(data, NUM_PROCESSES, simulate_failure=False)
    print(f"Expected Sum: {EXPECTED_SUM}")
    print(f"Normal Case Passed: {final_sum == EXPECTED_SUM}")

    # --- Case 2: Failure and recovery ---
    print("\n\n--- Running Failure & Recovery Test ---")
    print("--- First run: Simulating a process failure ---")
    clean_checkpoints()
    _, incomplete_run = run_simulation(data, NUM_PROCESSES, simulate_failure=True)

    if incomplete_run:
        print("\nFAILURE DETECTED. The first run was incomplete due to a simulated crash.")
        print("Initiating a new run to RECOVER from the last checkpoint.")

        # --- Second run: Recovery from checkpoint ---
        print("\n--- Second run: Recovery ---")
        # Do not clean checkpoints, so workers can resume from where they left off.
        final_sum, _ = run_simulation(data, NUM_PROCESSES, simulate_failure=False)

        print("\n--- Recovery Test Results ---")
        print(f"Expected Sum: {EXPECTED_SUM}")
        print(f"Final Sum (after recovery): {final_sum}")
        print(f"Recovery Case Passed: {final_sum == EXPECTED_SUM}")
    else:
        print("Error: The failure simulation did not behave as expected.")


if __name__ == "__main__":
    main()
