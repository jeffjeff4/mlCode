import threading
import time
import random

# A thread-safe list to store results and errors from each thread.
# This prevents race conditions when multiple threads try to write
# to the same list simultaneously.
from collections import deque


class SharedData:
    """A simple class to hold shared data in a thread-safe way."""

    def __init__(self):
        self.results = deque()
        self.lock = threading.Lock()


def worker_task(thread_id, shared_data):
    """
    A worker function for each thread.

    This function simulates a task that could succeed or fail.
    Failure is simulated by raising an exception.
    """
    try:
        # Simulate a random task duration between 0.5 and 2 seconds.
        task_duration = random.uniform(0.5, 2.0)
        time.sleep(task_duration)

        # Simulate a 20% chance of failure.
        if random.random() < 0.2:
            raise ValueError(f"Task for thread {thread_id} failed due to a simulated error.")

        result_message = f"Thread {thread_id} completed successfully in {task_duration:.2f} seconds."

        # Use a lock to ensure thread-safe writing to the shared data.
        with shared_data.lock:
            shared_data.results.append(result_message)

    except Exception as e:
        error_message = f"Thread {thread_id} caught an exception: {e}"
        # Use a lock to ensure thread-safe writing to the shared data.
        with shared_data.lock:
            shared_data.results.append(error_message)


def run_multi_threaded_process(num_threads):
    """
    Manages the creation and execution of multiple threads.

    This function starts each thread, waits for them to complete, and then
    processes the results.
    """
    # Create an instance of our shared data object.
    shared_data = SharedData()

    threads = []
    print(f"Starting {num_threads} threads...")

    # Create and start each thread.
    for i in range(num_threads):
        thread = threading.Thread(target=worker_task, args=(i, shared_data))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete.
    for thread in threads:
        thread.join()

    print("\nAll threads have completed. Gathering results...")

    # Process the results collected from all threads.
    success_count = 0
    failure_count = 0

    while shared_data.results:
        result = shared_data.results.popleft()
        if "successfully" in result:
            print(f"✅ {result}")
            success_count += 1
        else:
            print(f"❌ {result}")
            failure_count += 1

    print("\n--- Summary ---")
    print(f"Total tasks: {num_threads}")
    print(f"Successful tasks: {success_count}")
    print(f"Failed tasks: {failure_count}")


# Main execution block.
if __name__ == "__main__":
    # You can change the number of threads to see how it works with more tasks.
    run_multi_threaded_process(10)
