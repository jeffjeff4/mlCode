####assume we have 2 gpus, write function for tensor parallel in mlp later in a transformer layer
####this is a runnable version, could be used in interview
####this is NOT using multi processes

import numpy as np
import multiprocessing as mp


def gpu_process_worker(rank, pipe, weights1_slice, weights2_slice):
    """
    A worker function that simulates the operations on a single GPU.

    This function runs in a separate process. It waits for data and commands
    from the main process via a pipe, performs its share of the computation,
    communicates with other workers (via the main process in this simulation),
    and sends results back.

    Args:
        rank (int): The rank of the GPU (0 or 1).
        pipe (mp.Pipe): The communication channel to the main process.
        weights1_slice (np.ndarray): The vertical slice of the first weight matrix.
        weights2_slice (np.ndarray): The horizontal slice of the second weight matrix.
    """
    grad_weights1 = np.zeros_like(weights1_slice)
    grad_weights2 = np.zeros_like(weights2_slice)
    cache = {}

    while True:
        command, data = pipe.recv()

        if command == "forward":
            x = data
            # --- Local Forward Pass ---
            # First linear layer (Column Parallelism)
            z1 = np.dot(x, weights1_slice)
            a1 = np.maximum(0, z1)

            # Second linear layer (Row Parallelism) - Partial output
            output_partial = np.dot(a1, weights2_slice)

            # Cache intermediates for backward pass
            cache['x'] = x
            cache['z1'] = z1
            cache['a1'] = a1

            # Send partial output to main process for all-reduce
            pipe.send(output_partial)

        elif command == "backward":
            grad_output = data
            # --- Local Backward Pass ---
            x, z1, a1 = cache['x'], cache['z1'], cache['a1']

            # Gradient for the second layer weights
            grad_weights2 = np.dot(a1.T, grad_output)
            # Gradient flowing back to the activation output
            grad_a1 = np.dot(grad_output, weights2_slice.T)
            # Gradient through the ReLU activation
            grad_z1 = grad_a1 * (z1 > 0)
            # Gradient for the first layer weights
            grad_weights1 = np.dot(x.T, grad_z1)

            # In a real system, gradients would be updated here.
            # For this simulation, we'll send them back for verification.
            pipe.send((grad_weights1, grad_weights2))

        elif command == "shutdown":
            break


class TensorParallelMLP:
    """
    Manages a two-layer MLP with tensor parallelism across two processes.

    This class initializes the model, spawns two worker processes to simulate
    two GPUs, and orchestrates the forward and backward passes by communicating
    with the workers.
    """

    def __init__(self, input_size, hidden_size, output_size):
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be an even number for 2-GPU parallelism.")

        # Initialize full weights on the 'CPU' (main process)
        # In a real scenario, each GPU would initialize its slice directly.
        weights1 = np.random.randn(input_size, hidden_size)
        weights2 = np.random.randn(hidden_size, output_size)

        # Split weights for each GPU process
        weights1_gpu0, weights1_gpu1 = np.hsplit(weights1, 2)
        weights2_gpu0, weights2_gpu1 = np.vsplit(weights2, 2)

        self.weights = (weights1, weights2)  # Store for verification

        # Create communication pipes and processes
        self.pipe0, child_pipe0 = mp.Pipe()
        self.pipe1, child_pipe1 = mp.Pipe()

        self.proc0 = mp.Process(target=gpu_process_worker, args=(0, child_pipe0, weights1_gpu0, weights2_gpu0))
        self.proc1 = mp.Process(target=gpu_process_worker, args=(1, child_pipe1, weights1_gpu1, weights2_gpu1))

        self.proc0.start()
        self.proc1.start()

    def forward(self, x):
        """
        Performs the forward pass by sending data to worker processes.
        """
        # Send input data to both processes
        self.pipe0.send(("forward", x))
        self.pipe1.send(("forward", x))

        # Receive partial results
        output_gpu0 = self.pipe0.recv()
        output_gpu1 = self.pipe1.recv()

        # Perform the 'all-reduce' (summation) on the main process
        return output_gpu0 + output_gpu1

    def backward(self, grad_output):
        """
        Performs the backward pass by sending gradients to worker processes.
        """
        self.pipe0.send(("backward", grad_output))
        self.pipe1.send(("backward", grad_output))

        # Receive computed gradients from each process for verification
        grad_w1_gpu0, grad_w2_gpu0 = self.pipe0.recv()
        grad_w1_gpu1, grad_w2_gpu1 = self.pipe1.recv()

        return grad_w1_gpu0, grad_w1_gpu1, grad_w2_gpu0, grad_w2_gpu1

    def shutdown(self):
        """
        Sends a shutdown signal to the worker processes and joins them.
        """
        self.pipe0.send(("shutdown", None))
        self.pipe1.send(("shutdown", None))
        self.proc0.join()
        self.proc1.join()


import unittest
import numpy as np
#from tensor_parallel_mlp_mp import TensorParallelMLP


class TestTensorParallelMLP(unittest.TestCase):
    """
    Unit tests for the multiprocessing-based TensorParallelMLP class.
    """

    def setUp(self):
        """Set up the test environment before each test."""
        self.batch_size = 16
        self.input_size = 128
        self.hidden_size = 256
        self.output_size = 64
        # Ensure reproducibility
        np.random.seed(42)
        self.mlp = TensorParallelMLP(self.input_size, self.hidden_size, self.output_size)
        self.x = np.random.randn(self.batch_size, self.input_size)
        self.grad_output = np.random.randn(self.batch_size, self.output_size)

    def tearDown(self):
        """Clean up the worker processes after each test."""
        self.mlp.shutdown()

    def test_forward_pass_correctness(self):
        """
        Tests if the forward pass of the parallel MLP is equivalent to a sequential MLP.
        """
        # --- Parallel MLP computation ---
        output_parallel = self.mlp.forward(self.x)

        # --- Sequential (non-parallel) MLP computation for comparison ---
        weights1_full, weights2_full = self.mlp.weights
        z1_full = np.dot(self.x, weights1_full)
        a1_full = np.maximum(0, z1_full)
        output_sequential = np.dot(a1_full, weights2_full)

        # --- Comparison ---
        self.assertTrue(np.allclose(output_sequential, output_parallel, atol=1e-9),
                        "Forward pass output does not match the sequential version.")

    def test_backward_pass_correctness(self):
        """
        Tests if the backward pass of the parallel MLP is equivalent to a sequential MLP.
        """
        # --- Parallel backward pass ---
        # A forward pass is needed to set the cache in the workers
        self.mlp.forward(self.x)
        p_grad_w1_0, p_grad_w1_1, p_grad_w2_0, p_grad_w2_1 = self.mlp.backward(self.grad_output)

        # --- Sequential backward pass for comparison ---
        weights1_full, weights2_full = self.mlp.weights
        # Sequential forward pass to get intermediate values
        z1_full = np.dot(self.x, weights1_full)
        a1_full = np.maximum(0, z1_full)

        # Sequential backward pass
        grad_weights2_full = np.dot(a1_full.T, self.grad_output)
        grad_a1_full = np.dot(self.grad_output, weights2_full.T)
        grad_z1_full = grad_a1_full * (z1_full > 0)
        grad_weights1_full = np.dot(self.x.T, grad_z1_full)

        # Split the sequential gradients to compare with parallel gradients
        s_grad_w1_0, s_grad_w1_1 = np.hsplit(grad_weights1_full, 2)
        s_grad_w2_0, s_grad_w2_1 = np.vsplit(grad_weights2_full, 2)

        # --- Comparison ---
        self.assertTrue(np.allclose(s_grad_w1_0, p_grad_w1_0, atol=1e-9))
        self.assertTrue(np.allclose(s_grad_w1_1, p_grad_w1_1, atol=1e-9))
        self.assertTrue(np.allclose(s_grad_w2_0, p_grad_w2_0, atol=1e-9))
        self.assertTrue(np.allclose(s_grad_w2_1, p_grad_w2_1, atol=1e-9))

    def test_init_with_odd_hidden_size(self):
        """
        Tests that the initializer raises a ValueError for an odd hidden_size.

        Note: This test needs to be handled carefully as the error occurs in the
        main process before child processes are created.
        """
        with self.assertRaises(ValueError):
            # Temporarily create an instance that will fail, then shut it down if it somehow passes
            temp_mlp = TensorParallelMLP(self.input_size, self.hidden_size + 1, self.output_size)
            temp_mlp.shutdown()


if __name__ == '__main__':
    # Using spawn is recommended for compatibility across platforms (macOS, Windows)
    mp.set_start_method('spawn', force=True)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
