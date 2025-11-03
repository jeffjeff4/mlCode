####assume we have 2 gpus, write function for tensor parallel in attention layer in a transformer layer
####this is a runnable version, could be used in interview
####this is using multi processes

import numpy as np
import multiprocessing as mp


def scaled_dot_product_attention(q, k, v):
    """
    Computes the scaled dot-product attention.

    Args:
        q (np.ndarray): Queries, shape (batch, heads, seq_len, head_dim).
        k (np.ndarray): Keys, shape (batch, heads, seq_len, head_dim).
        v (np.ndarray): Values, shape (batch, heads, seq_len, head_dim).

    Returns:
        tuple: (output, attention_weights)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    output = np.matmul(weights, v)
    return output, weights


def gpu_process_worker(pipe, rank, num_heads, d_model, weights_qkv_slice, weights_o_slice):
    """
    A worker function that simulates the operations of a single GPU in the attention layer.

    Args:
        pipe (mp.Pipe): Communication channel to the main process.
        rank (int): The rank of this GPU process (0 or 1).
        num_heads (int): Total number of attention heads (in the full model).
        d_model (int): The dimension of the model.
        weights_qkv_slice (np.ndarray): The vertical slice of the combined Q, K, V weight matrix.
        weights_o_slice (np.ndarray): The horizontal slice of the output projection weight matrix.
    """
    cache = {}
    head_dim = d_model // num_heads
    # Each worker gets half the heads
    num_local_heads = num_heads // 2

    while True:
        command, data = pipe.recv()

        if command == "forward":
            x = data
            batch_size, seq_len, _ = x.shape

            # --- Local Forward Pass ---
            # 1. Project to local Q, K, V (Column Parallelism)
            qkv_local = np.dot(x, weights_qkv_slice)
            q_local, k_local, v_local = np.split(qkv_local, 3, axis=-1)

            # Reshape for multi-head attention
            q_local = q_local.reshape(batch_size, seq_len, num_local_heads, head_dim).transpose(0, 2, 1, 3)
            k_local = k_local.reshape(batch_size, seq_len, num_local_heads, head_dim).transpose(0, 2, 1, 3)
            v_local = v_local.reshape(batch_size, seq_len, num_local_heads, head_dim).transpose(0, 2, 1, 3)

            # 2. Compute local attention
            local_attention_output, _ = scaled_dot_product_attention(q_local, k_local, v_local)
            local_attention_output = local_attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

            # 3. Project local output (Row Parallelism) - gives a partial output
            output_partial = np.dot(local_attention_output, weights_o_slice)

            # Cache values for backward pass
            cache.update({'x': x, 'qkv_local': qkv_local, 'local_attention_output': local_attention_output})

            # Send partial output to main process for all-reduce
            pipe.send(output_partial)

        elif command == "shutdown":
            break


class TensorParallelAttention:
    """
    Manages a multi-head attention layer with tensor parallelism across two processes.
    """

    def __init__(self, d_model, num_heads):
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be divisible by 2 for 2-GPU parallelism.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Initialize full weights on the 'CPU' (main process)
        weights_q = np.random.randn(d_model, d_model)
        weights_k = np.random.randn(d_model, d_model)
        weights_v = np.random.randn(d_model, d_model)
        weights_o = np.random.randn(d_model, d_model)

        # Store for verification purposes
        self.weights = (weights_q, weights_k, weights_v, weights_o)

        # Split Q, K, V weights for column parallelism
        weights_q_gpu0, weights_q_gpu1 = np.hsplit(weights_q, 2)
        weights_k_gpu0, weights_k_gpu1 = np.hsplit(weights_k, 2)
        weights_v_gpu0, weights_v_gpu1 = np.hsplit(weights_v, 2)

        # Concatenate the slices for each GPU. Now each GPU has a clean set of Q, K, V weights for its heads.
        weights_qkv_gpu0 = np.concatenate([weights_q_gpu0, weights_k_gpu0, weights_v_gpu0], axis=-1)
        weights_qkv_gpu1 = np.concatenate([weights_q_gpu1, weights_k_gpu1, weights_v_gpu1], axis=-1)

        # Row parallelism for Output projection
        weights_o_gpu0, weights_o_gpu1 = np.vsplit(weights_o, 2)

        # Create communication pipes and processes
        self.pipe0, child_pipe0 = mp.Pipe()
        self.pipe1, child_pipe1 = mp.Pipe()

        self.proc0 = mp.Process(target=gpu_process_worker,
                                args=(child_pipe0, 0, num_heads, d_model, weights_qkv_gpu0, weights_o_gpu0))
        self.proc1 = mp.Process(target=gpu_process_worker,
                                args=(child_pipe1, 1, num_heads, d_model, weights_qkv_gpu1, weights_o_gpu1))

        self.proc0.start()
        self.proc1.start()

    def forward(self, x):
        """Performs the forward pass by sending data to worker processes."""
        self.pipe0.send(("forward", x))
        self.pipe1.send(("forward", x))

        output_gpu0 = self.pipe0.recv()
        output_gpu1 = self.pipe1.recv()

        # Perform the 'all-reduce' (summation) on the main process
        return output_gpu0 + output_gpu1

    def shutdown(self):
        """Sends a shutdown signal to the worker processes and joins them."""
        self.pipe0.send(("shutdown", None))
        self.pipe1.send(("shutdown", None))
        self.proc0.join()
        self.proc1.join()


import unittest
import numpy as np
import multiprocessing as mp
####from tensor_parallel_attention_mp import TensorParallelAttention, scaled_dot_product_attention


class TestTensorParallelAttention(unittest.TestCase):
    """
    Unit tests for the multiprocessing-based TensorParallelAttention class.
    """

    def setUp(self):
        """Set up the test environment before each test."""
        self.batch_size = 4
        self.seq_len = 10
        self.d_model = 128
        self.num_heads = 8

        # Ensure reproducibility for tests
        np.random.seed(42)

        self.attention_parallel = TensorParallelAttention(self.d_model, self.num_heads)
        self.x = np.random.randn(self.batch_size, self.seq_len, self.d_model)

    def tearDown(self):
        """Clean up the worker processes after each test."""
        self.attention_parallel.shutdown()

    def test_forward_pass_correctness(self):
        """
        Tests if the forward pass of the parallel attention is equivalent to a sequential version.
        """
        # --- Parallel Attention computation ---
        output_parallel = self.attention_parallel.forward(self.x)

        # --- Sequential (non-parallel) Attention computation for comparison ---
        # CORRECTED: Unpack all four weight matrices from the weights tuple.
        weights_q_full, weights_k_full, weights_v_full, weights_o_full = self.attention_parallel.weights
        head_dim = self.d_model // self.num_heads

        # 1. Project to full Q, K, V using their individual weight matrices
        q_full = np.dot(self.x, weights_q_full)
        k_full = np.dot(self.x, weights_k_full)
        v_full = np.dot(self.x, weights_v_full)

        # 2. Reshape for multi-head attention
        q_full = q_full.reshape(self.batch_size, self.seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k_full = k_full.reshape(self.batch_size, self.seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v_full = v_full.reshape(self.batch_size, self.seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # 3. Compute scaled dot-product attention
        attention_output_full, _ = scaled_dot_product_attention(q_full, k_full, v_full)

        # 4. Concatenate heads and project output
        attention_output_full = attention_output_full.transpose(0, 2, 1, 3).reshape(self.batch_size, self.seq_len,
                                                                                    self.d_model)
        output_sequential = np.dot(attention_output_full, weights_o_full)

        # --- Comparison ---
        self.assertTrue(np.allclose(output_sequential, output_parallel, atol=1e-7),
                        "Forward pass output does not match the sequential version.")

    def test_init_with_indivisible_heads(self):
        """
        Tests that the initializer raises a ValueError if num_heads is not divisible by 2.
        """
        with self.assertRaises(ValueError):
            # This instance will fail during initialization, so no shutdown is needed.
            TensorParallelAttention(d_model=self.d_model, num_heads=7)


if __name__ == '__main__':
    # Using spawn is recommended for compatibility across platforms (macOS, Windows)
    mp.set_start_method('spawn', force=True)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

