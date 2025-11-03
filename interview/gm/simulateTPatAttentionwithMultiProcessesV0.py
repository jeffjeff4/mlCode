####assume we have 2 gpus, write function for tensor parallel in attention layer in a transformer layer
####do not use multi process
####this is a runnable version, could be used in interview
####this is NOT using multi processes


import numpy as np


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
    # Stable softmax
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)

    output = np.matmul(weights, v)
    return output, weights


class TensorParallelAttention:
    """
    Manages a multi-head attention layer with tensor parallelism across two
    simulated GPUs in a single process.
    """

    def __init__(self, d_model, num_heads):
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be divisible by 2 for 2-GPU parallelism.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Initialize full weights for comparison purposes
        weights_q = np.random.randn(d_model, d_model)
        weights_k = np.random.randn(d_model, d_model)
        weights_v = np.random.randn(d_model, d_model)
        weights_o = np.random.randn(d_model, d_model)

        # Store full weights for verification in tests
        self.weights = (weights_q, weights_k, weights_v, weights_o)

        # --- Split weights for each simulated GPU ---

        # Column Parallelism for Q, K, V projection
        self.weights_q_gpu0, self.weights_q_gpu1 = np.hsplit(weights_q, 2)
        self.weights_k_gpu0, self.weights_k_gpu1 = np.hsplit(weights_k, 2)
        self.weights_v_gpu0, self.weights_v_gpu1 = np.hsplit(weights_v, 2)

        # Row Parallelism for Output projection
        self.weights_o_gpu0, self.weights_o_gpu1 = np.vsplit(weights_o, 2)

    def forward(self, x):
        """
        Performs the forward pass by simulating computation on two GPUs and
        then combining the results.

        Args:
            x (np.ndarray): Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            np.ndarray: The final output tensor after all-reduce.
        """
        batch_size, seq_len, _ = x.shape
        num_local_heads = self.num_heads // 2

        # --- GPU 0 Computation ---
        q0 = np.dot(x, self.weights_q_gpu0)
        k0 = np.dot(x, self.weights_k_gpu0)
        v0 = np.dot(x, self.weights_v_gpu0)

        q0 = q0.reshape(batch_size, seq_len, num_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        k0 = k0.reshape(batch_size, seq_len, num_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        v0 = v0.reshape(batch_size, seq_len, num_local_heads, self.head_dim).transpose(0, 2, 1, 3)

        attention_output0, _ = scaled_dot_product_attention(q0, k0, v0)
        attention_output0 = attention_output0.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output_partial0 = np.dot(attention_output0, self.weights_o_gpu0)

        # --- GPU 1 Computation ---
        q1 = np.dot(x, self.weights_q_gpu1)
        k1 = np.dot(x, self.weights_k_gpu1)
        v1 = np.dot(x, self.weights_v_gpu1)

        q1 = q1.reshape(batch_size, seq_len, num_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        k1 = k1.reshape(batch_size, seq_len, num_local_heads, self.head_dim).transpose(0, 2, 1, 3)
        v1 = v1.reshape(batch_size, seq_len, num_local_heads, self.head_dim).transpose(0, 2, 1, 3)

        attention_output1, _ = scaled_dot_product_attention(q1, k1, v1)
        attention_output1 = attention_output1.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output_partial1 = np.dot(attention_output1, self.weights_o_gpu1)

        # --- All-Reduce Step ---
        # The partial outputs from each GPU are summed to get the final result.
        final_output = output_partial0 + output_partial1

        return final_output


import unittest
import numpy as np
####from tensor_parallel_attention import TensorParallelAttention, scaled_dot_product_attention


class TestTensorParallelAttention(unittest.TestCase):
    """
    Unit tests for the single-process TensorParallelAttention class.
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

    def test_forward_pass_correctness(self):
        """
        Tests if the forward pass of the parallel attention is equivalent to a sequential version.
        """
        # --- Parallel Attention computation ---
        output_parallel = self.attention_parallel.forward(self.x)

        # --- Sequential (non-parallel) Attention computation for comparison ---
        weights_q_full, weights_k_full, weights_v_full, weights_o_full = self.attention_parallel.weights
        head_dim = self.d_model // self.num_heads

        # 1. Project to full Q, K, V
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
            TensorParallelAttention(d_model=self.d_model, num_heads=7)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
