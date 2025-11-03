####assume we have 2 gpus, write function for tensor parallel in mlp later in a transformer layer
####this is a runnable version, could be used in interview
####this is NOT using multi processes

import numpy as np

class TensorParallelMLP:
    """
    Implements a simple two-layer MLP with tensor parallelism across two GPUs.

    This class simulates the behavior of splitting the weights of an MLP
    across two devices and performing the forward and backward passes in a
    distributed manner. This is a foundational concept in model parallelism,
    allowing for the training of models that are too large to fit on a single GPU.

    The parallelism strategy used here is column parallelism for the first
    linear layer and row parallelism for the second linear layer.

    Attributes:
        input_size (int): The size of the input feature dimension.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output feature dimension.
        weights1_gpu0 (np.ndarray): Weight matrix for the first linear layer on GPU 0.
        weights1_gpu1 (np.ndarray): Weight matrix for the first linear layer on GPU 1.
        weights2_gpu0 (np.ndarray): Weight matrix for the second linear layer on GPU 0.
        weights2_gpu1 (np.ndarray): Weight matrix for the second linear layer on GPU 1.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the TensorParallelMLP and its distributed weights.

        Args:
            input_size (int): The dimension of the input to the MLP.
            hidden_size (int): The dimension of the hidden layer. Must be even.
            output_size (int): The dimension of the output of the MLP.
        """
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be an even number for 2-GPU parallelism.")

        # For the first linear layer (input_size -> hidden_size), we apply column parallelism.
        # This means we split the weight matrix along the output dimension (hidden_size).
        self.weights1_gpu0 = np.random.randn(input_size, hidden_size // 2)
        self.weights1_gpu1 = np.random.randn(input_size, hidden_size // 2)

        # For the second linear layer (hidden_size -> output_size), we apply row parallelism.
        # This means we split the weight matrix along the input dimension (hidden_size).
        self.weights2_gpu0 = np.random.randn(hidden_size // 2, output_size)
        self.weights2_gpu1 = np.random.randn(hidden_size // 2, output_size)

        # Gradients are initialized to zero
        self.grad_weights1_gpu0 = np.zeros_like(self.weights1_gpu0)
        self.grad_weights1_gpu1 = np.zeros_like(self.weights1_gpu1)
        self.grad_weights2_gpu0 = np.zeros_like(self.weights2_gpu0)
        self.grad_weights2_gpu1 = np.zeros_like(self.weights2_gpu1)

    def forward(self, x):
        """
        Performs the forward pass of the tensor-parallel MLP.

        Args:
            x (np.ndarray): The input tensor of shape (batch_size, input_size).

        Returns:
            np.ndarray: The output tensor of shape (batch_size, output_size).
            tuple: A cache of intermediate values needed for the backward pass.
        """
        # --- GPU 0 Operations ---
        # Linear transformation on GPU 0
        z1_gpu0 = np.dot(x, self.weights1_gpu0)
        # Activation function (ReLU) on GPU 0
        a1_gpu0 = np.maximum(0, z1_gpu0)

        # --- GPU 1 Operations ---
        # Linear transformation on GPU 1
        z1_gpu1 = np.dot(x, self.weights1_gpu1)
        # Activation function (ReLU) on GPU 1
        a1_gpu1 = np.maximum(0, z1_gpu1)

        # --- Communication and second layer ---
        # No 'all-gather' needed here. Each GPU computes its part and the results
        # are summed after the second matrix multiplication.

        # --- GPU 0 Operations ---
        # Second linear transformation on GPU 0
        output_gpu0 = np.dot(a1_gpu0, self.weights2_gpu0)

        # --- GPU 1 Operations ---
        # Second linear transformation on GPU 1
        output_gpu1 = np.dot(a1_gpu1, self.weights2_gpu1)

        # --- Communication: All-Reduce ---
        # The partial results from each GPU are summed to get the final output.
        # This is the all-reduce step.
        output = output_gpu0 + output_gpu1

        cache = (x, z1_gpu0, a1_gpu0, z1_gpu1, a1_gpu1)
        return output, cache

    def backward(self, grad_output, cache):
        """
        Performs the backward pass of the tensor-parallel MLP.

        Args:
            grad_output (np.ndarray): The gradient of the loss with respect to the output.
            cache (tuple): The cache of intermediate values from the forward pass.
        """
        x, z1_gpu0, a1_gpu0, z1_gpu1, a1_gpu1 = cache

        # The initial gradient `grad_output` is available on both GPUs.

        # --- GPU 0 Operations ---
        # Gradient for the second layer weights on GPU 0
        self.grad_weights2_gpu0 = np.dot(a1_gpu0.T, grad_output)
        # Gradient flowing back to the activation output on GPU 0
        grad_a1_gpu0 = np.dot(grad_output, self.weights2_gpu0.T)
        # Gradient through the ReLU activation on GPU 0
        grad_z1_gpu0 = grad_a1_gpu0 * (z1_gpu0 > 0)
        # Gradient for the first layer weights on GPU 0
        self.grad_weights1_gpu0 = np.dot(x.T, grad_z1_gpu0)

        # --- GPU 1 Operations ---
        # Gradient for the second layer weights on GPU 1
        self.grad_weights2_gpu1 = np.dot(a1_gpu1.T, grad_output)
        # Gradient flowing back to the activation output on GPU 1
        grad_a1_gpu1 = np.dot(grad_output, self.weights2_gpu1.T)
        # Gradient through the ReLU activation on GPU 1
        grad_z1_gpu1 = grad_a1_gpu1 * (z1_gpu1 > 0)
        # Gradient for the first layer weights on GPU 1
        self.grad_weights1_gpu1 = np.dot(x.T, grad_z1_gpu1)

        # Gradients for weights are now computed and stored.
        # In a real scenario, an optimizer step would follow.

    def get_gradients(self):
        """
        Returns the computed gradients for all weight matrices.

        Returns:
            dict: A dictionary containing the gradients for each weight matrix.
        """
        return {
            'weights1_gpu0': self.grad_weights1_gpu0,
            'weights1_gpu1': self.grad_weights1_gpu1,
            'weights2_gpu0': self.grad_weights2_gpu0,
            'weights2_gpu1': self.grad_weights2_gpu1
        }


####------------------------------------------------------------------------------------

import unittest
import numpy as np
####from tensor_parallel_mlp import TensorParallelMLP

class TestTensorParallelMLP(unittest.TestCase):
    """
    Unit tests for the TensorParallelMLP class.
    """

    def setUp(self):
        """
        Set up the test environment. This method is called before each test.
        """
        self.batch_size = 10
        self.input_size = 32
        self.hidden_size = 64
        self.output_size = 16
        # Ensure reproducibility for tests
        np.random.seed(42)
        self.mlp = TensorParallelMLP(self.input_size, self.hidden_size, self.output_size)
        self.x = np.random.randn(self.batch_size, self.input_size)
        self.grad_output = np.random.randn(self.batch_size, self.output_size)

    def test_forward_pass_correctness(self):
        """
        Tests if the forward pass of the parallel MLP is equivalent to a sequential MLP.
        """
        # --- Sequential (non-parallel) MLP computation for comparison ---
        # Reconstruct the full weight matrices from the distributed parts
        weights1_full = np.concatenate((self.mlp.weights1_gpu0, self.mlp.weights1_gpu1), axis=1)
        weights2_full = np.concatenate((self.mlp.weights2_gpu0, self.mlp.weights2_gpu1), axis=0)

        # Perform the forward pass sequentially
        z1_full = np.dot(self.x, weights1_full)
        a1_full = np.maximum(0, z1_full)
        output_sequential = np.dot(a1_full, weights2_full)

        # --- Parallel MLP computation ---
        output_parallel, _ = self.mlp.forward(self.x)

        # --- Comparison ---
        self.assertTrue(np.allclose(output_sequential, output_parallel, atol=1e-9),
                        "Forward pass output does not match the sequential version.")

    def test_backward_pass_correctness(self):
        """
        Tests if the backward pass of the parallel MLP is equivalent to a sequential MLP.
        """
        # --- Parallel backward pass ---
        _, cache = self.mlp.forward(self.x)
        self.mlp.backward(self.grad_output, cache)
        parallel_grads = self.mlp.get_gradients()

        # --- Sequential backward pass for comparison ---
        # Reconstruct the full weight matrices
        weights1_full = np.concatenate((self.mlp.weights1_gpu0, self.mlp.weights1_gpu1), axis=1)
        weights2_full = np.concatenate((self.mlp.weights2_gpu0, self.mlp.weights2_gpu1), axis=0)

        # Sequential forward pass to get intermediate values
        z1_full = np.dot(self.x, weights1_full)
        a1_full = np.maximum(0, z1_full)

        # Sequential backward pass
        grad_weights2_full = np.dot(a1_full.T, self.grad_output)
        grad_a1_full = np.dot(self.grad_output, weights2_full.T)
        grad_z1_full = grad_a1_full * (z1_full > 0)
        grad_weights1_full = np.dot(self.x.T, grad_z1_full)

        # Split the sequential gradients to compare with parallel gradients
        grad_weights1_gpu0_seq = grad_weights1_full[:, :self.hidden_size // 2]
        grad_weights1_gpu1_seq = grad_weights1_full[:, self.hidden_size // 2:]
        grad_weights2_gpu0_seq = grad_weights2_full[:self.hidden_size // 2, :]
        grad_weights2_gpu1_seq = grad_weights2_full[self.hidden_size // 2:, :]

        # --- Comparison ---
        self.assertTrue(np.allclose(grad_weights1_gpu0_seq, parallel_grads['weights1_gpu0'], atol=1e-9))
        self.assertTrue(np.allclose(grad_weights1_gpu1_seq, parallel_grads['weights1_gpu1'], atol=1e-9))
        self.assertTrue(np.allclose(grad_weights2_gpu0_seq, parallel_grads['weights2_gpu0'], atol=1e-9))
        self.assertTrue(np.allclose(grad_weights2_gpu1_seq, parallel_grads['weights2_gpu1'], atol=1e-9))

    def test_edge_case_batch_size_one(self):
        """
        Tests the MLP with a batch size of 1, a common edge case.
        """
        x_single = np.random.randn(1, self.input_size)
        grad_output_single = np.random.randn(1, self.output_size)

        # --- Sequential calculation ---
        weights1_full = np.concatenate((self.mlp.weights1_gpu0, self.mlp.weights1_gpu1), axis=1)
        weights2_full = np.concatenate((self.mlp.weights2_gpu0, self.mlp.weights2_gpu1), axis=0)
        z1_full = np.dot(x_single, weights1_full)
        a1_full = np.maximum(0, z1_full)
        output_sequential = np.dot(a1_full, weights2_full)

        # --- Parallel calculation ---
        output_parallel, cache = self.mlp.forward(x_single)
        self.mlp.backward(grad_output_single, cache)

        self.assertTrue(np.allclose(output_sequential, output_parallel, atol=1e-9),
                        "Forward pass failed for batch size of 1.")
        # We can also check if gradients are computed without errors
        self.assertIsNotNone(self.mlp.get_gradients())

    def test_init_with_odd_hidden_size(self):
        """
        Tests that the initializer raises a ValueError for an odd hidden_size.
        """
        with self.assertRaises(ValueError):
            TensorParallelMLP(self.input_size, self.hidden_size + 1, self.output_size)


if __name__ == '__main__':
    # This allows the test to be run from the command line
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


####------------------------------------------------------------------------------------

####Analysis of the Tensor-Parallel MLP Implementation
####
####This implementation demonstrates a fundamental tensor parallelism strategy for a two-layer MLP across two GPUs. The core idea is to split the model's weights and computation, allowing for the training of larger models than would fit on a single device.
####
####Parallelism Strategy
####
####First Linear Layer (Column Parallelism): The weight matrix W1 of size (input_size, hidden_size) is split into two matrices, W1_gpu0 and W1_gpu1, each of size (input_size, hidden_size / 2). The input x is broadcast to both GPUs. Each GPU computes its part of the hidden activation independently.
####
####z1_gpu0 = x @ W1_gpu0
####
####z1_gpu1 = x @ W1_gpu1
####
####Second Linear Layer (Row Parallelism): The weight matrix W2 of size (hidden_size, output_size) is split into two matrices, W2_gpu0 of size (hidden_size / 2, output_size) and W2_gpu1 of size (hidden_size / 2, output_size). Each GPU uses its local activation from the first layer to compute a partial output.
####
####output_gpu0 = a1_gpu0 @ W2_gpu0
####
####output_gpu1 = a1_gpu1 @ W2_gpu1
####
####Communication (All-Reduce): The final step in the forward pass is to sum the partial outputs from each GPU. This is an all-reduce operation, where the results from all devices are aggregated and the final sum is distributed back to all devices.
####
####output = output_gpu0 + output_gpu1
####
####The backward pass mirrors this logic, applying the chain rule to the distributed computations.
####
####How to Run
####
####To execute the code and run the tests, you would typically save the files and run the test script from your terminal:
####
####python -m unittest test_tensor_parallel_mlp.py
####
####
####This will verify that the forward and backward passes of the tensor-parallel MLP produce the same results as a standard sequential MLP, ensuring the implementation is correct.
####
####Complexity Analysis
####
####Let:
####
####b be the batch size
####
####i be the input size
####
####h be the hidden size
####
####o be the output size
####
####p be the number of GPUs (in this case, p=2)
####
####Time Complexity
####
####Forward Pass:
####
####First Linear Layer: O(b * i * h / p) on each GPU.
####
####Activation: O(b * h / p) on each GPU.
####
####Second Linear Layer: O(b * (h / p) * o) on each GPU.
####
####All-Reduce: The complexity of communication depends on the network topology and algorithm, but a common model is O(log(p) * (b * o)).
####
####The overall forward time complexity is dominated by computation and is approximately O(b * i * h / p + b * h * o / p).
####
####Backward Pass: The backward pass has a similar computational complexity to the forward pass, as the matrix multiplications involved are of similar dimensions.
####
####Gradient of W2: O(b * (h / p) * o)
####
####Gradient back to a1: O(b * o * (h / p))
####
####Gradient through activation: O(b * h / p)
####
####Gradient of W1: O(b * i * (h / p))
####
####The overall backward time complexity is approximately O(b * o * h / p + b * i * h / p).
####
####Space Complexity
####
####Weights: Each GPU stores a fraction of the total weights.
####
####W1: O(i * h / p) on each GPU.
####
####W2: O((h / p) * o) on each GPU.
####
####Total weights per GPU: O((i * h + h * o) / p)
####
####Activations: During the forward pass, intermediate activations are stored for the backward pass.
####
####x: O(b * i) (input is broadcast)
####
####a1: O(b * h / p) on each GPU.
####
####grad_output: O(b * o) (broadcast)
####
####The space complexity for storing model parameters is effectively reduced by a factor of p. The space required for activations is also reduced for distributed layers, which is a key benefit for training with large batch sizes.








####------------------------------------------------------------------------------------

import unittest
import numpy as np
#from tensor_parallel_mlp import TensorParallelMLP

class TestTensorParallelMLP(unittest.TestCase):
    """
    Unit tests for the TensorParallelMLP class.
    """

    def setUp(self):
        """
        Set up the test environment. This method is called before each test.
        """
        self.batch_size = 10
        self.input_size = 32
        self.hidden_size = 64
        self.output_size = 16
        # Ensure reproducibility for tests
        np.random.seed(42)
        self.mlp = TensorParallelMLP(self.input_size, self.hidden_size, self.output_size)
        self.x = np.random.randn(self.batch_size, self.input_size)
        self.grad_output = np.random.randn(self.batch_size, self.output_size)

    def test_forward_pass_correctness(self):
        """
        Tests if the forward pass of the parallel MLP is equivalent to a sequential MLP.
        """
        # --- Sequential (non-parallel) MLP computation for comparison ---
        # Reconstruct the full weight matrices from the distributed parts
        weights1_full = np.concatenate((self.mlp.weights1_gpu0, self.mlp.weights1_gpu1), axis=1)
        weights2_full = np.concatenate((self.mlp.weights2_gpu0, self.mlp.weights2_gpu1), axis=0)

        # Perform the forward pass sequentially
        z1_full = np.dot(self.x, weights1_full)
        a1_full = np.maximum(0, z1_full)
        output_sequential = np.dot(a1_full, weights2_full)

        # --- Parallel MLP computation ---
        output_parallel, _ = self.mlp.forward(self.x)

        # --- Comparison ---
        self.assertTrue(np.allclose(output_sequential, output_parallel, atol=1e-9),
                        "Forward pass output does not match the sequential version.")

    def test_backward_pass_correctness(self):
        """
        Tests if the backward pass of the parallel MLP is equivalent to a sequential MLP.
        """
        # --- Parallel backward pass ---
        _, cache = self.mlp.forward(self.x)
        self.mlp.backward(self.grad_output, cache)
        parallel_grads = self.mlp.get_gradients()

        # --- Sequential backward pass for comparison ---
        # Reconstruct the full weight matrices
        weights1_full = np.concatenate((self.mlp.weights1_gpu0, self.mlp.weights1_gpu1), axis=1)
        weights2_full = np.concatenate((self.mlp.weights2_gpu0, self.mlp.weights2_gpu1), axis=0)

        # Sequential forward pass to get intermediate values
        z1_full = np.dot(self.x, weights1_full)
        a1_full = np.maximum(0, z1_full)

        # Sequential backward pass
        grad_weights2_full = np.dot(a1_full.T, self.grad_output)
        grad_a1_full = np.dot(self.grad_output, weights2_full.T)
        grad_z1_full = grad_a1_full * (z1_full > 0)
        grad_weights1_full = np.dot(self.x.T, grad_z1_full)

        # Split the sequential gradients to compare with parallel gradients
        grad_weights1_gpu0_seq = grad_weights1_full[:, :self.hidden_size // 2]
        grad_weights1_gpu1_seq = grad_weights1_full[:, self.hidden_size // 2:]
        grad_weights2_gpu0_seq = grad_weights2_full[:self.hidden_size // 2, :]
        grad_weights2_gpu1_seq = grad_weights2_full[self.hidden_size // 2:, :]

        # --- Comparison ---
        self.assertTrue(np.allclose(grad_weights1_gpu0_seq, parallel_grads['weights1_gpu0'], atol=1e-9))
        self.assertTrue(np.allclose(grad_weights1_gpu1_seq, parallel_grads['weights1_gpu1'], atol=1e-9))
        self.assertTrue(np.allclose(grad_weights2_gpu0_seq, parallel_grads['weights2_gpu0'], atol=1e-9))
        self.assertTrue(np.allclose(grad_weights2_gpu1_seq, parallel_grads['weights2_gpu1'], atol=1e-9))

    def test_edge_case_batch_size_one(self):
        """
        Tests the MLP with a batch size of 1, a common edge case.
        """
        x_single = np.random.randn(1, self.input_size)
        grad_output_single = np.random.randn(1, self.output_size)

        # --- Sequential calculation ---
        weights1_full = np.concatenate((self.mlp.weights1_gpu0, self.mlp.weights1_gpu1), axis=1)
        weights2_full = np.concatenate((self.mlp.weights2_gpu0, self.mlp.weights2_gpu1), axis=0)
        z1_full = np.dot(x_single, weights1_full)
        a1_full = np.maximum(0, z1_full)
        output_sequential = np.dot(a1_full, weights2_full)

        # --- Parallel calculation ---
        output_parallel, cache = self.mlp.forward(x_single)
        self.mlp.backward(grad_output_single, cache)

        self.assertTrue(np.allclose(output_sequential, output_parallel, atol=1e-9),
                        "Forward pass failed for batch size of 1.")
        # We can also check if gradients are computed without errors
        self.assertIsNotNone(self.mlp.get_gradients())

    def test_init_with_odd_hidden_size(self):
        """
        Tests that the initializer raises a ValueError for an odd hidden_size.
        """
        with self.assertRaises(ValueError):
            TensorParallelMLP(self.input_size, self.hidden_size + 1, self.output_size)


if __name__ == '__main__':
    # This allows the test to be run from the command line
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

