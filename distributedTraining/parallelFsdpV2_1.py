####not runnble, could not be used in an interview

import numpy as np
import multiprocessing as mp
from multiprocessing import Queue

# Simple linear layer model for demonstration
class SimpleLinearModel:
    def __init__(self, input_dim, output_dim):
        self.weight = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)

    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, grad_output, weight):
        """
        Backward pass using the provided weight matrix.
        Args:
            x: Input data, shape (batch_size, input_dim)
            grad_output: Gradient of loss w.r.t. output, shape (batch_size, output_dim)
            weight: Weight matrix, shape (input_dim, output_dim)
        Returns:
            grad_input: Gradient w.r.t. input, shape (batch_size, input_dim)
            grad_weight: Gradient w.r.t. weight, shape (input_dim, output_dim)
            grad_bias: Gradient w.r.t. bias, shape (output_dim)
        """
        grad_weight = np.dot(x.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, weight.T)
        # Ensure grad_bias is 1D
        if grad_bias.ndim == 0:
            grad_bias = np.array([grad_bias])
        return grad_input, grad_weight, grad_bias

    def update(self, grad_weight, grad_bias, lr=0.01):
        if grad_weight.shape != self.weight.shape:
            raise ValueError(f"grad_weight shape {grad_weight.shape} does not match self.weight shape {self.weight.shape}")
        if grad_bias.shape != self.bias.shape:
            raise ValueError(f"grad_bias shape {grad_bias.shape} does not match self.bias shape {self.bias.shape}")
        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias

# FSDP Helper Functions
def all_gather(param, rank, world_size, queues, is_gradient=False):
    # All-gather: Collect parameter or gradient shards from all ranks
    full_params = [None] * world_size
    # Send my shard to all other ranks
    param = np.asarray(param)  # Ensure param is a NumPy array
    for j in range(world_size):
        if j != rank:
            queues[j].put((rank, param))  # Send (rank, param) tuple
    # Receive shards from all other ranks
    for i in range(world_size):
        if i != rank:
            try:
                sender_rank, shard = queues[rank].get(timeout=5.0)
                full_params[sender_rank] = np.asarray(shard)
            except mp.queues.Empty:
                raise RuntimeError(f"Rank {rank} failed to receive shard from rank {i}")
    # Set my own shard
    full_params[rank] = param
    # Check for None values
    if any(p is None for p in full_params):
        raise RuntimeError(f"Rank {rank}: full_params contains None: {full_params}")
    # Validate shapes
    param_shapes = [p.shape for p in full_params]
    if len(set(param_shapes)) > 1:
        raise ValueError(f"Rank {rank}: Inconsistent shapes in full_params: {param_shapes}")
    # Concatenate or stack based on dimensionality and context
    if full_params[0].ndim == 2:  # Weight (2D)
        return np.concatenate(full_params, axis=-1)  # (world_size, input_dim, shard_size) -> (input_dim, output_dim)
    elif full_params[0].ndim == 1:  # Bias or gradient (1D)
        if is_gradient:
            return np.stack(full_params, axis=0)  # (world_size, shard_size)
        else:
            return np.concatenate(full_params, axis=0)  # (output_dim,)
    else:
        raise ValueError(f"Rank {rank}: Unexpected param dimension {full_params[0].ndim}")

def reduce_scatter(grad, rank, world_size, queues, expected_ndim, grad_type="unknown"):
    # Reduce-scatter: Sum gradients and scatter shards
    grad = np.asarray(grad)  # Ensure grad is a NumPy array
    if grad.ndim != expected_ndim:
        raise ValueError(f"Rank {rank}: {grad_type} grad has shape {grad.shape}, expected {expected_ndim}D array")
    all_grads = all_gather(grad, rank, world_size, queues, is_gradient=True)
    if not isinstance(all_grads, np.ndarray):
        all_grads = np.array(all_grads)
    if all_grads.ndim < 2:
        raise ValueError(f"Rank {rank}: {grad_type} all_grads has shape {all_grads.shape}, expected at least 2D")
    total_grad = np.sum(all_grads, axis=0)  # Sum along rank dimension
    if total_grad.ndim == 0:
        raise ValueError(f"Rank {rank}: {grad_type} total_grad is a scalar, expected at least 1D array")
    # Handle 1D (bias) and 2D (weight) gradients
    shard_size = total_grad.shape[-1] // world_size if total_grad.ndim > 1 else total_grad.shape[0] // world_size
    if total_grad.ndim > 1:
        result = total_grad[:, rank * shard_size : (rank + 1) * shard_size]
    else:
        result = total_grad[rank * shard_size : (rank + 1) * shard_size]
    if rank == 0:
        print(f"Rank {rank}: reduce_scatter {grad_type} grad shape {grad.shape}, all_grads shape {all_grads.shape}, total_grad shape {total_grad.shape}, result shape {result.shape}, expected_ndim {expected_ndim}")
    return result

# Worker process for each rank
def worker(rank, world_size, input_data, target, lr, epochs, queues):
    # Initialize model shard (parameter sharding)
    input_dim = input_data.shape[1]
    output_dim = target.shape[1]
    shard_size = output_dim // world_size
    if output_dim % world_size != 0:
        raise ValueError(f"Output dim {output_dim} must be divisible by world_size {world_size}")
    model = SimpleLinearModel(input_dim, shard_size)  # Each rank holds a shard

    for epoch in range(epochs):
        # Forward pass
        try:
            full_weight = all_gather(model.weight, rank, world_size, queues, is_gradient=False)
            full_bias = all_gather(model.bias, rank, world_size, queues, is_gradient=False)
            if rank == 0:
                print(f"Rank {rank}, Epoch {epoch + 1}: full_weight shape {full_weight.shape}, full_bias shape {full_bias.shape}")
        except Exception as e:
            print(f"Rank {rank} failed in all_gather: {e}")
            raise

        output = np.dot(input_data, full_weight) + full_bias

        # Compute loss and grad_output (simplified MSE)
        loss = np.mean((output - target) ** 2)
        grad_output = 2 * (output - target) / output.size

        # Backward pass using full_weight
        grad_input, grad_weight, grad_bias = model.backward(input_data, grad_output, full_weight)
        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch + 1}: grad_weight shape {grad_weight.shape}, grad_bias shape {grad_bias.shape}")

        # Reduce-scatter gradients (ensure correct order)
        try:
            grad_weight_shard = reduce_scatter(grad_weight, rank, world_size, queues, expected_ndim=2, grad_type="weight")
            grad_bias_shard = reduce_scatter(grad_bias, rank, world_size, queues, expected_ndim=1, grad_type="bias")
            if rank == 0:
                print(f"Rank {rank}, Epoch {epoch + 1}: grad_weight_shard shape {grad_weight_shard.shape}, grad_bias_shard shape {grad_bias_shard.shape}")
        except Exception as e:
            print(f"Rank {rank} failed in reduce_scatter: {e}")
            raise

        # Update shard
        try:
            model.update(grad_weight_shard, grad_bias_shard, lr)
        except Exception as e:
            print(f"Rank {rank} failed in update: {e}")
            raise

        if rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # Gather full model on rank 0
    if rank == 0:
        try:
            full_weight = all_gather(model.weight, rank, world_size, queues, is_gradient=False)
            full_bias = all_gather(model.bias, rank, world_size, queues, is_gradient=False)
            print("Final full model weight shape:", full_weight.shape)
            print("Final full model bias shape:", full_bias.shape)
        except Exception as e:
            print(f"Rank {rank} failed in final all_gather: {e}")
            raise

# Main function to start processes
def main():
    world_size = 2  # Number of processes (GPUs/CPUs)
    num_epochs = 5
    learning_rate = 0.01
    batch_size = 10
    input_dim = 5
    output_dim = 4  # Must be divisible by world_size

    # Generate sample data
    np.random.seed(42)  # For reproducibility
    input_data = np.random.randn(batch_size, input_dim)
    target = np.random.randn(batch_size, output_dim)

    # Queues for communication (one per rank)
    queues = [Queue() for _ in range(world_size)]

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, input_data, target, learning_rate, num_epochs, queues))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()