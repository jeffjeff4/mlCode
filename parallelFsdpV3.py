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

        return grad_input, grad_weight, grad_bias

    def update(self, grad_weight_shard, grad_bias_shard, lr=0.01):
        if grad_weight_shard.shape != self.weight.shape:
            raise ValueError(
                f"grad_weight_shard shape {grad_weight_shard.shape} does not match self.weight shape {self.weight.shape}")
        if grad_bias_shard.shape != self.bias.shape:
            raise ValueError(
                f"grad_bias_shard shape {grad_bias_shard.shape} does not match self.bias shape {self.bias.shape}")

        self.weight -= lr * grad_weight_shard
        self.bias -= lr * grad_bias_shard


# FSDP Helper Functions
def all_gather(param, rank, world_size, queues):
    """
    All-gather: Collect parameter or gradient shards from all ranks.
    """
    param = np.asarray(param)

    # Send my shard to all other ranks
    for j in range(world_size):
        if j != rank:
            queues[j].put((rank, param))

    # Receive shards from all other ranks
    full_params = [None] * world_size
    full_params[rank] = param
    for i in range(world_size - 1):
        try:
            sender_rank, shard = queues[rank].get(timeout=5.0)
            full_params[sender_rank] = np.asarray(shard)
        except mp.queues.Empty:
            raise RuntimeError(f"Rank {rank} failed to receive shard.")

    # Concatenate based on dimensionality
    if param.ndim == 2:  # Weight
        return np.concatenate(full_params, axis=1)
    elif param.ndim == 1:  # Bias
        return np.concatenate(full_params, axis=0)
    else:
        raise ValueError("Unsupported param dimension")


def reduce_and_scatter(grad, rank, world_size, queues):
    """
    Combines the reduce and scatter steps.
    Each rank sends its full gradient, and then gets back its final, summed shard.
    """
    grad = np.asarray(grad)

    # 1. All-gather all full gradients
    # Each rank will send its full gradient and receive everyone else's.
    all_grads = [None] * world_size
    all_grads[rank] = grad

    # Send my gradient to all other ranks
    for j in range(world_size):
        if j != rank:
            queues[j].put((rank, grad))

    # Receive gradients from all other ranks
    for _ in range(world_size - 1):
        try:
            sender_rank, received_grad = queues[rank].get(timeout=5.0)
            all_grads[sender_rank] = np.asarray(received_grad)
        except mp.queues.Empty:
            raise RuntimeError(f"Rank {rank} failed to receive all gradients.")

    # 2. Sum the gradients
    total_grad = np.sum(all_grads, axis=0)

    # 3. Scatter the summed gradients
    # Slice the total gradient to get our shard, matching the parameter sharding.
    if grad.ndim == 2:  # Weight gradient
        shard_size = total_grad.shape[1] // world_size
        return total_grad[:, rank * shard_size: (rank + 1) * shard_size]
    elif grad.ndim == 1:  # Bias gradient
        shard_size = total_grad.shape[0] // world_size
        return total_grad[rank * shard_size: (rank + 1) * shard_size]
    else:
        raise ValueError("Unsupported gradient dimension")


# Worker process for each rank
def worker(rank, world_size, input_data, target, lr, epochs, queues):
    # Initialize model shard (parameter sharding)
    input_dim = input_data.shape[1]
    output_dim = target.shape[1]
    shard_size = output_dim // world_size
    if output_dim % world_size != 0:
        raise ValueError(f"Output dim {output_dim} must be divisible by world_size {world_size}")
    model = SimpleLinearModel(input_dim, shard_size)

    for epoch in range(epochs):
        # Forward pass: All-gather full parameters
        full_weight = all_gather(model.weight, rank, world_size, queues)
        full_bias = all_gather(model.bias, rank, world_size, queues)

        output = np.dot(input_data, full_weight) + full_bias

        # Compute loss and grad_output
        loss = np.mean((output - target) ** 2)
        grad_output = 2 * (output - target) / output.size

        # Backward pass using full_weight to compute full gradients
        _, grad_weight, grad_bias = model.backward(input_data, grad_output, full_weight)

        # Reduce-scatter gradients to get the correct shard
        grad_weight_shard = reduce_and_scatter(grad_weight, rank, world_size, queues)
        grad_bias_shard = reduce_and_scatter(grad_bias, rank, world_size, queues)

        # Update shard
        model.update(grad_weight_shard, grad_bias_shard, lr)

        if rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")


# Main function to start processes
def main():
    world_size = 2
    num_epochs = 5
    learning_rate = 0.01
    batch_size = 10
    input_dim = 5
    output_dim = 4

    # Generate sample data
    np.random.seed(42)
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
