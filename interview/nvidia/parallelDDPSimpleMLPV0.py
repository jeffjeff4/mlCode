import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np


# --- 1. DDP Setup and Cleanup (for CPU) ---

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


# --- 2. Generated Dataset ---

class SyntheticDataset(Dataset):
    """
    A simple synthetic dataset for a regression task.
    y = X @ [true_w] + [true_b] + noise
    """

    def __init__(self, n_samples, input_size, output_size, seed=42):
        self.n_samples = n_samples
        np.random.seed(seed)

        true_w = np.random.randn(input_size, output_size).astype(np.float32)
        true_b = np.random.randn(output_size).astype(np.float32)

        self.X = np.random.randn(n_samples, input_size).astype(np.float32)
        noise = np.random.randn(n_samples, output_size).astype(np.float32) * 0.1
        self.y = (self.X @ true_w + true_b + noise)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return tensors directly, as we're on CPU
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# --- 3. The MLP Model (Fully Manual) ---

class ManualMLP:
    """
    A simple MLP with fully manual forward/backward passes.
    This does *not* inherit from nn.Module.
    """

    def __init__(self, input_size, hidden_size, output_size, rank, world_size):
        # Seed to ensure all processes *could* initialize the same,
        # but we will broadcast from rank 0 to guarantee it.
        torch.manual_seed(42)

        # --- 1. Initialize Parameters ---
        self.W1 = torch.randn(input_size, hidden_size) * 0.1
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.1
        self.b2 = torch.zeros(output_size)
        self.params = [self.W1, self.b1, self.W2, self.b2]

        # --- 2. Manually Broadcast Weights (Part of "Manual DDP") ---
        # This is a critical step. We must ensure all models
        # start with the exact same weights from rank 0.
        for p in self.params:
            dist.broadcast(p, src=0)

        # --- 3. Buffers for intermediate values (for backward pass) ---
        self.x_input = None
        self.z1 = None
        self.a1 = None
        self.z2 = None  # (This is also y_pred)

        # --- 4. Buffers for gradients ---
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        self.grads = [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]

    def forward(self, x):
        """
        Manual forward pass.
        We save intermediate values needed for the backward pass.
        """
        self.x_input = x

        # Layer 1: z1 = X @ W1 + b1
        self.z1 = torch.matmul(x, self.W1) + self.b1

        # Activation 1: a1 = ReLU(z1)
        self.a1 = torch.maximum(torch.tensor(0.0), self.z1)

        # Layer 2: z2 = a1 @ W2 + b2
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2

        return self.z2  # z2 is our prediction, y_pred

    def backward(self, y_true):
        """
        Manual backward pass (Chain Rule).
        This computes the gradients *only* for the local batch.
        """
        N = y_true.shape[0]  # Batch size
        y_pred = self.z2

        # --- Start at the end: Gradient of Loss w.r.t. Prediction ---
        # Loss = (1/N) * sum( (y_pred - y_true)^2 )
        # dL/dy_pred = (2/N) * (y_pred - y_true)
        grad_y_pred = (2.0 / N) * (y_pred - y_true)

        # --- Step 1: Gradients for Layer 2 (W2, b2) ---
        # y_pred (or z2) = a1 @ W2 + b2
        grad_z2 = grad_y_pred  # Gradient just passes through

        # dL/dW2 = (dL/dz2) * (dz2/dW2) = grad_z2 * a1
        self.grad_W2 = torch.matmul(self.a1.T, grad_z2)

        # dL/db2 = (dL/dz2) * (dz2/db2) = grad_z2 * 1
        self.grad_b2 = torch.sum(grad_z2, axis=0)

        # --- Step 2: Propagate Gradients to Layer 1 (W1, b1) ---

        # dL/da1 = (dL/dz2) * (dz2/da1) = grad_z2 @ W2.T
        grad_a1 = torch.matmul(grad_z2, self.W2.T)

        # dL/dz1 = (dL/da1) * (da1/dz1)
        # da1/dz1 is the derivative of ReLU
        relu_deriv = (self.z1 > 0).float()
        grad_z1 = grad_a1 * relu_deriv  # Element-wise

        # dL/dW1 = (dL/dz1) * (dz1/dW1) = grad_z1 * x_input
        self.grad_W1 = torch.matmul(self.x_input.T, grad_z1)

        # dL/db1 = (dL/dz1) * (dz1/db1) = grad_z1 * 1
        self.grad_b1 = torch.sum(grad_z1, axis=0)

        # Store grads in a list for easier averaging
        self.grads = [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]

    def average_gradients(self, world_size):
        """
        This is the **MANUAL DDP** step.
        We use all_reduce (sum) and then divide by world_size
        to get the average gradient across all processes.
        """
        for grad_tensor in self.grads:
            # Sum all gradients from all processes
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
            # Divide by the number of processes to get the average
            grad_tensor /= world_size

    def update_weights(self, lr):
        """
        Manual optimizer step (SGD).
        This is identical on all processes because they all
        have the same averaged gradients.
        """
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2


# --- 4. Manual Loss Function ---

def mse_loss(y_pred, y_true):
    """Manual Mean Squared Error loss."""
    return torch.mean((y_pred - y_true) ** 2)


# --- 5. Training Function ---

def train_epoch(epoch, model, dataloader, lr, rank, world_size):
    """Runs a single training epoch."""
    dataloader.sampler.set_epoch(epoch)

    total_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        # 1. FORWARD PASS (Manual)
        y_pred = model.forward(inputs)
        loss = mse_loss(y_pred, labels)

        # 2. BACKWARD PASS (Manual)
        # This computes local gradients (e.g., model.grad_W1)
        model.backward(labels)

        # 3. DDP Gradient Synchronization (Manual)
        # This averages the gradients across all processes
        model.average_gradients(world_size)

        # 4. WEIGHT UPDATE (Manual)
        # This updates the weights. Since all processes have
        # the same averaged gradients, all models stay in sync.
        model.update_weights(lr)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    if rank == 0:
        print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")


# --- 6. Evaluation Function ---

def evaluate(model, dataloader, rank, world_size):
    """Runs evaluation on the validation dataset."""
    total_loss = 0.0

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, labels in dataloader:
            y_pred = model.forward(inputs)
            loss = mse_loss(y_pred, labels)
            total_loss += loss.item()

    # We must average the loss across all processes
    loss_tensor = torch.tensor(total_loss, dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    avg_loss = loss_tensor.item() / (len(dataloader) * world_size)

    if rank == 0:
        print(f"Validation Loss: {avg_loss:.4f}")


# --- 7. Main Worker Function (Run by each process) ---

def main_worker():
    """The main function to be run by each process."""

    # --- BEFORE ---
    # rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])

    # --- AFTER ---
    # Use .get() with default values of "0" and "1"
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    setup(rank, world_size)

    try:
        # --- Config ---
        input_size = 10
        hidden_size = 32
        output_size = 1
        epochs = 5
        batch_size = 32  # This is the batch size *per process*
        learning_rate = 0.01

        # --- Datasets and Samplers ---
        train_dataset = SyntheticDataset(1024, input_size, output_size, seed=42)
        val_dataset = SyntheticDataset(256, input_size, output_size, seed=123)

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        # --- DataLoaders ---
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False
        )

        # --- Model ---
        # The model's __init__ handles broadcasting the initial
        # weights from rank 0 to all other processes.
        model = ManualMLP(input_size, hidden_size, output_size, rank, world_size)

        if rank == 0:
            print("Starting manual DDP training on CPU...")

        # --- Run ---
        for epoch in range(epochs):
            train_epoch(epoch, model, train_loader, learning_rate, rank, world_size)
            evaluate(model, val_loader, rank, world_size)

    finally:
        cleanup()
        if rank == 0:
            print("DDP training complete.")


if __name__ == "__main__":
    # torchrun will spawn `nproc_per_node` processes,
    # each running this script and calling main_worker().
    main_worker()
