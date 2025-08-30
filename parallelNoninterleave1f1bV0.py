####Key Changes
####
####Per-Micro-Batch Model Copies:
####Instead of stashing weights via model.state_dict(), we create a deep copy of the entire model (copy.deepcopy(model)) for each micro-batch's forward pass.
####Each forward and backward pass uses its own model copy (model_copy), ensuring that the weights used in the forward pass are preserved for the backward pass, avoiding in-place modifications to the original model's weights.
####
####Gradient Transfer:
####After do_backward, gradients from the model copy are transferred to the original model (param.grad = copy_param.grad.clone()) before calling optimizer.step().
####This ensures that the optimizer updates the original model without affecting the autograd graph of the model copy used for the backward pass.
####
####Cloned Tensors:
####Continued using clone() for activation, output, and gradients to prevent in-place modifications.
####Ensured output in do_backward is cloned and re-marked as requires_grad_(True).
####
####Optimizer Management:
####optimizer.zero_grad() is called after each step() to clear gradients, preventing accumulation.
####Updates are applied to the original model only after gradients are computed using the model copy, isolating the autograd graph.
####
####Memory Management:
####Model copies are cleared (model_copies[j] = None) after their backward pass to reduce memory usage.
####
####Why This Fixes the Error
####Avoiding In-place Modification: By using a separate model copy for each micro-batch, the weights used in the forward pass (model_copy.linear.weight) are isolated from the original model's weights (model.linear.weight). The optimizer's in-place update on model does not affect the autograd graph of model_copy, which is used for gradient computation.
####Preserving Autograd Graph: The output tensor's computation graph references model_copy.linear.weight, which remains unchanged during the backward pass. The original model's weights are updated only after gradients are computed, avoiding the version mismatch (version 2; expected version 1).
####Safe Weight Stashing: The deep copy of the model ensures that each micro-batch's forward and backward passes use consistent weights, fulfilling PipeDream's requirement for weight consistency without interfering with the autograd engine.
####
####Running the Code
####Requirements:
####PyTorch (pip install torch).
####CPU or GPU with CUDA (at least 4 GPUs if using nccl; falls back to gloo for CPU).
####
####Execution:
####Save as pipedream_1f1b_fixed_v2.py.
####Run: python pipedream_1f1b_fixed_v2.py.
####For GPUs: CUDA_VISIBLE_DEVICES=0,1,2,3 python pipedream_1f1b_fixed_v2.py.
####
####Expected Output:
####Example: Rank 3 Average loss: 0.12345 (actual value varies due to random data).
####The code runs 4 processes, each handling one stage, with proper 1F1B scheduling.
####
####Debugging:
####Uncomment torch.autograd.set_detect_anomaly(True) to trace any remaining issues (disable for production).
####If the port 29500 is in use, change MASTER_PORT to another value (e.g., 29501).
####
####PipeDream 1F1B Schedule
####For world_size=4, num_micro_batches=8:
####textTime: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
####Rank 0: F1 F2 F3 F4 F5 F6 F7 F8 B8 B7 B6 B5 B4 B3 B2 B1 -  -  -  -
####Rank 1: -  F1 F2 F3 F4 F5 F6 F7 F8 B8 B7 B6 B5 B4 B3 B2 B1 -  -
####Rank 2: -  -  F1 F2 F3 F4 F5 F6 F7 F8 B8 B7 B6 B5 B4 B3 B2 B1 -
####Rank 3: -  -  -  F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8
####
####Warm-up: Time 0-3, fills pipeline (F1-F4).
####Steady: Time 4-15, alternates F/B (e.g., Rank 3: F1→B1, F2→B2).
####Drain: Time 16-19, completes B5-B8.
####
####Additional Notes
####Memory Trade-off: Using copy.deepcopy(model) for each micro-batch increases memory usage compared to weight stashing via state_dict. For larger models, consider optimizing by stashing only necessary parameters or using checkpointing.
####Scalability: The simple 4-layer linear model is for demonstration. For a real Transformer, replace StageModel with a Transformer layer (e.g., nn.TransformerEncoderLayer).
####Multi-node: For multi-node setups, replace localhost with the master node’s IP.
####Debugging: If errors persist, enable anomaly detection or share the full error trace with PyTorch version and hardware details (CPU/GPU, number of GPUs).
####
####If you encounter new errors, need a larger model example, or want to extend to interleaved 1F1B, please provide details, and I’ll further refine the solution!


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import copy


# Simple model: 4 linear layers, each stage one layer
class StageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, is_first=False, is_last=False):
        super().__init__()
        self.linear = nn.Linear(input_dim if is_first else hidden_dim, hidden_dim)
        self.is_last = is_last
        if self.is_last:
            self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)


# Function for each process (stage)
def stage_process(rank, world_size, num_micro_batches, batch_size, input_dim, hidden_dim):
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # Initialize distributed process group
    backend = 'gloo' if not torch.cuda.is_available() else 'nccl'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Device
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() and torch.cuda.device_count() >= world_size else 'cpu')

    # Model stage
    is_first = rank == 0
    is_last = rank == world_size - 1
    model = StageModel(input_dim, hidden_dim, is_first=is_first, is_last=is_last).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Simulate data
    inputs = [torch.randn(batch_size, input_dim).to(device) for _ in range(num_micro_batches)] if is_first else None
    targets = [torch.randn(batch_size, hidden_dim).to(device) for _ in range(num_micro_batches)] if is_last else None

    # Forward function
    def do_forward(micro_batch_idx, model_copy):
        if is_first:
            activation = inputs[micro_batch_idx].clone()
        else:
            activation = torch.zeros(batch_size, hidden_dim, device=device)
            dist.recv(tensor=activation, src=rank - 1)

        activation = activation.clone().requires_grad_(True)
        output = model_copy(activation)

        if not is_last:
            dist.send(tensor=output.detach().clone(), dst=rank + 1)
        return activation, output

    # Backward function
    def do_backward(activation, output, micro_batch_idx, model_copy):
        output = output.clone().requires_grad_(True)
        if is_last:
            loss = model_copy.loss_fn(output, targets[micro_batch_idx])
            grad_output = torch.autograd.grad(loss, output, create_graph=False)[0]
        else:
            grad_output = torch.zeros_like(output)
            dist.recv(tensor=grad_output, src=rank + 1)

        # Backward pass
        torch.autograd.backward(output, grad_output, retain_graph=False)

        if not is_first:
            grad_activation = activation.grad.clone()
            dist.send(tensor=grad_activation, dst=rank - 1)

        return loss.item() if is_last else 0.0

    # PipeDream 1F1B Schedule
    losses = []
    activations = [None] * num_micro_batches
    outputs = [None] * num_micro_batches
    model_copies = [None] * num_micro_batches  # Store model copies

    # Enable anomaly detection for debugging (optional)
    # torch.autograd.set_detect_anomaly(True)

    for phase in ['warmup', 'steady', 'drain']:
        if phase == 'warmup':
            # Fill pipeline with forwards (p-1 micro-batches)
            for i in range(world_size - 1):
                model_copy = copy.deepcopy(model)  # Deep copy model
                model_copies[i] = model_copy
                activations[i], outputs[i] = do_forward(i, model_copy)

        elif phase == 'steady':
            # Alternate 1F1B
            for i in range(world_size - 1, num_micro_batches):
                # Forward
                model_copy = copy.deepcopy(model)
                model_copies[i] = model_copy
                activations[i], outputs[i] = do_forward(i, model_copy)

                # Backward for (i - (p-1))
                j = i - (world_size - 1)
                if j >= 0:
                    model_copy = model_copies[j]
                    loss = do_backward(activations[j], outputs[j], j, model_copy)
                    if is_last:
                        losses.append(loss)
                    # Update original model with gradients from model_copy
                    for param, copy_param in zip(model.parameters(), model_copy.parameters()):
                        if copy_param.grad is not None:
                            param.grad = copy_param.grad.clone()
                    optimizer.step()
                    optimizer.zero_grad()
                    model_copies[j] = None  # Clear model copy

        elif phase == 'drain':
            # Drain remaining backwards
            for i in range(num_micro_batches - (world_size - 1), num_micro_batches):
                model_copy = model_copies[i]
                loss = do_backward(activations[i], outputs[i], i, model_copy)
                if is_last:
                    losses.append(loss)
                # Update original model
                for param, copy_param in zip(model.parameters(), model_copy.parameters()):
                    if copy_param.grad is not None:
                        param.grad = copy_param.grad.clone()
                optimizer.step()
                optimizer.zero_grad()
                model_copies[i] = None

    if is_last:
        print(f"Rank {rank} Average loss: {sum(losses) / len(losses)}")

    dist.destroy_process_group()


# Main function to spawn processes
def main():
    world_size = 4  # Number of stages
    num_micro_batches = 8
    batch_size = 2
    input_dim = 4
    hidden_dim = 4

    mp.set_start_method('spawn', force=True)
    mp.spawn(stage_process, args=(world_size, num_micro_batches, batch_size, input_dim, hidden_dim), nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()