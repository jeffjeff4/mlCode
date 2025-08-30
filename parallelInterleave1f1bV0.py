import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import copy


# Simple model: Each stage handles multiple chunks (linear layers)
class StageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, chunks, is_first=False, is_last=False):
        super().__init__()
        self.chunks = chunks  # List of chunk indices (e.g., [1, 3] for stage 0)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if is_first and i == 0 else hidden_dim, hidden_dim)
            for i in range(len(chunks))
        ])
        self.is_last = is_last
        if self.is_last:
            self.loss_fn = nn.MSELoss()

    def forward(self, x, chunk_idx):
        layer_idx = self.chunks.index(chunk_idx)
        return self.layers[layer_idx](x)


# Function for each process (stage)
def stage_process(rank, world_size, num_micro_batches, batch_size, input_dim, hidden_dim, chunks_per_stage):
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
    chunks = chunks_per_stage[rank]
    model = StageModel(input_dim, hidden_dim, chunks, is_first=is_first, is_last=is_last).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Simulate data
    inputs = [torch.randn(batch_size, input_dim).to(device) for _ in range(num_micro_batches)] if is_first else None
    targets = [torch.randn(batch_size, hidden_dim).to(device) for _ in range(num_micro_batches)] if is_last else None

    # Forward function
    def do_forward(micro_batch_idx, chunk_idx, model_copy):
        if is_first and chunk_idx == 1:  # First chunk of first stage
            activation = inputs[micro_batch_idx].clone()
        else:
            activation = torch.zeros(batch_size, hidden_dim, device=device)
            prev_rank = 0 if chunk_idx in [2, 4] else 1  # Chunk 1,3: rank 0; Chunk 2,4: rank 1
            dist.recv(tensor=activation, src=prev_rank)

        activation = activation.clone().requires_grad_(True)
        output = model_copy(activation, chunk_idx)

        if not (is_last and chunk_idx == 4):  # Chunk 4 is the last
            next_rank = 1 if chunk_idx in [1, 3] else 0  # Chunk 1->2, 3->4: rank 0->1; Chunk 2->3: rank 1->0
            dist.send(tensor=output.detach().clone(), dst=next_rank)
        return activation, output

    # Backward function
    def do_backward(activation, output, micro_batch_idx, chunk_idx, model_copy):
        if output is None or activation is None:
            raise ValueError(f"Output or activation is None for micro_batch {micro_batch_idx}, chunk {chunk_idx}")
        output = output.clone().requires_grad_(True)
        if is_last and chunk_idx == 4:  # Last chunk
            loss = model_copy.loss_fn(output, targets[micro_batch_idx])
            grad_output = torch.autograd.grad(loss, output, create_graph=False)[0]
        else:
            grad_output = torch.zeros_like(output)
            next_rank = 0 if chunk_idx in [1, 3] else 1  # Chunk 4->3, 2->1: rank 1->0; Chunk 3->2: rank 0->1
            dist.recv(tensor=grad_output, src=next_rank)

        torch.autograd.backward(output, grad_output, retain_graph=False)

        if not (is_first and chunk_idx == 1):
            grad_activation = activation.grad.clone()
            prev_rank = 0 if chunk_idx in [2, 4] else 1
            dist.send(tensor=grad_activation, dst=prev_rank)

        return loss.item() if is_last and chunk_idx == 4 else 0.0

    # Interleaved 1F1B Schedule
    losses = []
    num_chunks = sum(len(c) for c in chunks_per_stage.values())  # Total chunks (4)
    activations = {(i, c): None for i in range(num_micro_batches) for c in range(1, num_chunks + 1)}
    outputs = {(i, c): None for i in range(num_micro_batches) for c in range(1, num_chunks + 1)}
    model_copies = {(i, c): None for i in range(num_micro_batches) for c in range(1, num_chunks + 1)}

    # Corrected Schedule
    schedule = []
    # Warm-up: Fill pipeline
    for i in range(num_micro_batches):
        for c in range(1, min(i * 2 + 1, num_chunks + 1)):
            schedule.append(('F', i, c))
    # Steady: Interleave F and B
    for i in range(num_micro_batches):
        for c in range(max(1, (i - num_micro_batches + world_size) * 2 + 1),
                       min(i * 2 + world_size + 1, num_chunks + 1)):
            schedule.append(('F', i, c))
            if i >= world_size:
                schedule.append(('B', i - world_size, num_chunks - (c - 1)))
    # Drain: Remaining backwards
    for i in range(num_micro_batches - world_size, num_micro_batches):
        for c in range(num_chunks, max(0, num_chunks - (i + world_size - num_micro_batches) * 2), -1):
            schedule.append(('B', i, c))

    # Execute schedule
    for op, micro_batch_idx, chunk_idx in schedule:
        if chunk_idx in chunks:  # This stage handles this chunk
            key = (micro_batch_idx, chunk_idx)
            if op == 'B':
                if outputs[key] is None or model_copies[key] is None:
                    continue
                model_copy = model_copies[key]
                loss = do_backward(activations[key], outputs[key], micro_batch_idx, chunk_idx, model_copy)
                if is_last and chunk_idx == 4:
                    losses.append(loss)
                for param, copy_param in zip(model.parameters(), model_copy.parameters()):
                    if copy_param.grad is not None:
                        param.grad = copy_param.grad.clone()
                optimizer.step()
                optimizer.zero_grad()
                model_copies[key] = None
                activations[key] = None
                outputs[key] = None
            else:
                model_copy = copy.deepcopy(model)
                model_copies[key] = model_copy
                activations[key], outputs[key] = do_forward(micro_batch_idx, chunk_idx, model_copy)

    if is_last:
        if losses:
            print(f"Rank {rank} Average loss: {sum(losses) / len(losses)}")
        else:
            print(f"Rank {rank}: No losses recorded")

    dist.destroy_process_group()


# Main function
def main():
    world_size = 2
    num_micro_batches = 8
    batch_size = 2
    input_dim = 4
    hidden_dim = 4
    chunks_per_stage = {0: [1, 3], 1: [2, 4]}  # Interleaved chunks

    mp.set_start_method('spawn', force=True)
    mp.spawn(stage_process, args=(world_size, num_micro_batches, batch_size, input_dim, hidden_dim, chunks_per_stage),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
