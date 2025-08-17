#################################################################
##correct version
#################################################################

# filename: attention_tp_backward_fix_gloo.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import os


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} successfully set up.")


def cleanup():
    dist.destroy_process_group()


# --- Custom AllGather Function for the Backward Pass ---
class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.world_size = dist.get_world_size()

        local_output = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(local_output, tensor)
        full_output = torch.cat(local_output, dim=-1)

        return full_output

    @staticmethod
    def backward(ctx, grad_output):
        # We cannot use dist.reduce_scatter with Gloo.
        # Instead, we perform an all_reduce and then chunk the result.

        # 1. All-Reduce the gradient from all processes.
        # This will give every process the sum of gradients from all chunks.
        all_reduced_grad = torch.empty_like(grad_output)
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, async_op=False)
        all_reduced_grad.copy_(grad_output)  # copy the result back

        # 2. Split the full summed gradient into chunks and select the local one.
        grad_chunks = list(torch.chunk(all_reduced_grad, ctx.world_size, dim=-1))

        # 3. Each rank returns its corresponding chunk as its local gradient.
        local_grad = grad_chunks[dist.get_rank()]

        return local_grad


def run_tp_attention_with_backward(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cpu")

    # === Input ===
    batch, seq_len, hidden_size = 1, 1, 8
    split_size = hidden_size // world_size
    input_tensor = torch.randn(batch, seq_len, hidden_size, device=device, requires_grad=True)

    # === Linear Weights (Column Parallel) ===
    q_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
    k_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
    v_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)

    # === Q/K/V projections ===
    q_part = input_tensor @ q_weight
    k_part = input_tensor @ k_weight
    v_part = input_tensor @ v_weight

    # === All-Gather Q/K/V ===
    Q = AllGather.apply(q_part)
    K = AllGather.apply(k_part)
    V = AllGather.apply(v_part)

    # === Attention Computation ===
    attn_scores = Q @ K.transpose(-2, -1) / (hidden_size ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = attn_probs @ V

    # === Output Projection (Row Parallel) ===
    output_proj_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
    local_output = attn_output @ output_proj_weight

    # === All-Gather Output for the final tensor ===
    full_output = AllGather.apply(local_output)

    # === Fake Loss (L2 norm) ===
    loss = (full_output ** 2).mean()
    print(f"[Rank {rank}] loss requires_grad: {loss.requires_grad}, loss.grad_fn: {loss.grad_fn}")

    loss.backward()

    # The gradients should now be correctly computed on the weights
    print(f"[Rank {rank}] q_weight.grad shape: {q_weight.grad.shape}, sum: {q_weight.grad.sum()}")
    print(
        f"[Rank {rank}] output_proj_weight.grad shape: {output_proj_weight.grad.shape}, sum: {output_proj_weight.grad.sum()}")

    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_tp_attention_with_backward, args=(world_size,), nprocs=world_size)



#####################################################################
######wrong version
#####################################################################
##### filename: attention_tp_backward_fix.py
####import torch
####import torch.distributed as dist
####import torch.multiprocessing as mp
####import torch.nn.functional as F
####import os
####
####
####def setup(rank, world_size):
####    os.environ["MASTER_ADDR"] = "localhost"
####    os.environ["MASTER_PORT"] = "12356"
####    dist.init_process_group("gloo", rank=rank, world_size=world_size)
####    print(f"Rank {rank} successfully set up.")
####
####
####def cleanup():
####    dist.destroy_process_group()
####
####
##### --- Custom AllGather Function for the Backward Pass ---
####class AllGather(torch.autograd.Function):
####    @staticmethod
####    def forward(ctx, tensor):
####        # Save world_size to use in the backward pass
####        ctx.world_size = dist.get_world_size()
####
####        # All-gather the input tensor from all ranks
####        local_output = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
####        dist.all_gather(local_output, tensor)
####        full_output = torch.cat(local_output, dim=-1)
####
####        return full_output
####
####    @staticmethod
####    def backward(ctx, grad_output):
####        # In the backward pass, we need to reduce_scatter the gradients
####        # The gradient of the full output is `grad_output`
####        # We need to scatter this gradient back to each local rank
####
####        # Split the full gradient into chunks
####        grad_chunks = list(torch.chunk(grad_output, ctx.world_size, dim=-1))
####
####        # This will be the local gradient for the current rank
####        local_grad = grad_chunks[dist.get_rank()].contiguous()
####
####        # Sum the gradients across all ranks (Reduce) and then scatter them
####        # `dist.reduce_scatter` is the efficient way to do this
####        output_grad = torch.empty_like(local_grad)
####        dist.reduce_scatter(output_grad, grad_chunks)
####
####        return output_grad
####
####
####def run_tp_attention_with_backward(rank, world_size):
####    setup(rank, world_size)
####    device = torch.device("cpu")
####
####    # === Input ===
####    batch, seq_len, hidden_size = 1, 1, 8
####    split_size = hidden_size // world_size
####    input_tensor = torch.randn(batch, seq_len, hidden_size, device=device, requires_grad=True)
####
####    # === Linear Weights (Column Parallel) ===
####    q_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####    k_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####    v_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####
####    # === Q/K/V projections ===
####    q_part = input_tensor @ q_weight
####    k_part = input_tensor @ k_weight
####    v_part = input_tensor @ v_weight
####
####    # === All-Gather Q/K/V ===
####    Q = AllGather.apply(q_part)
####    K = AllGather.apply(k_part)
####    V = AllGather.apply(v_part)
####
####    # === Attention Computation ===
####    attn_scores = Q @ K.transpose(-2, -1) / (hidden_size ** 0.5)
####    attn_probs = F.softmax(attn_scores, dim=-1)
####    attn_output = attn_probs @ V
####
####    # === Output Projection (Row Parallel) ===
####    output_proj_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####    local_output = attn_output @ output_proj_weight
####
####    # === All-Gather Output for the final tensor ===
####    full_output = AllGather.apply(local_output)
####
####    # === Fake Loss (L2 norm) ===
####    loss = (full_output ** 2).mean()
####    print(f"[Rank {rank}] loss requires_grad: {loss.requires_grad}, loss.grad_fn: {loss.grad_fn}")
####
####    loss.backward()
####
####    # The `grad_output` in the custom backward function is the gradient from `loss`
####    # We can now inspect the gradients on the weights
####    print(f"[Rank {rank}] q_weight.grad requires_grad: {q_weight.grad.requires_grad}")
####    print(f"[Rank {rank}] q_weight.grad shape: {q_weight.grad.shape}, sum: {q_weight.grad.sum()}")
####
####    cleanup()
####
####
####if __name__ == "__main__":
####    world_size = 2
####    mp.spawn(run_tp_attention_with_backward, args=(world_size,), nprocs=world_size)