# filename: attention_tp_backward.py
#can not run, can not be used in interview
#####################################################################
######wrong version
#####################################################################


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import os

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    #mac does not have nccl, using gloo
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #torch.cuda.set_device(rank)
    device = torch.device("cpu")

def cleanup():
    dist.destroy_process_group()

def run_tp_attention_with_backward(rank, world_size):
    setup(rank, world_size)
    #device = torch.device(f"cuda:{rank}")
    device = torch.device(f"cpu:{rank}")

    # === Input ===
    batch, seq_len, hidden_size = 1, 1, 8
    split_size = hidden_size // world_size
    input_tensor = torch.randn(batch, seq_len, hidden_size, device=device, requires_grad=True)

    # === Linear Weights ===
    q_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
    k_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
    v_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)

    # === Q/K/V projections (Column Parallel) ===
    q_part = input_tensor @ q_weight  # [1, 1, 4]
    k_part = input_tensor @ k_weight
    v_part = input_tensor @ v_weight

    # === All-Gather Q/K/V ===
    q_list = [torch.empty_like(q_part) for _ in range(world_size)]
    k_list = [torch.empty_like(k_part) for _ in range(world_size)]
    v_list = [torch.empty_like(v_part) for _ in range(world_size)]

    dist.all_gather(q_list, q_part)
    dist.all_gather(k_list, k_part)
    dist.all_gather(v_list, v_part)

    Q = torch.cat(q_list, dim=-1)
    K = torch.cat(k_list, dim=-1)
    V = torch.cat(v_list, dim=-1)

    # === Attention Computation ===
    attn_scores = Q @ K.transpose(-2, -1) / (hidden_size ** 0.5)
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = attn_probs @ V  # [1,1,8]

    # === Output Projection (Row Parallel) ===
    output_proj_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
    local_output = attn_output @ output_proj_weight  # [1,1,4]

    # === All-Gather Output ===
    output_parts = [torch.empty_like(local_output) for _ in range(world_size)]
    dist.all_gather(output_parts, local_output)
    full_output = torch.cat(output_parts, dim=-1)  # [1,1,8]

    # === Fake Loss (L2 norm) ===
    loss = (full_output ** 2).mean()
    print(loss.requires_grad, loss.grad_fn)
    loss.backward()

    # === Gradient from full_output: Reduce-Scatter into each output column ===
    #full_output_grad = full_output.grad if full_output.requires_grad else full_output.clone().detach()
    #full_output_grad = full_output.grad if full_output.requires_grad else full_output.clone().detach().requires_grad_(True)
    full_output_grad = full_output.grad if full_output.requires_grad else full_output.clone().requires_grad_(True)
    #full_output_grad = torch.ones_like(full_output)  # simulate dL/dOut

    grad_chunks = list(torch.chunk(full_output_grad, world_size, dim=-1))
    local_grad = torch.empty_like(grad_chunks[0])
    dist.reduce_scatter(local_grad, grad_chunks)

    print(f"[Rank {rank}] Local grad shape: {local_grad.shape}, value: {local_grad}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_tp_attention_with_backward, args=(world_size,), nprocs=world_size)
