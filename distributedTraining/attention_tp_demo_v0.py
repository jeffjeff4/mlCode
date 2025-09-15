####compare below 2 codes
####questions:
####1. why code1, loss.backward() runs correctly,
####1) loss.require_grad=True
####2) loss.grad_fn is not None
####
####but why  code2, loss.backward() runs not correctly,
####1) loss.require_grad=False
####2) loss.grad_fn is None
####
####part1, code1
#####################################################################
######correct version
#####################################################################
####
##### filename: attention_tp_backward_fix_gloo.py
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
####        ctx.world_size = dist.get_world_size()
####
####        local_output = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
####        dist.all_gather(local_output, tensor)
####        full_output = torch.cat(local_output, dim=-1)
####
####        return full_output
####
####    @staticmethod
####    def backward(ctx, grad_output):
####        # We cannot use dist.reduce_scatter with Gloo.
####        # Instead, we perform an all_reduce and then chunk the result.
####
####        # 1. All-Reduce the gradient from all processes.
####        # This will give every process the sum of gradients from all chunks.
####        all_reduced_grad = torch.empty_like(grad_output)
####        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, async_op=False)
####        all_reduced_grad.copy_(grad_output)  # copy the result back
####
####        # 2. Split the full summed gradient into chunks and select the local one.
####        grad_chunks = list(torch.chunk(all_reduced_grad, ctx.world_size, dim=-1))
####
####        # 3. Each rank returns its corresponding chunk as its local gradient.
####        local_grad = grad_chunks[dist.get_rank()]
####
####        return local_grad
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
####    # The gradients should now be correctly computed on the weights
####    print(f"[Rank {rank}] q_weight.grad shape: {q_weight.grad.shape}, sum: {q_weight.grad.sum()}")
####    print(
####        f"[Rank {rank}] output_proj_weight.grad shape: {output_proj_weight.grad.shape}, sum: {output_proj_weight.grad.sum()}")
####
####    cleanup()
####
####
####if __name__ == "__main__":
####    world_size = 2
####    mp.spawn(run_tp_attention_with_backward, args=(world_size,), nprocs=world_size)
####
####part2, code2
####
##### filename: attention_tp_backward.py
##### can not run, can not be used in interview
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
####    # mac does not have nccl, using gloo
####    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
####    dist.init_process_group("gloo", rank=rank, world_size=world_size)
####    # torch.cuda.set_device(rank)
####    device = torch.device("cpu")
####
####
####def cleanup():
####    dist.destroy_process_group()
####
####
####def run_tp_attention_with_backward(rank, world_size):
####    setup(rank, world_size)
####    # device = torch.device(f"cuda:{rank}")
####    device = torch.device(f"cpu:{rank}")
####
####    # === Input ===
####    batch, seq_len, hidden_size = 1, 1, 8
####    split_size = hidden_size // world_size
####    input_tensor = torch.randn(batch, seq_len, hidden_size, device=device, requires_grad=True)
####
####    # === Linear Weights ===
####    q_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####    k_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####    v_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####
####    # === Q/K/V projections (Column Parallel) ===
####    q_part = input_tensor @ q_weight  # [1, 1, 4]
####    k_part = input_tensor @ k_weight
####    v_part = input_tensor @ v_weight
####
####    # === All-Gather Q/K/V ===
####    q_list = [torch.empty_like(q_part) for _ in range(world_size)]
####    k_list = [torch.empty_like(k_part) for _ in range(world_size)]
####    v_list = [torch.empty_like(v_part) for _ in range(world_size)]
####
####    dist.all_gather(q_list, q_part)
####    dist.all_gather(k_list, k_part)
####    dist.all_gather(v_list, v_part)
####
####    Q = torch.cat(q_list, dim=-1)
####    K = torch.cat(k_list, dim=-1)
####    V = torch.cat(v_list, dim=-1)
####
####    # === Attention Computation ===
####    attn_scores = Q @ K.transpose(-2, -1) / (hidden_size ** 0.5)
####    attn_probs = F.softmax(attn_scores, dim=-1)
####    attn_output = attn_probs @ V  # [1,1,8]
####
####    # === Output Projection (Row Parallel) ===
####    output_proj_weight = torch.randn(hidden_size, split_size, device=device, requires_grad=True)
####    local_output = attn_output @ output_proj_weight  # [1,1,4]
####
####    # === All-Gather Output ===
####    output_parts = [torch.empty_like(local_output) for _ in range(world_size)]
####    dist.all_gather(output_parts, local_output)
####    full_output = torch.cat(output_parts, dim=-1)  # [1,1,8]
####
####    # === Fake Loss (L2 norm) ===
####    loss = (full_output ** 2).mean()
####    print(loss.requires_grad, loss.grad_fn)
####    loss.backward()
####
####    # === Gradient from full_output: Reduce-Scatter into each output column ===
####    # full_output_grad = full_output.grad if full_output.requires_grad else full_output.clone().detach()
####    # full_output_grad = full_output.grad if full_output.requires_grad else full_output.clone().detach().requires_grad_(True)
####    full_output_grad = full_output.grad if full_output.requires_grad else full_output.clone().requires_grad_(True)
####    # full_output_grad = torch.ones_like(full_output)  # simulate dL/dOut
####
####    grad_chunks = list(torch.chunk(full_output_grad, world_size, dim=-1))
####    local_grad = torch.empty_like(grad_chunks[0])
####    dist.reduce_scatter(local_grad, grad_chunks)
####
####    print(f"[Rank {rank}] Local grad shape: {local_grad.shape}, value: {local_grad}")
####
####    cleanup()
####
####
####if __name__ == "__main__":
####    world_size = 2
####    mp.spawn(run_tp_attention_with_backward, args=(world_size,), nprocs=world_size)
####
####这是一个非常棒的对比，它揭示了PyTorch计算图构建和分布式训练中的关键细节。让我们来分析为什么Code1能正确运行而Code2不能。
####
####核心问题分析
####Code1
####能正确运行的原因：
####loss.requires_grad = True：这是因为在Code1中，AllGather是一个自定义的torch.autograd.Function，它在
####forward
####和
####backward
####中都正确地维护了计算图。
####
####loss.grad_fn is not None：这是因为整个计算过程都在PyTorch的计算图中：
####
####AllGather.apply()
####会记录操作到计算图中
####
####所有的矩阵运算（
####
####@
####
####, softmax, mean）都会自动记录梯度计算
####
####因此
####loss
####有一个完整的反向传播路径
####
####Code2
####不能正确运行的原因：
####loss.requires_grad = False：这是因为Code2使用了dist.all_gather()，这是一个非自动微分操作，它切断了计算图。
####
####loss.grad_fn is None：同样是因为计算图被切断，PyTorch无法构建反向传播路径。
####
####详细对比
####计算图构建方式对比
####特性                          Code1(正确)                                   Code2(错误)
####All - Gather实现              自定义torch.autograd.Function                  直接使用dist.all_gather()
####计算图连续性                      ✅ 保持完整                                  ❌ 被切断
####梯度传播                        自动通过backward方法                          需要手动处理
####requires_grad                   ✅ True                                      ❌ False
####grad_fn                         ✅ 存在                                       ❌ None
####Code1
####的关键优势：AllGather
####自定义函数
####python
####
####
####class AllGather(torch.autograd.Function):
####    @staticmethod
####    def forward(ctx, tensor):
####        # ... 前向传播实现
####        return full_output
####
####    @staticmethod
####    def backward(ctx, grad_output):
####        # ... 反向传播实现
####        return local_grad
####
####
####这个自定义函数：
####
####在forward中记录必要信息（如world_size）
####
####在backward中实现正确的梯度计算
####
####保持计算图的连续性
####
####Code2
####的问题：计算图断裂
####在Code2中：
####
####python
##### 这里切断了计算图！
####q_list = [torch.empty_like(q_part) for _ in range(world_size)]
####dist.all_gather(q_list, q_part)  # ⚠️ 非自动微分操作
####Q = torch.cat(q_list, dim=-1)  # 从这时起，Q与之前的计算脱节
####具体问题定位
####Code2
####中的问题点：
####第58 - 64
####行：使用dist.all_gather()
####收集张量
####
####第78 - 81
####行：再次使用dist.all_gather()
####收集输出
####
####第85行：loss = (full_output ** 2).mean() - 此时full_output已经脱离计算图
####
####错误的具体表现：
####python
####print(loss.requires_grad, loss.grad_fn)  # 输出: False, None
####loss.backward()  # ❌ 失败：因为loss没有grad_fn，不知道如何反向传播
####为什么会有这样的差异？
####PyTorch
####的自动微分机制
####PyTorch的自动微分需要：
####
####计算图连续性：所有操作都必须是PyTorch知道如何微分的
####
####requires_grad = True：输入张量需要梯度追踪
####
####grad_fn：每个操作都要记录如何计算梯度
####
####dist.all_gather()
####的问题
####dist.all_gather()
####是一个分布式通信原语，不是PyTorch自动微分系统的一部分。它：
####
####不记录计算图
####
####不提供梯度计算
####
####会切断requires_grad的传播
####
####解决方案
####要让Code2工作，你需要：
####
####方案1：使用自定义Function（像Code1那样）
####这是推荐的做法，因为它保持了计算图的完整性。
####
####方案2：手动计算梯度（复杂且容易出错）
####python
####
####
##### 伪代码：手动实现反向传播
####def manual_backward():
####    # 1. 手动计算 full_output 的梯度
####    full_output_grad = 2 * full_output / full_output.numel()
####
####    # 2. 手动实现 all_gather 的反向（reduce_scatter）
####    output_grad_chunks = list(torch.chunk(full_output_grad, world_size, dim=-1))
####    local_grad = torch.empty_like(output_grad_chunks[0])
####    dist.reduce_scatter(local_grad, output_grad_chunks)
####
####    # 3. 继续手动计算更前面的梯度...
####    # ... 非常复杂！
####
####
####方案3：使用Gradient
####Checkpointing（高级技巧）
####python
##### 使用 checkpoint 来避免计算图断裂
####from torch.utils.checkpoint import checkpoint
####
####
####def custom_all_gather_with_checkpoint(tensor):
####    # 使用checkpoint包装非自动微分操作
####    return checkpoint(lambda x: _real_all_gather(x), tensor)
####
####
####def _real_all_gather(tensor):
####    # 实际的all_gather实现
####    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
####    dist.all_gather(tensor_list, tensor)
####    return torch.cat(tensor_list, dim=-1)
####
####
####总结
####方面                  Code1                  Code2
####计算图                 完整                      断裂
####自动微分                ✅ 支持                ❌ 不支持
####实现复杂度               中等                  低（但功能不全）
####维护性                 好                       差
####推荐度                 ⭐⭐⭐⭐⭐               ⭐
####
####根本原因：Code2使用了原生的dist.all_gather()，这是一个非自动微分操作，切断了PyTorch的计算图，导致loss无法反向传播。
####
####Code1通过自定义torch.autograd.Function正确地实现了前向和反向传播，保持了计算图的完整性，因此可以正常工作。
####
####这就是为什么在需要自动微分的分布式训练中，我们通常需要为通信操作编写自定义的autograd Function。

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