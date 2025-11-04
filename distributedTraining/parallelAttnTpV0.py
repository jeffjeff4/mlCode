
####------------------------------------------------------------------------------------------------------------------
####tensor parallel does the column split on Q, K, V separately, but not in this code, why?
####
####You are 100% correct. This is an excellent question that gets to the heart of optimized implementations versus conceptual ones.
####
####You are right that conceptually, Tensor Parallelism splits the W_q, W_k, and W_v matrices.
####
####The code is doing this, but it's doing it in a fused and highly optimized way.
####
####Here is the explanation:
####
####1. The "Conceptual" (Slow) Way
####You could write the code like this, and it would be Column-Parallel:
####
####Python
####
##### --- CONCEPTUAL (SLOWER) IMPLEMENTATION ---
####
##### In __init__:
####self.W_Q_local = torch.randn(d_model, self.d_model_local)
####self.W_K_local = torch.randn(d_model, self.d_model_local)
####self.W_V_local = torch.randn(d_model, self.d_model_local)
####
##### In forward:
##### Three separate, smaller matrix multiplications
####Q_i = X @ self.W_Q_local
####K_i = X @ self.W_K_local
####V_i = X @ self.W_V_local
####This is a "column split" on Q, K, and V separately. It works, but it's inefficient. You are launching three separate matrix multiplication (GEMM) kernels, which has a lot of overhead.
####
####2. The "Fused" (Fast) Way (What Our Code Does)
####This is a standard performance optimization used in all modern Transformer libraries (like Hugging Face, Megatron-LM, etc.).
####
####Step 1: Fuse the Weights (in __init__)
####
####Instead of three small weight matrices, we concatenate them into one big matrix before we even start.
####
####W_QKV_fused = torch.cat([W_Q, W_K, W_V], dim=1)
####
####We then apply the same column-parallel split to this one fused matrix:
####
####self.W_QKV = [W_q_i, W_k_i, W_v_i] (This is our local shard, shape [d_model, 3 * d_model_local])
####
####Step 2: One Big Matmul (in forward)
####
####This is the line you are asking about. We now perform one large, efficient matrix multiplication instead of three small ones.
####
####Python
####
##### This one line...
####QKV_local = X @ self.W_QKV
####...is mathematically identical to:
####
####Python
####
##### ...these three lines:
####QKV_local = X @ [W_q_i, W_k_i, W_v_i]  # (Conceptually)
####QKV_local = [X @ W_q_i, X @ W_k_i, X @ W_v_i]
####QKV_local = [Q_i, K_i, V_i]
####Step 3: Un-fuse the Result
####
####The output QKV_local is a single, wide tensor ([B, S, 3 * d_model_local]) that contains our local Q_i, K_i, and V_i all concatenated together.
####
####This line:
####
####Python
####
####Q_i, K_i, V_i = torch.chunk(QKV_local, 3, dim=-1)
####...simply splits the result back into three separate tensors.
####
####Summary
####You are right: we are splitting Q, K, and V.
####
####The code is just smart about it. It fuses them together, does one big matmul, and then splits the result. This gives the exact same answer as doing three separate splits and matmuls, but it is much faster.
####
########------------------------------------------------------------------------------------------------------------------
####in this code, where we do row split on O matrix?
####def forward(self, X):
####
####This is a great question, and it's the other half of the puzzle.
####
####Just like the "column split," the "row split" is not a command in the forward pass.
####
####It's how the self.W_O (the output projection matrix) was defined in the __init__ function. The forward pass then uses this pre-sharded matrix.
####
####1. Where the "Row Split" Happens (in __init__)
####In the __init__ method (from the full code), we have these lines:
####
##### --- From __init__ ---
####self.d_model_local = ... # e.g., d_model / world_size
####
##### This is the "row split":
##### W_O_local has shape [D_model_local, D_model]
####self.W_O = torch.randn(self.d_model_local, self.d_model)
####
####A "full" W_O matrix would have the shape [d_model, d_model].
####
####By defining self.W_O with the shape [d_model_local, d_model], we have created a "short" matrix that has been split horizontally (row-wise). Each process holds one of these horizontal shards.
####
####2. The Statement That Executes the Row-Parallel Operation (in forward)
####The line in your forward pass that uses this sharded weight is here, in Section 3:
####
##### --- 3. Row-Parallel Forward (W_O) ---
####...
####output_local = self.Z_i_unprojected @ self.W_O
####
####Why this is the answer:
####
####Input is Sharded: The input to this layer, self.Z_i_unprojected, is sharded. Its shape is [B, S, D_model_local]. (It's the output from the previous column-parallel layer).
####
####Weight is Sharded: The weight self.W_O is row-sharded. Its shape is [D_model_local, D_model].
####
####The Math: The matrix multiplication [B, S, D_model_local] @ [D_model_local, D_model] works. It produces a tensor output_local with the full shape [B, S, D_model].
####
####However, this output_local is only a partial result. It's the result of only this process's shard of the input (Z_i) multiplied by only this process's shard of the weight (W_O_i).
####
####The true final output is the sum of all these partial outputs from all processes.
####
####That is why the very next line is the key communication step for Row-Parallelism:
####
####dist.all_reduce(output_local, op=dist.ReduceOp.SUM)
####
####This all_reduce call sums all the output_local partial results, and after this line, all processes have the identical, correct final output.
####
####
########------------------------------------------------------------------------------------------------------------------
####this code using tensor parallel, but which statement does the column split for tensor parallel?
####
####This is a fantastic question, as it gets to the core of how this works.
####
####The "column split" is not a function call in the forward pass. Instead, it's a fundamental part of the design of the model's weights in the __init__ function.
####
####The forward pass then uses these pre-sharded weights.
####
####Here is the breakdown:
####
####1. Where the "Split" Actually Happens (in __init__)
####
####The "column split" happens when we define self.W_QKV. In the __init__ method (which you can see in the full code), we have these lines:
####
####Python
####
##### --- From __init__ ---
####self.num_heads_local = num_heads // world_size
####self.d_head = d_model // num_heads
####self.d_model_local = self.num_heads_local * self.d_head
####
##### This is the "column split":
####self.W_QKV = torch.randn(d_model, 3 * self.d_model_local)
####A "full" W_QKV matrix would have the shape [d_model, 3 * d_model].
####
####By defining self.W_QKV with the shape [d_model, 3 * d_model_local], we are creating a shard of the full matrix. This is the "column" that this process is responsible for.
####
####2. The Statement That Executes the Column-Parallel Operation (in forward)
####
####The line in your forward pass that uses this sharded weight is here:
####
####Python
####
##### --- 1. Column-Parallel Forward (W_QKV) ---
##### X @ W_QKV_local -> [B, S, 3 * D_model_local]
##### This is a local operation.
####QKV_local = X @ self.W_QKV
####Why this is the answer:
####
####X has the full shape [batch_size, seq_len, d_model]. It is identical on all processes.
####
####self.W_QKV has the sharded shape [d_model, 3 * d_model_local].
####
####The result, QKV_local, therefore has the sharded shape [batch_size, seq_len, 3 * d_model_local].
####
####This single matrix multiplication X @ self.W_QKV performs the core "Column-Parallel" operation. It takes the full input X and multiplies it by its local column of the weights (self.W_QKV) to produce its local shard of the output (QKV_local).
####
####Critically, this line does not require any communication. No all_reduce or all_gather is needed here, which is the whole point of a Column-Parallel forward pass.


####------------------------------------------------------------------------------------------------------------------
####please explain the formulas in this code:
####def backward(self, dOutput):
####
####This is an excellent piece of code to analyze. It's the "heart" of Tensor Parallelism, combining matrix calculus (backpropagation) with distributed communication.
####
####Let's walk through the backward function step-by-step, explaining the why behind each formula.
####
####Core Concept: Backpropagation is just the chain rule applied in reverse order of the forward pass.
####
####Forward Pass Summary: X -> QKV_local -> (Q,K,V)_i -> scores_i -> Weights_i -> Z_i -> output_local -> all_reduce -> Output
####
####Backward Pass Summary: dOutput -> d(output_local) -> dZ_i -> dWeights_i -> dScores_i -> d(Q,K,V)_i -> dQKV_local -> dX (with an all_reduce).
####
####The input to this function is dOutput, which is the gradient of the final Loss with respect to the Output (i.e., dL/dOutput).
####
####1. Row-Parallel Backward (for W_O)
####This section backpropagates through the last layer of the module, the Row-Parallel layer W_O.
####
####Forward Pass Recall: The forward pass for this layer was: output_local = self.Z_i_unprojected @ self.W_O, followed by an all_reduce sum. So, the full operation is: Output = (Z_0 @ W_0) + (Z_1 @ W_1) + ...
####
####We have dOutput (dL/dOutput) and want two things:
####
####The gradient for the weights, dL/dW_O (for our local W_O).
####
####The gradient to pass backward, dL/dZ_i (for our local Z_i).
####
####Formula 1: self.grad_W_O = Z_i_2d.T @ dOutput_2d
####
####Goal: Calculate the gradient for our local weight shard, dL/dW_O.
####
####Calculus (Chain Rule):
####
####dL/dW_O = (dL/dOutput) * (dOutput / dW_O)
####
####Because Output = ... + (Z_i @ W_O) + ...
####
####The derivative dOutput / dW_O is just Z_i.T.
####
####Therefore, dL/dW_O = Z_i.T @ (dL/dOutput).
####
####Code: self.grad_W_O = Z_i_2d.T @ dOutput_2d
####
####This is the exact implementation of that formula. The .view(-1, ...) (e.g., Z_i_2d) is just a "reshape" to turn the 3D tensors ([B, S, D]) into 2D matrices ([B*S, D]) so we can use a single, efficient matrix multiplication.
####
####Formula 2: dZ_i = dOutput @ self.W_O.T
####
####Goal: Calculate the gradient to pass backward to the self-attention mechanism, dL/dZ_i.
####
####Calculus (Chain Rule):
####
####dL/dZ_i = (dL/dOutput) * (dOutput / dZ_i)
####
####Because Output = ... + (Z_i @ W_O) + ...
####
####The derivative dOutput / dZ_i is W_O.T.
####
####Therefore, dL/dZ_i = (dL/dOutput) @ W_O.T.
####
####Code: dZ_i = dOutput @ self.W_O.T
####
####This is a direct, one-to-one match with the formula. No communication is needed here because dOutput is already identical on all processes.
####
####2. Local Self-Attention Backward
####Now we have dZ_i, the gradient for the output of the local attention heads. We need to backpropagate through the attention mechanism itself.
####
####Forward Pass Recall: Z_i_heads = self.Weights_i @ self.V_i
####
####Formula 3 & 4: dWeights_i = ... and dV_i = ...
####
####Goal: We have dZ_i_heads (dL/dZ_i_heads). We need dL/dWeights_i and dL/dV_i.
####
####Calculus (MatMul Backward): For a simple Z = W @ V:
####
####dL/dW = (dL/dZ) @ V.T
####
####dL/dV = W.T @ (dL/dZ)
####
####Code:
####
####dWeights_i = dZ_i_heads @ self.V_i.transpose(-1, -2)
####
####dV_i = self.Weights_i.transpose(-1, -2) @ dZ_i_heads
####
####This is a perfect implementation of the matrix multiplication backward pass.
####
####Formula 5: dScores_i = self.Weights_i * (dWeights_i - (dWeights_i * self.Weights_i).sum(dim=-1, keepdim=True))
####
####Goal: We have dWeights_i (dL/dWeights_i). We need dScores_i (dL/dScores_i).
####
####Forward Pass Recall: Weights_i = torch.softmax(scores_i)
####
####Calculus (Softmax Backward): This is the most complex derivative. The gradient of a softmax is famously tricky. A stable and correct implementation is:
####
####dScores = Weights * (dWeights - sum(dWeights * Weights))
####
####(Where * is element-wise multiplication and sum is along the softmax dimension).
####
####Code: The code implements this exactly.
####
####dWeights_i_stable = dWeights_i - (dWeights_i * self.Weights_i).sum(dim=-1, keepdim=True)
####
####dScores_i = self.Weights_i * dWeights_i_stable
####
####Formula 6: dScores_i = dScores_i / math.sqrt(self.d_head)
####
####Goal: We have the gradient after scaling. We need the gradient before scaling.
####
####Forward Pass Recall: scores_i = (Q_i @ K_i.T) / sqrt(d_h)
####
####Calculus (Scaling Backward): For Y = X / c:
####
####dL/dX = (dL/dY) * (dY / dX) = (dL/dY) / c
####
####Code: dScores_i = dScores_i / math.sqrt(self.d_head)
####
####This is a direct implementation. We just divide the incoming gradient by the same constant we used in the forward pass.
####
####Formula 7 & 8: dQ_i = ... and dK_i = ...
####
####Goal: We have dScores_i (dL/dScores_i). We need dL/dQ_i and dL/dK_i.
####
####Forward Pass Recall: scores_i = Q_i @ K_i.transpose(-1, -2)
####
####Calculus (MatMul Backward): For S = Q @ K.T:
####
####dL/dQ = (dL/dS) @ K
####
####dL/dK = (dL/dS).T @ Q
####
####Code:
####
####dQ_i = dScores_i @ self.K_i
####
####dK_i = dScores_i.transpose(-1, -2) @ self.Q_i
####
####This is a perfect match to the calculus.
####
####3. Column-Parallel Backward (for W_QKV)
####We have now backpropagated through the attention logic. We have dQ_i, dK_i, and dV_i. We must pass these gradients back through the first layer, the Column-Parallel layer W_QKV.
####
####Forward Pass Recall: QKV_local = self.X_cache @ self.W_QKV
####
####Formula 9: self.grad_W_QKV = X_2d.T @ dQKV_local_2d
####
####Goal: Calculate the gradient for our local weight shard, dL/dW_QKV.
####
####Setup: We first torch.cat dQ, dK, and dV into dQKV_local (dL/dQKV_local).
####
####Calculus (Chain Rule):
####
####dL/dW_QKV = (dL/dQKV_local) * (dQKV_local / dW_QKV)
####
####The derivative dQKV_local / dW_QKV is X_cache.T.
####
####Therefore, dL/dW_QKV = X_cache.T @ (dL/dQKV_local).
####
####Code: self.grad_W_QKV = X_2d.T @ dQKV_local_2d
####
####This is the exact formula, just reshaped to 2D for a single matmul.
####
####Formula 10: dX_local_2d = dQKV_local_2d @ self.W_QKV.T
####
####Goal: Calculate the gradient to pass backward to the input, dL/dX.
####
####Calculus (Chain Rule):
####
####dL/dX = (dL/dQKV_local) * (dQKV_local / dX)
####
####The derivative dQKV_local / dX is W_QKV.T.
####
####Therefore, dL/dX = (dL/dQKV_local) @ W_QKV.T.
####
####Code: dX_local_2d = dQKV_local_2d @ self.W_QKV.T
####
####This is a direct match. However, this is a local gradient.
####
####Why? In the forward pass, X was multiplied by W_QKV = [W_0, W_1, ...]. The full dL/dX is the sum of the gradients from each of these parallel operations.
####
####dL/dX = (dL/dQKV_0 @ W_0.T) + (dL/dQKV_1 @ W_1.T) + ...
####
####dX_local_2d is just one term in that sum.
####
####Formula 11: dist.all_reduce(dX_local_2d, op=dist.ReduceOp.SUM)
####
####Goal: Sum all the local dX gradients to get the final, correct dX gradient.
####
####Calculus: This implements the sum: dX = dX_0 + dX_1 + ...
####
####Code: This is the key communication step in the Column-Parallel backward pass. After this line, dX_local_2d on all processes holds the same, correct, and complete gradient for the input X.

####------------------------------------------------------------------------------------------------------------------
####in the backward() function,
####Z_i_2d = self.Z_i_unprojected.view(-1, self.d_model_local)  # [B*S, D_local]
####
####does this self.Z_i_unprojected mean ONLY one shard of (Z_i @ W_O), or the reduced result of all of (Z_i @ W_O)?
####
####That is a fantastic question, and it's the single most important one for understanding the backward pass.
####
####You are correct. self.Z_i_unprojected is ONLY the local shard.
####
####It is NOT the reduced result. It is the local input to the self.W_O matrix multiply, which we cached during the forward pass.
####
####Let's trace it in the forward function:
####Step 2 (Local Attention): We calculate the local attention output Z_i_heads. We then reshape it into:
####
####Python
####
####self.Z_i_unprojected = Z_i_heads.transpose(1, 2).reshape(...)
####At this moment, self.Z_i_unprojected is local data. Its shape is [B, S, D_model_local]. It only has the data for the heads on this process.
####
####Step 3 (Row-Parallel): We use this local shard as the input to our local weight shard (self.W_O):
####
####Python
####
####output_local = self.Z_i_unprojected @ self.W_O
####output_local is now a partial result.
####
####Communication: We then sum everyone's partial results:
####
####Python
####
####dist.all_reduce(output_local, op=dist.ReduceOp.SUM)
####The final, reduced result is what's left in the output_local variable, which is then returned by the forward function.
####
####We do not save this reduced result. We saved self.Z_i_unprojected (the input from Step 2) precisely because we need it for the backward pass.
####
####Why We Need It in backward()
####In backward(), we are calculating the gradient for our local weight self.W_O.
####
####The formula for the gradient of output_local = self.Z_i_unprojected @ self.W_O is:
####
####grad_W_O = self.Z_i_unprojected.T @ dOutput
####
####To calculate our local gradient, we must have our original local input, self.Z_i_unprojected. That is exactly why we saved it.
####
########------------------------------------------------------------------------------------------------------------------
####could you please derive the formula for this?
####dScores_i = self.Weights_i * (dWeights_i - (dWeights_i * self.Weights_i).sum(dim=-1, keepdim=True))



####------------------------------------------------------------------------------------------------------------------

####------------------------------------------------------------------------------------------------------------------

####------------------------------------------------------------------------------------------------------------------

####------------------------------------------------------------------------------------------------------------------

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import math


def setup(rank, world_size):
    """Initializes the distributed process group (for CPU)."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # Any free port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Initialized process {rank}/{world_size} on 'gloo' backend.")


def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()


class ManualTensorParallelAttention:
    """
    Implements a single head of self-attention with manual backprop
    and "Megatron-LM" style Tensor Parallelism.

    This implementation splits the attention heads across processes.
    - W_QKV is a "Column-Parallel" Linear layer.
    - W_O is a "Row-Parallel" Linear layer.
    """

    def __init__(self, d_model, num_heads, rank, world_size):
        # Ensure world_size divides num_heads
        assert num_heads % world_size == 0, "Num heads must be divisible by world_size"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.rank = rank
        self.world_size = world_size

        # --- Local dimensions ---
        self.num_heads_local = num_heads // world_size
        self.d_head = d_model // num_heads
        self.d_model_local = self.num_heads_local * self.d_head

        # --- 1. Column-Parallel Layer (W_QKV) ---
        # Each rank holds only its "shard" of the weights.
        # Input (X) has shape [B, S, D_model]
        # W_QKV_local has shape [D_model, 3 * D_model_local]
        self.W_QKV = torch.randn(d_model, 3 * self.d_model_local)

        # --- 2. Row-Parallel Layer (W_O) ---
        # Each rank holds only its "shard" of the weights.
        # Input (Z_i) has shape [B, S, D_model_local]
        # W_O_local has shape [D_model_local, D_model]
        self.W_O = torch.randn(self.d_model_local, d_model)

        # --- Gradients (manual) ---
        self.grad_W_QKV = torch.zeros_like(self.W_QKV)
        self.grad_W_O = torch.zeros_like(self.W_O)

        # --- Intermediate values for backprop ---
        self.X_cache = None  # [B, S, D_model]
        self.Q_i = None  # [B, num_heads_local, S, D_head]
        self.K_i = None  # [B, num_heads_local, S, D_head]
        self.V_i = None  # [B, num_heads_local, S, D_head]
        self.Weights_i = None  # [B, num_heads_local, S, S]
        self.Z_i_unprojected = None  # [B, S, D_model_local]

    def forward(self, X):
        """
        Manually implements the forward pass for Tensor-Parallel Attention.
        X shape: [batch_size, seq_len, d_model]
        """
        # Cache X for backward pass
        self.X_cache = X
        batch_size, seq_len, _ = X.shape

        # --- 1. Column-Parallel Forward (W_QKV) ---
        # X @ W_QKV_local -> [B, S, 3 * D_model_local]
        # This is a local operation.
        QKV_local = X @ self.W_QKV

        # Split into local Q, K, V
        # Each has shape [B, S, D_model_local]
        Q_i, K_i, V_i = torch.chunk(QKV_local, 3, dim=-1)

        # Reshape for multi-head attention
        # [B, S, D_model_local] -> [B, S, num_heads_local, d_head] -> [B, num_heads_local, S, d_head]
        self.Q_i = Q_i.view(batch_size, seq_len, self.num_heads_local, self.d_head).transpose(1, 2)
        self.K_i = K_i.view(batch_size, seq_len, self.num_heads_local, self.d_head).transpose(1, 2)
        self.V_i = V_i.view(batch_size, seq_len, self.num_heads_local, self.d_head).transpose(1, 2)

        # --- 2. Local Self-Attention ---
        # Calculate scores for local heads
        # [B, h_i, S, d_h] @ [B, h_i, d_h, S] -> [B, h_i, S, S]
        scores_i = (self.Q_i @ self.K_i.transpose(-1, -2)) / math.sqrt(self.d_head)

        # Softmax (local)
        self.Weights_i = torch.softmax(scores_i, dim=-1)

        # Apply weights to V (local)
        # [B, h_i, S, S] @ [B, h_i, S, d_h] -> [B, h_i, S, d_h]
        Z_i_heads = self.Weights_i @ self.V_i

        # Combine heads
        # [B, h_i, S, d_h] -> [B, S, h_i, d_h] -> [B, S, D_model_local]
        self.Z_i_unprojected = Z_i_heads.transpose(1, 2).reshape(batch_size, seq_len, self.d_model_local)

        # --- 3. Row-Parallel Forward (W_O) ---
        # [B, S, D_model_local] @ [D_model_local, D_model] -> [B, S, D_model]
        # This is a local operation, resulting in a partial output.
        output_local = self.Z_i_unprojected @ self.W_O

        # *** TENSOR PARALLEL COMMUNICATION (Forward) ***
        # We must sum the partial outputs from all processes.
        # This all_reduce is the key to Row-Parallelism.
        dist.all_reduce(output_local, op=dist.ReduceOp.SUM)

        # Now, output_local on all processes holds the correct final output
        return output_local

    def backward(self, dOutput):
        """
        Manually implements the backward pass.
        dOutput shape: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = dOutput.shape

        # --- 1. Row-Parallel Backward (W_O) ---
        # dOutput is identical on all processes.

        # Gradient for W_O (local)
        # [D_model_local, S, B] @ [B, S, D_model] -> [D_model_local, D_model]
        # (Need to reshape for 2D matmul)
        Z_i_2d = self.Z_i_unprojected.view(-1, self.d_model_local)  # [B*S, D_local]
        dOutput_2d = dOutput.view(-1, self.d_model)
        self.grad_W_O = Z_i_2d.T @ dOutput_2d

        # Gradient for Z_i (local)
        # [B, S, D_model] @ [D_model, D_model_local] -> [B, S, D_model_local]
        dZ_i = dOutput @ self.W_O.T

        # --- 2. Local Self-Attention Backward ---
        # Reshape dZ_i back to heads
        # [B, S, D_model_local] -> [B, S, h_i, d_h] -> [B, h_i, S, d_h]
        dZ_i_heads = dZ_i.view(batch_size, seq_len, self.num_heads_local, self.d_head).transpose(1, 2)

        # Backprop: Z_i_heads = self.Weights_i @ self.V_i
        # dWeights_i: [B, h_i, S, S]
        dWeights_i = dZ_i_heads @ self.V_i.transpose(-1, -2)
        # dV_i: [B, h_i, S, d_h]
        dV_i = self.Weights_i.transpose(-1, -2) @ dZ_i_heads

        # Backprop: self.Weights_i = torch.softmax(scores_i, dim=-1)
        # This is the tricky one.
        # dS = W * (dW - sum(dW * W))
        dWeights_i_stable = dWeights_i - (dWeights_i * self.Weights_i).sum(dim=-1, keepdim=True)
        dScores_i = self.Weights_i * dWeights_i_stable

        # Backprop: scores_i = (Q_i @ K_i.T) / sqrt(d_h)
        dScores_i = dScores_i / math.sqrt(self.d_head)

        # dQ_i: [B, h_i, S, d_h]
        dQ_i = dScores_i @ self.K_i
        # dK_i: [B, h_i, S, d_h]
        dK_i = dScores_i.transpose(-1, -2) @ self.Q_i

        # --- 3. Column-Parallel Backward (W_QKV) ---
        # Combine gradients for Q, K, V
        # Un-reshape from heads [B, h_i, S, d_h] -> [B, S, D_model_local]
        dQ = dQ_i.transpose(1, 2).reshape(batch_size, seq_len, self.d_model_local)
        dK = dK_i.transpose(1, 2).reshape(batch_size, seq_len, self.d_model_local)
        dV = dV_i.transpose(1, 2).reshape(batch_size, seq_len, self.d_model_local)

        # Concatenate into one gradient tensor
        # [B, S, 3 * D_model_local]
        dQKV_local = torch.cat([dQ, dK, dV], dim=-1)

        # Gradient for W_QKV (local)
        # [D_model, S, B] @ [B, S, 3*D_local] -> [D_model, 3*D_local]
        # (Reshape for 2D matmul)
        X_2d = self.X_cache.view(-1, self.d_model)  # [B*S, D_model]
        dQKV_local_2d = dQKV_local.view(-1, 3 * self.d_model_local)  # [B*S, 3*D_local]
        self.grad_W_QKV = X_2d.T @ dQKV_local_2d

        # Gradient for X (local, partial)
        # [B*S, 3*D_local] @ [3*D_local, D_model] -> [B*S, D_model]
        dX_local_2d = dQKV_local_2d @ self.W_QKV.T

        # *** TENSOR PARALLEL COMMUNICATION (Backward) ***
        # The gradient for X was split across processes. We must sum them.
        # This all_reduce is the key to Column-Parallelism's backward pass.
        dist.all_reduce(dX_local_2d, op=dist.ReduceOp.SUM)

        # Reshape back to original X shape
        return dX_local_2d.view_as(self.X_cache)

    def update_weights(self, lr):
        """Manually update weights using gradients."""
        self.W_QKV -= self.grad_W_QKV * lr
        self.W_O -= self.grad_W_O * lr

    def zero_grad(self):
        """Manually zero gradients."""
        self.grad_W_QKV.zero_()
        self.grad_W_O.zero_()

    def sync_initial_weights(self, src=0):
        """Ensure all processes start with the same weights."""
        dist.broadcast(self.W_QKV, src=src)
        dist.broadcast(self.W_O, src=src)


# --- Data Generation and Training Loop ---

def get_simple_dataset(n_samples, seq_len, d_model):
    """Generates a simple dataset. The task is auto-encoding."""
    X = torch.randn(n_samples, seq_len, d_model)
    Y = X.clone()  # Target is to reconstruct the input
    return torch.utils.data.TensorDataset(X, Y)


def main_worker(rank, world_size, epochs=5, batch_size=4, d_model=32, num_heads=4):
    """
    The main worker function.
    'rank' and 'world_size' are passed by mp.spawn().
    """
    setup(rank, world_size)

    # Model parameters
    seq_len = 8
    lr = 0.01

    # --- 1. Create Model ---
    model = ManualTensorParallelAttention(d_model, num_heads, rank, world_size)

    # *** TENSOR PARALLEL COMMUNICATION (Setup) ***
    # Must broadcast initial weights so all models are identical
    model.sync_initial_weights(src=0)

    # --- 2. Create Dataset and Dataloader ---
    # Create the full dataset, but only rank 0 prints
    if rank == 0:
        print(f"Generating dataset: {batch_size * 10} samples, {seq_len} seq_len, {d_model} d_model")
    dataset = get_simple_dataset(n_samples=batch_size * 10, seq_len=seq_len, d_model=d_model)

    # DistributedSampler ensures each rank gets a unique, non-overlapping
    # slice of the data.
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    # --- 3. Training Loop ---
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Important for shuffling
        total_loss = 0.0

        for X_batch, Y_batch in loader:
            # --- Forward Pass ---
            model.zero_grad()
            Y_pred = model.forward(X_batch)

            # --- Manual Loss (MSE) ---
            # All ranks have the same Y_pred and Y_batch,
            # so all compute the same loss.
            loss = ((Y_pred - Y_batch) ** 2).mean()
            total_loss += loss.item()

            # --- Manual Backward Pass ---
            # Calculate the initial gradient (derivative of MSE loss)
            # dOutput = dLoss/dY_pred = 2 * (Y_pred - Y_batch) / N
            dOutput = 2.0 * (Y_pred - Y_batch) / Y_batch.numel()

            model.backward(dOutput)

            # --- Update Weights ---
            model.update_weights(lr)

        # Only rank 0 should print the loss
        if rank == 0:
            print(f"Epoch {epoch} | Avg Loss: {total_loss / len(loader):.6f}")

    # --- 4. Evaluation (Example) ---
    #
    # *** THIS IS THE CORRECTED SECTION ***
    #

    # Sync all processes before evaluation
    dist.barrier()
    if rank == 0:
        print("\n--- Evaluation ---")

    # 1. Create empty containers on ALL processes
    X_eval = torch.zeros(batch_size, seq_len, d_model)
    Y_eval = torch.zeros(batch_size, seq_len, d_model)

    # 2. Rank 0 will get a batch of data and fill the containers
    if rank == 0:
        # Use try-except in case loader is empty
        try:
            X_eval_data, Y_eval_data = next(iter(loader))
            X_eval.copy_(X_eval_data)
            Y_eval.copy_(Y_eval_data)

        except StopIteration:
            print("Eval loader empty, using zeros.")

    # 3. Broadcast the data from Rank 0 to ALL other processes
    dist.broadcast(X_eval, src=0)
    dist.broadcast(Y_eval, src=0)

    # 4. ALL processes MUST participate in the forward pass,
    #    as it contains an all_reduce operation.
    with torch.no_grad():
        Y_pred = model.forward(X_eval)
        # Loss is now computed identically on all processes
        eval_loss = ((Y_pred - Y_eval) ** 2).mean()

    # 5. Only Rank 0 prints the final result
    if rank == 0:
        print(f"Final Eval Loss: {eval_loss.item():.6f}")
        # print(f"Input [0,0]: {X_eval[0, 0, :4].numpy()}")
        # print(f"Pred  [0,0]: {Y_pred[0, 0, :4].numpy()}")

    # 6. ALL processes must call cleanup
    cleanup()


if __name__ == "__main__":
    world_size = 2  # <-- CRITICAL: Must match num_heads or be a divisor
    num_heads = 4  # e.g., 4 heads, 2 processes -> 2 heads/process
    d_model = 32  # e.g., 32 d_model, 4 heads -> 8 d_head

    # We must use 'spawn' to launch the processes
    print(f"Starting {world_size} processes for Tensor Parallelism...")
    mp.spawn(
        main_worker,
        args=(world_size, 5, 4, d_model, num_heads),
        nprocs=world_size,
        join=True
    )
    print("All processes finished.")
