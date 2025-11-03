####This is a critical concept in high-performance computing. Here’s the "why" and "how."
####
####1. The Problem: GPUs are Starving
####
####We think of GPUs as massive parallel calculators, but for many operations (like a Transformer FFN), they are actually memory-bound, not compute-bound.
####
####Think of the GPU as a factory:
####
####GPU Cores (ALUs): Thousands of tiny, very fast "workers."
####
####Global Memory (HBM): A massive "warehouse" far away.
####
####On-Chip Memory (SRAM/Registers): A tiny "workbench" right next to each worker.
####
####The un-fused output = relu(linear(x)) operation works like this:
####
####Kernel 1 (Linear):
####
####Workers walk to the "warehouse" (Global Memory) to get x, W, and b.
####
####They walk back to their "workbench" (SRAM/Registers).
####
####They compute y = x @ W + b (very fast).
####
####They walk back to the warehouse to write the intermediate result y down.
####
####Kernel 2 (ReLU):
####
####Workers walk back to the warehouse to pick up y.
####
####They walk back to their "workbench".
####
####They compute z = max(0, y) (very, very fast).
####
####They walk back to the warehouse to write the final result z.
####
####The "workers" (compute) spend almost all their time walking to and from the "warehouse" (memory). This is the memory bandwidth bottleneck.
####
####2. The Solution: Kernel Fusion
####
####Kernel fusion combines these two trips into one. A fused kernel does this:
####
####Fused Kernel (Linear + ReLU):
####
####Workers walk to the "warehouse" to get x, W, and b.
####
####They walk back to their "workbench".
####
####They compute y = x @ W + b (fast).
####
####The result y stays on the workbench (in a register).
####
####They immediately compute z = max(0, y) (fast).
####
####They walk to the "warehouse" only once to write the final result z.
####
####We've eliminated an entire round-trip to global memory, effectively doubling the speed for this operation.
####
####3. How torch.jit.script Does This
####
####When you call torch.jit.script(model), PyTorch's JIT (Just-In-Time) compiler:
####
####Parses your Python forward method into an intermediate representation (IR), creating a computational graph.
####
####Runs an optimizer over this graph.
####
####This optimizer has a list of "fusion rules." It sees the pattern aten::relu (the ReLU operation) immediately following aten::addmm (the linear layer operation).
####
####It says, "I have an optimized, fused kernel for that!"
####
####It replaces those two nodes in the graph with a single node, often called prim::FusionGroup.
####
####When you run scripted_model(x), it calls this single, highly-optimized, fused kernel instead of the two separate ones.
####
####4. Bonus: How You'd Actually Write This (Triton Sketch)
####
####Tools like torch.jit are great, but for ultimate performance, you'd write the kernel yourself. Modern tools like Triton (from OpenAI, now heavily used at NVIDIA) let you do this in Python.
####
####Here is a conceptual sketch of what that Triton kernel would look like. Don't worry about the exact syntax, just notice the logic:
####
####import triton
####import triton.language as tl
####
####
####@triton.jit  # This decorator JIT-compiles this Python to a CUDA kernel
####def fused_linear_relu_kernel(
####        X_ptr, W_ptr, B_ptr, Z_ptr,  # Pointers to tensors in Global Memory
####        M, N, K,  # Matrix dimensions
####        BLOCK_SIZE: tl.constexpr  # A parameter for tuning
####):
####    # This code runs in parallel for one "block" of the output
####
####    # 1. Load data from Global Memory ("warehouse")
####    x_block = tl.load(X_ptr + offsets)
####    w_block = tl.load(W_ptr + offsets)
####    b_block = tl.load(B_ptr + offsets)
####
####    # 2. Compute on-chip ("workbench")
####    # This is all in registers/SRAM
####    y_block = tl.dot(x_block, w_block)  # Matmul
####    y_block = y_block + b_block  # Bias add
####
####    # 3. Apply ReLU *while still on-chip*
####    z_block = tl.maximum(0, y_block)
####
####    # 4. Store *final result* back to Global Memory ("warehouse")
####    tl.store(Z_ptr + offsets, z_block)
####
####
####This sketch directly implements the "fused" logic: Load -> Compute Matmul -> Compute Add -> Compute ReLU -> Store. All compute steps happen in fast on-chip memory before the single write-back.


import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. The Standard PyTorch Module
# This module will be our "before" example.
class FeedForward(nn.Module):
    """
    A simple feed-forward layer:
    output = relu(x @ W.T + b)
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        When run in standard "eager mode", this launches two kernels:
        1. aten::addmm (for the linear layer)
        2. aten::relu  (for the activation)
        """
        y = self.linear(x)
        z = torch.relu(y)
        return z


# --- Example of creating a fused kernel ---
if __name__ == "__main__":

    # --- Setup ---
    B, D_IN, D_OUT = 64, 128, 256

    # Create a model instance
    model = FeedForward(D_IN, D_OUT).eval()  # .eval() for inference

    # Create dummy input data
    x = torch.randn(B, D_IN)

    print("--- 1. Eager Mode (Un-fused) ---")
    print("Running the standard model. This launches two separate kernels.")
    # We can't easily see the kernels, but we know this is how
    # eager mode works.
    out_eager = model(x)
    print("Eager mode execution complete.")

    print("\n--- 2. JIT Script Mode (Fused) ---")
    print("Compiling the model with torch.jit.script...")

    # 2. Create the Fused Version
    # torch.jit.script inspects the `forward` method's Python code
    # and compiles it into a high-performance graph.
    # The JIT compiler is smart enough to fuse common patterns
    # like "linear then relu".
    try:
        scripted_model = torch.jit.script(model)

        print("Compilation successful.")

        # Run the fused model
        out_jit = scripted_model(x)

        # 3. Prove It's Fused
        print("\n--- 3. Proof of Fusion (Inspecting the Graph) ---")
        print("The JIT compiler's optimized graph:")

        # .graph attribute shows the optimized computational graph
        # We are looking for a 'prim::FusionGroup' or 'aten::relu'
        # and 'aten::addmm' (the linear op) to be in the same block.
        print(scripted_model.graph)

        # In the printed graph, you will see that the compiler
        # has taken `self.linear` and `torch.relu` and
        # combined them into a single optimized operation.
        # You might see a `prim::FusionGroup` which explicitly
        # groups operations to be run in a single kernel.

        # Check that outputs are the same
        assert torch.allclose(out_eager, out_jit), "Outputs do not match!"
        print("\nJIT and Eager outputs match.")
        print("This demonstrates that `torch.jit.script` can fuse")
        print("`nn.Linear` and `torch.relu` into a single operation,")
        print("reducing kernel launch overhead and memory bandwidth usage.")

    except Exception as e:
        print(f"\nCould not JIT script the model: {e}")
        print("This can happen with complex Python control flow,")
        print("but for a simple FFN, it should work.")
