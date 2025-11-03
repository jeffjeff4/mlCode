####Here's the conceptual breakdown of what we just did and why it's so important.
####
####1. Data Representation (FP32 vs. INT8)
####
####The core idea is a trade-off between precision and size.
####
####FP32 (32-bit Float): This is the default. It uses 32 bits (4 bytes) to store one number. It can represent a huge range of numbers with very high precision (e.g., 3.14159265...).
####
####INT8 (8-bit Integer): This uses only 8 bits (1 byte). It can only represent $2^8 = 256$ distinct values (from -128 to 127).
####
####Why do it?
####
####4x Smaller Model: A 100GB FP32 model becomes a 25GB INT8 model.
####
####4x Less Memory Bandwidth: Moving 25GB from GPU memory (HBM) to the compute cores (SRAM) is 4x faster than moving 100GB. This is often the real bottleneck.
####
####Massively Faster Math: GPU cores (especially NVIDIA's Tensor Cores) can perform INT8 operations far faster than FP32 operations.
####
####To make this work, we must map the full FP32 range of our weights (e.g., [-0.2, 0.2]) to the full INT8 range ([-127, 127]). The scale factor is the "magic number" that lets us convert back and forth.
####
####fp32_value ≈ (int8_value / 127.0) * scale
####
####2. The "Fake" Quantized Multiply
####
####This is a technique used for Quantization-Aware Training (QAT).
####
####The Problem: The round() and .to(torch.int8) operations are non-differentiable. Their gradient is zero everywhere, so you can't backpropagate through them.
####
####The Solution: During the training forward pass, you "simulate" the quantization error:
####
####Forward Pass:
####weights_q, scale = quantize_int8(weights)
####weights_deq = dequantize_fp32(weights_q, scale)
####output = torch.matmul(x, weights_deq)
####
####Backward Pass:
####
####The backward pass flows through weights_deq.
####
####This is an FP32 tensor, so gradients flow normally.
####
####Crucially, weights_deq has the "rounding error" baked into it. The model learns to be resilient to this error. It learns to produce correct outputs even with the imprecise, quantized weights.
####
####This "fake" step is only for training.
####
####3. "Bonus": How True Hardware Quantization Works (Inference)
####
####This is the real speed-up. On modern hardware (like an NVIDIA H100 GPU), you never de-quantize the weights.
####
####Offline (Once): You run our quantize_int8 function on your trained FP32 weights. You save the W_int8 tensor (25GB) and the scale_w (4 bytes) to disk.
####
####At Inference (For every new request):
####
####Step 1: The W_int8 tensor is loaded into GPU memory.
####
####Step 2: A new FP32 input X_fp32 arrives.
####
####Step 3: The hardware on-the-fly quantizes the input: X_int8, scale_x = quantize(X_fp32). This is extremely fast.
####
####Step 4 (The Magic): The GPU's Tensor Cores perform the matrix multiply directly in the integer domain:
####Output_int32 = matmul_int8(X_int8, W_int8)
####(The output is 32-bit to prevent overflow from accumulating the 8-bit products)
####
####Step 5 (De-quantize): You only de-quantize the final, small output tensor back to FP32, using the scales you saved:
####Output_fp32 = Output_int32.float() * (scale_w * scale_x)
####
####This is dramatically faster because the most expensive operation—the massive matmul—happens entirely in the 8-bit domain, leveraging specialized hardware.



import torch


def quantize_int8(
        weights: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    """
    Performs simple per-tensor, asymmetric 8-bit quantization.

    This function implements the formula from the prompt:
    quantized_weight = (weight / abs_max) * 127

    Args:
        weights (torch.Tensor): The 32-bit floating-point (FP32) weight tensor.

    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing:
            - weights_int8: The quantized 8-bit integer (INT8) tensor.
            - scale: The FP32 scale factor (the absolute max value).
    """

    # 1. Find the absolute maximum value (the "scale").
    # This scale is a single FP32 scalar.
    scale = torch.abs(weights).max()

    # 2. Calculate the quantized int8 weights.

    # Normalize weights to the range [-1.0, 1.0]
    normalized_weights = weights / scale

    # Scale to the INT8 range [-127.0, 127.0]
    # (We use 127, not 128, to maintain a symmetric range)
    scaled_int_weights = normalized_weights * 127

    # 3. Cast to torch.int8.
    # .round() is crucial to get the closest integer value
    # before casting, rather than just truncating (which .to() can do).
    weights_int8 = scaled_int_weights.round().to(torch.int8)

    # 4. Return the int8 weights and the fp32 scale factor.
    return weights_int8, scale


def dequantize_fp32(
        weights_int8: torch.Tensor,
        scale: torch.Tensor
) -> torch.Tensor:
    """
    De-quantizes an INT8 tensor back to FP32, given its scale.
    This is the inverse of the `quantize_int8` function.

    Args:
        weights_int8 (torch.Tensor): The INT8 tensor.
        scale (torch.Tensor): The FP32 scale factor.

    Returns:
        torch.Tensor: The de-quantized FP32 tensor.
    """

    # Cast to float and normalize back to [-1.0, 1.0]
    normalized_weights = weights_int8.float() / 127.0

    # Scale back to the original FP32 range
    dequantized_weights = normalized_weights * scale

    return dequantized_weights


# --- Example Usage ---
if __name__ == "__main__":
    # 0. Setup
    B, D_IN, D_OUT = 64, 128, 256

    # Create a realistic weight tensor (e.g., from a linear layer)
    # Weights are typically small, centered around 0.
    fp32_weights = torch.randn(D_IN, D_OUT) * 0.02

    # Create dummy input
    x_input = torch.randn(B, D_IN)

    # --- 1. Original FP32 Operation (The "Ground Truth") ---
    print("--- 1. Original FP32 Operation ---")

    # Calculate memory size in KB
    fp32_mem_kb = (fp32_weights.nelement() * 32 / 8) / 1024
    print(f"FP32 weights memory: {fp32_mem_kb:.2f} KB")

    output_fp32 = torch.matmul(x_input, fp32_weights)

    # --- 2. Quantization ---
    print("\n--- 2. Quantization ---")
    int8_weights, scale = quantize_int8(fp32_weights)

    # Calculate memory size (plus 4 bytes for the 1 scale)
    int8_mem_kb = (int8_weights.nelement() * 8 / 8 + 4) / 1024
    print(f"INT8 weights memory:  {int8_mem_kb:.2f} KB  (Note: 4x smaller)")
    print(f"Scale factor (fp32): {scale.item()}")

    # --- 3. "Fake" Quantized Matrix Multiply ---
    print('\n--- 3. "Fake" Quantized Multiply (for QAT) ---')
    print("De-quantizing weights *before* matmul to simulate error...")

    # This is the "fake" part you asked about:
    # 1. De-quantize the INT8 weights back to FP32 (which introduces error)
    dequantized_w = dequantize_fp32(int8_weights, scale)

    # 2. Perform the matmul in FP32, but with the error-prone weights
    output_fake_quant = torch.matmul(x_input, dequantized_w)

    print("Fake quantized matmul complete.")

    # --- 4. Comparison ---
    print("\n--- 4. Comparison ---")

    # Calculate the error introduced by quantization
    mse = torch.mean((output_fp32 - output_fake_quant) ** 2)
    print(f"Mean Squared Error (quantization error): {mse.item():.2e}")

    # Check that they are close (but not identical)
    assert mse < 1e-5  # Error should be very small
    assert not torch.allclose(output_fp32, output_fake_quant)

    print("Outputs are very close. Quantization/de-quantization is correct.")
    print("The small error is the (expected) cost of quantization.")
