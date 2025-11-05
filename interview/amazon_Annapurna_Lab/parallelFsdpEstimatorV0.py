####please write code to estimate
####given a model, say llama 7b, estimate memory consumption, parameters, using fsdp parallel, split to k gpus. please including everything, e.g., gradients, activations, etc
####1. how much computation flops each gpu needs to calculate
####2. how much memory, each gpu needs to calculate, if activation uses a lot memory, calculate it
####3. how much data needs to be transferred,
####4. anything else needs to be counted



import math


class FSDPEstimator:
    """
    Estimates the per-GPU resource requirements for training an LLM
    using Fully Sharded Data Parallelism (FSDP).
    """

    def __init__(self, model_params_b: float, hidden_dim: int, num_layers: int):
        """
        Initializes the estimator with the model's architecture.
        """
        self.total_params = model_params_b * 1e9
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _to_gb(self, bytes_val: float) -> float:
        """Converts bytes to gigabytes."""
        return bytes_val / (1024 ** 3)

    def calculate_per_gpu_flops(self, global_batch_size: int, seq_len: int, k: int) -> float:
        """
        1. Calculates the computational FLOPs per GPU for one iteration.
        In FSDP (a Data Parallel method), each GPU processes 1/k of the batch.
        """
        if k <= 0: return 0
        per_gpu_batch_size = global_batch_size / k
        # Each GPU runs a full forward/backward pass on its local batch
        flops_per_gpu = 6 * self.total_params * per_gpu_batch_size * seq_len
        return flops_per_gpu

    def calculate_per_gpu_memory(self, global_batch_size: int, seq_len: int, k: int, model_precision_bytes: int = 2,
                                 use_activation_checkpointing: bool = True) -> dict:
        """
        2. Calculates per-GPU memory consumption with a detailed breakdown.
        """
        if k <= 0: return {}

        # a) Sharded Components: Params, Gradients, and Optimizer States
        # Each GPU holds only its slice of these.
        params_per_gpu = self.total_params / k
        param_mem = params_per_gpu * model_precision_bytes

        # Gradients are also sharded. We account for a temporary full gradient during Reduce-Scatter.
        grad_mem = params_per_gpu * model_precision_bytes

        # Optimizer states (e.g., Adam) are sharded.
        optimizer_mem = params_per_gpu * 4 * 2  # (Momentum + Variance in FP32)

        # b) Replicated/Per-GPU Components: Activations
        # CRITICAL: Each GPU stores the full activations for its OWN micro-batch.
        per_gpu_batch_size = global_batch_size / k
        activation_mem = per_gpu_batch_size * seq_len * self.hidden_dim * self.num_layers * model_precision_bytes

        if use_activation_checkpointing:
            # Activation checkpointing trades compute for memory.
            # A common approximation for its savings. sqrt(num_layers) is a decent heuristic.
            activation_mem /= math.sqrt(self.num_layers)

        # c) Temporary Peak Memory: Full Layer Weights
        # FSDP materializes the full weights for one layer at a time.
        # The largest layer in a Transformer is usually the FFN.
        ffn_params = (self.hidden_dim * self.hidden_dim * 4) + (self.hidden_dim * 4 * self.hidden_dim)
        peak_layer_mem = ffn_params * model_precision_bytes

        misc_overhead = 5 * (1024 ** 3)
        total_mem = param_mem + grad_mem + optimizer_mem + activation_mem + peak_layer_mem + misc_overhead

        return {
            "Sharded Params": self._to_gb(param_mem),
            "Sharded Optimizer States": self._to_gb(optimizer_mem),
            "Activations (Post-Checkpointing)": self._to_gb(activation_mem),
            "Peak Temporary Layer Mem": self._to_gb(peak_layer_mem),
            "Misc Overhead": self._to_gb(misc_overhead),
            "TOTAL ESTIMATED MEMORY": self._to_gb(total_mem)
        }

    def calculate_per_gpu_communication(self, k: int, model_precision_bytes: int = 2) -> dict:
        """
        3. Estimates the data transfer volume per GPU per iteration.
        Communication in FSDP is for model parameters and gradients.
        """
        if k <= 1:
            return {"Total Communication per Iteration (GB)": 0}

        # The total data volume moved in collectives (All-Gather + Reduce-Scatter) is
        # proportional to the total model size, twice (once for fwd, once for bwd).
        total_model_size_bytes = self.total_params * model_precision_bytes

        # Effective volume per GPU using the 2*(k-1)/k model for efficient collectives
        comm_volume = 2 * (2 * ((k - 1) / k) * total_model_size_bytes)

        return {
            "Total Communication per Iteration (GB)": self._to_gb(comm_volume)
        }

    def estimate_other_factors(self, time_per_iter_s: float, peak_gpu_tflops: float, interconnect_bandwidth_gbps: float,
                               k: int, global_batch_size: int, seq_len: int) -> dict:
        """
        4. Calculates metrics for "other factors" that limit performance.
        """
        if k <= 0 or time_per_iter_s <= 0: return {}

        achieved_flops = self.calculate_per_gpu_flops(global_batch_size, seq_len, k)
        achieved_tflops = (achieved_flops / time_per_iter_s) / 1e12
        mfu = (achieved_tflops / peak_gpu_tflops) * 100 if peak_gpu_tflops > 0 else 0

        comm_gb = self.calculate_per_gpu_communication(k).get("Total Communication per Iteration (GB)", 0)
        comm_time_s = (comm_gb * 8) / interconnect_bandwidth_gbps if interconnect_bandwidth_gbps > 0 else float('inf')

        comm_bound_ratio = (comm_time_s / time_per_iter_s) * 100 if time_per_iter_s > 0 else 0

        return {
            "a) Achieved TFLOPS (Per GPU)": f"{achieved_tflops:.2f}",
            "b) Model FLOPS Utilization (MFU)": f"{mfu:.2f}%",
            "c) Theoretical Comm Time (No Overlap)": f"{comm_time_s:.4f} s",
            "d) Communication Bound Ratio": f"{comm_bound_ratio:.2f}% (Upper bound, assumes no compute/comm overlap)",
        }


if __name__ == '__main__':
    llama_7b_config = {'model_params_b': 7, 'hidden_dim': 4096, 'num_layers': 32}
    PEAK_GPU_TFLOPS = 312  # A100 BF16
    INTERCONNECT_BANDWIDTH_GBPS = 200 * 8  # 200 GB/s InfiniBand (typical for multi-node)
    GLOBAL_BATCH_SIZE = 64
    SEQUENCE_LENGTH = 4096
    mock_iteration_times = {1: 60.0, 2: 32.0, 4: 17.0, 8: 9.5, 16: 5.5}

    estimator = FSDPEstimator(**llama_7b_config)

    print("--- FSDP Scaling Estimates for Llama 7B on A100-like GPUs ---")

    for k_gpus in [1, 2, 4, 8, 16]:
        if k_gpus > GLOBAL_BATCH_SIZE: continue
        print(f"\n========================= ESTIMATE FOR k = {k_gpus} GPU(s) =========================")

        flops = estimator.calculate_per_gpu_flops(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        memory = estimator.calculate_per_gpu_memory(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        comms = estimator.calculate_per_gpu_communication(k_gpus)

        print(f"1. PER-GPU FLOPs per Iteration        : {flops / 1e15:.2f} PetaFLOPs")
        print("\n2. PER-GPU MEMORY BREAKDOWN:")
        for key, val in memory.items(): print(f"   - {key:<30}: {val:.2f} GB")

        print("\n3. PER-GPU DATA TRANSFER:")
        for key, val in comms.items(): print(f"   - {key:<30}: {val:.2f} GB")

        print("\n4. OTHER LIMITING FACTORS (Performance Analysis):")
        iter_time = mock_iteration_times[k_gpus]
        print(f"   (Assuming measured iteration time: {iter_time:.2f}s)")
        other = estimator.estimate_other_factors(iter_time, PEAK_GPU_TFLOPS, INTERCONNECT_BANDWIDTH_GBPS, k_gpus,
                                                 GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH)
        for key, val in other.items(): print(f"   - {key:<30}: {val}")

    print("=========================================================================\n")
