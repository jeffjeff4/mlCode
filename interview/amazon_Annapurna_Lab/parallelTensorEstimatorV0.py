####please write code to estimate
####given a model, say llama 7b, estimate memory consumption, parameters, using tensor parallel, split to k gpus. please including everything, e.g., gradients, activations, etc
####1. how much computation flops each gpu needs to calculate
####2. how much memory, each gpu needs to calculate, if activation uses a lot memory, calculate it
####3. how much data needs to be transferred,
####4. anything else needs to be counted



import math


class TensorParallelEstimator:
    """
    Estimates the per-GPU resource requirements for training an LLM
    using Tensor Parallelism (TP), including secondary performance factors.
    """

    def __init__(self, model_params_b: float, hidden_dim: int, num_layers: int, num_heads: int):
        """
        Initializes the estimator with the model's architecture.
        """
        self.total_params = model_params_b * 1e9
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def _to_gb(self, bytes_val: float) -> float:
        """Converts bytes to gigabytes."""
        return bytes_val / (1024 ** 3)

    def calculate_per_gpu_flops(self, global_batch_size: int, seq_len: int, k: int) -> float:
        """
        1. Calculates the computational FLOPs required per GPU for one iteration.
        The total training FLOPs (6 * N * B * S) are distributed across k GPUs.
        """
        if k == 0: return 0
        total_flops_per_iter = 6 * self.total_params * global_batch_size * seq_len
        return total_flops_per_iter / k

    def calculate_per_gpu_memory(self, global_batch_size: int, seq_len: int, k: int,
                                 model_precision_bytes: int = 2) -> dict:
        """
        2. Calculates the per-GPU memory consumption with a detailed breakdown.
        """
        if k <= 0: return {}

        # Parameters and Optimizer states are partitioned across k GPUs.
        params_per_gpu = self.total_params / k
        param_mem = params_per_gpu * model_precision_bytes

        grad_mem = params_per_gpu * model_precision_bytes
        momentum_mem = params_per_gpu * 4  # Adam states are usually FP32
        variance_mem = params_per_gpu * 4
        optimizer_mem = grad_mem + momentum_mem + variance_mem

        # Activations are partly partitioned. A 1/k split is a strong approximation.
        # This calculation is for the full activation tensor size, then partitioned.
        total_activations_bytes = global_batch_size * seq_len * self.hidden_dim * self.num_layers * model_precision_bytes
        activation_mem = total_activations_bytes / k

        # A buffer for CUDA context, fragmentation, and other runtime overheads.
        misc_overhead = 5 * (1024 ** 3)
        total_mem = param_mem + optimizer_mem + activation_mem + misc_overhead

        return {
            "Parameters": self._to_gb(param_mem),
            "Optimizer States (Adam)": self._to_gb(optimizer_mem),
            "Activations (Approximation)": self._to_gb(activation_mem),
            "Misc Overhead": self._to_gb(misc_overhead),
            "TOTAL ESTIMATED MEMORY": self._to_gb(total_mem)
        }

    def calculate_per_gpu_communication(self, global_batch_size: int, seq_len: int, k: int,
                                        model_precision_bytes: int = 2) -> dict:
        """
        3. Estimates the data transfer volume required per GPU per iteration.
        """
        if k <= 1:
            return {"Total Communication per Iteration (GB)": 0}

        # Per-GPU batch size is needed for activation tensor size calculation
        per_gpu_batch_size = global_batch_size / k

        # Size of the activation tensor that needs to be communicated
        activation_tensor_size = per_gpu_batch_size * seq_len * self.hidden_dim * model_precision_bytes

        # An All-Reduce communication is needed twice per transformer layer (in fwd pass, mirrored in bwd pass).
        # Data volume for one GPU in one All-Reduce: 2 * (k-1)/k * message_size
        comm_per_all_reduce = 2 * ((k - 1) / k) * activation_tensor_size
        total_comm = self.num_layers * 2 * comm_per_all_reduce

        return {
            "Total Communication per Iteration (GB)": self._to_gb(total_comm)
        }

    def estimate_other_factors(self, time_per_iter_s: float, peak_gpu_tflops: float, interconnect_bandwidth_gbps: float,
                               k: int, global_batch_size: int, seq_len: int) -> dict:
        """
        4. Calculates metrics for "other factors" that limit performance.
        """
        if k <= 0 or time_per_iter_s <= 0: return {}

        # a) Model FLOPS Utilization (MFU)
        achieved_flops = self.calculate_per_gpu_flops(global_batch_size, seq_len, k)
        achieved_tflops = (achieved_flops / time_per_iter_s) / 1e12
        mfu = (achieved_tflops / peak_gpu_tflops) * 100 if peak_gpu_tflops > 0 else 0

        # b) Network Communication Impact
        comm_gb = self.calculate_per_gpu_communication(global_batch_size, seq_len, k).get(
            "Total Communication per Iteration (GB)", 0)
        # Theoretical time spent just on communication
        comm_time_s = (comm_gb * 8) / interconnect_bandwidth_gbps if interconnect_bandwidth_gbps > 0 else float(
            'inf')  # convert GB to Gb

        # c) Compute vs. Communication Balance
        compute_time_s = time_per_iter_s - comm_time_s
        comm_bound_ratio = (comm_time_s / time_per_iter_s) * 100 if time_per_iter_s > 0 else 0

        return {
            "a) Achieved TFLOPS (Per GPU)": f"{achieved_tflops:.2f}",
            "b) Model FLOPS Utilization (MFU)": f"{mfu:.2f}%",
            "c) Theoretical Comm Time per Iter": f"{comm_time_s:.4f} s",
            "d) Communication Bound Ratio": f"{comm_bound_ratio:.2f}% (Percentage of iteration time spent on communication)",
        }


if __name__ == '__main__':
    # --- MODEL CONFIG: Llama 7B ---
    llama_7b_config = {
        'model_params_b': 7, 'hidden_dim': 4096,
        'num_layers': 32, 'num_heads': 32
    }

    # --- HARDWARE CONFIG: NVIDIA A100 ---
    # (BF16 performance, from spec sheet)
    PEAK_GPU_TFLOPS = 312
    # (NVLink bandwidth in a DGX A100 node)
    INTERCONNECT_BANDWIDTH_GBPS = 600 * 8  # 600 GB/s * 8-bit/Byte

    # --- SCENARIO CONFIG ---
    GLOBAL_BATCH_SIZE = 32
    SEQUENCE_LENGTH = 2048
    # Mocked iteration time (in a real scenario, you would measure this)
    # This might decrease as k increases, but communication adds overhead.
    # Let's model a realistic scenario where adding GPUs has diminishing returns.
    mock_iteration_times = {1: 30.0, 2: 16.0, 4: 8.8, 8: 5.5}

    estimator = TensorParallelEstimator(**llama_7b_config)

    print("--- Tensor Parallelism Scaling Estimates for Llama 7B on A100-like GPUs ---")

    for k_gpus in [1, 2, 4, 8]:
        print(f"\n========================= ESTIMATE FOR k = {k_gpus} GPU(s) =========================")

        flops_per_iter = estimator.calculate_per_gpu_flops(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        memory_breakdown = estimator.calculate_per_gpu_memory(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        communication_breakdown = estimator.calculate_per_gpu_communication(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)

        print(f"1. PER-GPU FLOPs per Iteration        : {flops_per_iter / 1e15:.2f} PetaFLOPs")
        print("\n2. PER-GPU MEMORY BREAKDOWN:")
        for key, val in memory_breakdown.items():
            print(f"   - {key:<25}: {val:.2f} GB")

        print("\n3. PER-GPU DATA TRANSFER:")
        for key, val in communication_breakdown.items():
            print(f"   - {key:<30}: {val:.2f} GB")

        print("\n4. OTHER LIMITING FACTORS (Performance Analysis):")
        iter_time = mock_iteration_times[k_gpus]
        print(f"   (Assuming measured iteration time: {iter_time:.2f}s)")
        other_factors = estimator.estimate_other_factors(iter_time, PEAK_GPU_TFLOPS, INTERCONNECT_BANDWIDTH_GBPS,
                                                         k_gpus, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH)
        for key, val in other_factors.items():
            print(f"   - {key:<30}: {val}")

    print("=========================================================================\n")
