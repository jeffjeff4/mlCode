import math


class SequenceParallelEstimator:
    """
    Estimates the per-GPU resource requirements for training an LLM
    using Sequence Parallelism (SP).
    """

    def __init__(self, model_params_b: float, hidden_dim: int, num_layers: int, num_heads: int):
        self.total_params = model_params_b * 1e9
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def _to_gb(self, bytes_val: float) -> float:
        return bytes_val / (1024 ** 3)

    def calculate_per_gpu_flops(self, global_batch_size: int, seq_len: int, k: int) -> float:
        """
        1. Calculates the computational FLOPs per GPU for one iteration.
        In SP, each GPU processes the full model on a sharded sequence.
        The total computation is split evenly across the k GPUs.
        """
        if k <= 0: return 0
        total_flops_per_iter = 6 * self.total_params * global_batch_size * seq_len
        # The computation is distributed across the sequence dimension, so it's split over k GPUs.
        return total_flops_per_iter / k

    def calculate_per_gpu_memory(self, global_batch_size: int, seq_len: int, k: int,
                                 model_precision_bytes: int = 2) -> dict:
        """
        2. Calculates per-GPU memory consumption.
        The main benefit of SP is sharding the activation memory.
        """
        if k <= 0: return {}

        # a) Replicated Components: Model, Gradients, Optimizer States are NOT sharded by SP.
        param_mem = self.total_params * model_precision_bytes
        grad_mem = self.total_params * model_precision_bytes
        optimizer_mem = self.total_params * 4 * 2  # Adam states (Momentum + Variance in FP32)

        # b) Sharded Activation Memory: This is the key innovation of SP.
        # Total activation memory if not sharded:
        total_activation_mem = global_batch_size * seq_len * self.hidden_dim * self.num_layers * model_precision_bytes
        # With SP, the sequence length dimension of activations is effectively divided by k.
        sharded_activation_mem = total_activation_mem / k

        misc_overhead = 5 * (1024 ** 3)
        total_mem = param_mem + grad_mem + optimizer_mem + sharded_activation_mem + misc_overhead

        return {
            "Replicated Params": self._to_gb(param_mem),
            "Replicated Gradients": self._to_gb(grad_mem),
            "Replicated Optimizer States": self._to_gb(optimizer_mem),
            "Sharded Activations": self._to_gb(sharded_activation_mem),
            "Misc Overhead": self._to_gb(misc_overhead),
            "TOTAL ESTIMATED MEMORY": self._to_gb(total_mem)
        }

    def calculate_per_gpu_communication(self, global_batch_size: int, seq_len: int, k: int,
                                        model_precision_bytes: int = 2) -> dict:
        """
        3. Estimates the data transfer volume from All-to-All collectives.
        """
        if k <= 1:
            return {"Total Communication per Iteration (GB)": 0}

        # Data transferred is a slice of the activation tensor (Q, K, V).
        # Size of the tensor slice per GPU before All-to-All.
        # B x (S/k) x H
        local_activation_slice_size = global_batch_size * (seq_len / k) * self.hidden_dim * model_precision_bytes

        # In an All-to-All, each GPU sends (k-1) chunks and receives (k-1) chunks.
        # Theoretical data volume moved by one GPU: 2 * (k-1)/k * total_data_across_gpus
        # Here, the message size itself is the local slice.
        comm_per_all_to_all = 2 * ((k - 1) / k) * local_activation_slice_size * k  # Multiply by k for total data

        # This happens for Q, K, V in attention fwd, and for gradient comms in bwd.
        # A standard model is 2 All-to-Alls per layer (fwd/bwd attention).
        total_comm = self.num_layers * 2 * comm_per_all_to_all

        return {
            "Total Communication per Iteration (GB)": self._to_gb(total_comm)
        }

    def estimate_other_factors(self, time_per_iter_s: float, peak_gpu_tflops: float, interconnect_bandwidth_gbps: float,
                               k: int, global_batch_size: int, seq_len: int) -> dict:
        """
        4. Calculates metrics for "other factors" that limit performance.
        """
        if k <= 1 or time_per_iter_s <= 0:
            return {"a) Note": "No parallelism or invalid iteration time."}

        achieved_flops = self.calculate_per_gpu_flops(global_batch_size, seq_len, k)
        achieved_tflops = (achieved_flops / time_per_iter_s) / 1e12
        mfu = (achieved_tflops / peak_gpu_tflops) * 100 if peak_gpu_tflops > 0 else 0

        comm_gb = self.calculate_per_gpu_communication(global_batch_size, seq_len, k).get(
            "Total Communication per Iteration (GB)", 0)
        comm_time_s = (comm_gb * 8) / interconnect_bandwidth_gbps if interconnect_bandwidth_gbps > 0 else float('inf')
        comm_bound_ratio = (comm_time_s / time_per_iter_s) * 100 if time_per_iter_s > 0 else 0

        return {
            "a) Achieved TFLOPS (Per GPU)": f"{achieved_tflops:.2f}",
            "b) Model FLOPS Utilization (MFU)": f"{mfu:.2f}%",
            "c) Theoretical Comm Time (No Overlap)": f"{comm_time_s:.4f} s",
            "d) Communication Bound Ratio": f"{comm_bound_ratio:.2f}% (SP is highly sensitive to this)",
        }


if __name__ == '__main__':
    llama_7b_config = {'model_params_b': 7, 'hidden_dim': 4096, 'num_layers': 32, 'num_heads': 32}
    PEAK_GPU_TFLOPS = 312
    INTERCONNECT_BANDWIDTH_GBPS = 600 * 8  # NVLink for intra-node SP

    # A very long sequence length to demonstrate SP's use case
    GLOBAL_BATCH_SIZE = 8
    SEQUENCE_LENGTH = 16384

    estimator = SequenceParallelEstimator(**llama_7b_config)

    print("--- Sequence Parallelism (SP) Scaling Estimates for Llama 7B (Long Sequence) ---")
    mock_iteration_times = {1: 120.0, 2: 65.0, 4: 38.0}

    for k_gpus in [1, 2, 4]:
        if k_gpus > GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH: continue
        print(f"\n========================= ESTIMATE FOR k = {k_gpus} GPU(s) =========================")

        flops = estimator.calculate_per_gpu_flops(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        memory = estimator.calculate_per_gpu_memory(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        comms = estimator.calculate_per_gpu_communication(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)

        print(f"1. PER-GPU FLOPs per Iteration        : {flops / 1e15:.2f} PetaFLOPs")
        print("\n2. PER-GPU MEMORY BREAKDOWN:")
        for key, val in memory.items(): print(f"   - {key:<30}: {val:.2f} GB")

        print("\n3. PER-GPU DATA TRANSFER (All-to-All):")
        for key, val in comms.items(): print(f"   - {key:<30}: {val:.2f} GB")

        print("\n4. OTHER LIMITING FACTORS (Performance Analysis):")
        iter_time = mock_iteration_times.get(k_gpus, float('inf'))
        print(f"   (Assuming measured iteration time: {iter_time:.2f}s)")
        other = estimator.estimate_other_factors(iter_time, PEAK_GPU_TFLOPS, INTERCONNECT_BANDWIDTH_GBPS, k_gpus,
                                                 GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH)
        for key, val in other.items(): print(f"   - {key:<30}: {val}")

    print("=========================================================================\n")
