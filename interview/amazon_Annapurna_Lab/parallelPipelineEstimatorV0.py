####please write code to estimate
####given a model, say llama 7b, estimate memory consumption, parameters, using pipeline parallel, split to k gpus. please including everything, e.g., gradients, activations, etc
####1. how much computation flops each gpu needs to calculate
####2. how much memory, each gpu needs to calculate, if activation uses a lot memory, calculate it
####3. how much data needs to be transferred,
####4. anything else needs to be counted

import math


class PipelineParallelEstimator:
    """
    Estimates the per-GPU resource requirements for training an LLM
    using Pipeline Parallelism (PP).
    """

    def __init__(self, model_params_b: float, hidden_dim: int, num_layers: int):
        self.total_params = model_params_b * 1e9
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _to_gb(self, bytes_val: float) -> float:
        return bytes_val / (1024 ** 3)

    def calculate_per_gpu_flops(self, global_batch_size: int, seq_len: int, k: int) -> float:
        """
        1. Calculates the computational FLOPs per GPU for one iteration.
        The model layers are split across k GPUs, so the computation is also split.
        """
        if k <= 0: return 0
        total_flops_per_iter = 6 * self.total_params * global_batch_size * seq_len
        return total_flops_per_iter / k

    def calculate_per_gpu_memory(self, global_batch_size: int, k: int, num_micro_batches: int, seq_len: int,
                                 model_precision_bytes: int = 2) -> dict:
        """
        2. Calculates per-GPU memory consumption.
        """
        if k <= 0: return {}

        # a) Sharded Components: Params, Gradients, and Optimizer States are sharded by layer.
        params_per_gpu = self.total_params / k
        param_mem = params_per_gpu * model_precision_bytes
        grad_mem = params_per_gpu * model_precision_bytes
        optimizer_mem = params_per_gpu * 4 * 2  # Adam states

        # b) Activation Memory: This is the most complex part for PP.
        # Each stage must store the activations for its micro-batches.
        # For a simple GPipe schedule, a stage can hold up to (k-1) activations
        # for micro-batches that are in flight in other stages.
        micro_batch_size = global_batch_size / num_micro_batches

        # Activations for ONE micro-batch for the layers ON THIS GPU.
        layers_per_gpu = self.num_layers / k
        activations_per_micro_batch = micro_batch_size * seq_len * self.hidden_dim * layers_per_gpu * model_precision_bytes

        # Peak memory is holding activations for multiple micro-batches simultaneously.
        # This is often managed and can be complex, but a good approximation is that
        # the total activation memory is for the full per-GPU batch.
        peak_activation_mem = (
                                          global_batch_size / k) * seq_len * self.hidden_dim * layers_per_gpu * model_precision_bytes

        misc_overhead = 5 * (1024 ** 3)
        total_mem = param_mem + grad_mem + optimizer_mem + peak_activation_mem + misc_overhead

        return {
            "Sharded Params (by layer)": self._to_gb(param_mem),
            "Sharded Optimizer States": self._to_gb(optimizer_mem),
            "Peak Activation Memory": self._to_gb(peak_activation_mem),
            "Misc Overhead": self._to_gb(misc_overhead),
            "TOTAL ESTIMATED MEMORY": self._to_gb(total_mem)
        }

    def calculate_per_gpu_communication(self, global_batch_size: int, k: int, num_micro_batches: int, seq_len: int,
                                        model_precision_bytes: int = 2) -> dict:
        """
        3. Estimates the data transfer volume between adjacent pipeline stages.
        """
        if k <= 1:
            return {"Total Communication per Iteration (GB)": 0}

        micro_batch_size = global_batch_size / num_micro_batches

        # Data transferred is the activation tensor between stages.
        activation_tensor_size = micro_batch_size * seq_len * self.hidden_dim * model_precision_bytes

        # This transfer happens for each micro-batch in the forward pass and backward pass.
        total_comm = 2 * num_micro_batches * activation_tensor_size

        return {
            "Total Communication per Iteration (GB)": self._to_gb(total_comm)
        }

    def estimate_other_factors(self, k: int, num_micro_batches: int) -> dict:
        """
        4. Calculates the pipeline bubble, the key performance limiter for PP.
        """
        if k <= 1:
            return {"a) Pipeline Bubble Overhead": "0.00%"}

        # The bubble is the fraction of time the hardware is idle.
        # Formula for a simple GPipe schedule: (k-1) / (m + k - 1)
        # However, a simpler and more intuitive formula is just (k-1)/m
        # This represents the ramp-up/down phases over the total work phases.
        bubble_fraction = (k - 1) / num_micro_batches

        # Effective utilization is 1 minus the bubble fraction.
        effective_utilization = 1 - bubble_fraction

        return {
            "a) Pipeline Bubble Overhead": f"{bubble_fraction * 100:.2f}%",
            "b) Effective Hardware Utilization": f"{effective_utilization * 100:.2f}%"
        }


if __name__ == '__main__':
    llama_7b_config = {'model_params_b': 7, 'hidden_dim': 4096, 'num_layers': 32}

    # --- SCENARIO CONFIG ---
    # In PP, the global batch size is split into many small micro-batches
    GLOBAL_BATCH_SIZE = 64
    NUM_MICRO_BATCHES = 32  # Must be >= k, and ideally >> k
    SEQUENCE_LENGTH = 4096

    estimator = PipelineParallelEstimator(**llama_7b_config)

    print("--- Pipeline Parallelism (PP) Scaling Estimates for Llama 7B ---")

    for k_gpus in [1, 2, 4, 8]:
        if NUM_MICRO_BATCHES < k_gpus:
            continue
        print(f"\n========================= ESTIMATE FOR k = {k_gpus} STAGE(S) =========================")

        flops = estimator.calculate_per_gpu_flops(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        memory = estimator.calculate_per_gpu_memory(GLOBAL_BATCH_SIZE, k_gpus, NUM_MICRO_BATCHES, SEQUENCE_LENGTH)
        comms = estimator.calculate_per_gpu_communication(GLOBAL_BATCH_SIZE, k_gpus, NUM_MICRO_BATCHES, SEQUENCE_LENGTH)

        print(f"1. PER-GPU FLOPs per Iteration        : {flops / 1e15:.2f} PetaFLOPs")
        print("\n2. PER-GPU MEMORY BREAKDOWN:")
        for key, val in memory.items(): print(f"   - {key:<30}: {val:.2f} GB")

        print("\n3. PER-GPU DATA TRANSFER (to next stage):")
        for key, val in comms.items(): print(f"   - {key:<30}: {val:.2f} GB")

        print("\n4. OTHER LIMITING FACTORS (The Pipeline Bubble):")
        other = estimator.estimate_other_factors(k_gpus, NUM_MICRO_BATCHES)
        for key, val in other.items(): print(f"   - {key:<30}: {val}")

    print("=========================================================================\n")
