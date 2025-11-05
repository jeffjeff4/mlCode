import math


class MLPEstimator:
    """
    Estimates the training feasibility of a simple MLP on a given hardware.
    """

    def __init__(self, num_layers: int, hidden_dim: int, chip_memory_gb: float, chip_tflops: float):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.chip_memory_gb = chip_memory_gb
        self.chip_tflops = chip_tflops

        # Calculate total parameters
        params_per_layer = (hidden_dim * hidden_dim) + hidden_dim
        self.total_params = num_layers * params_per_layer

    def analyze(self, batch_size: int, mfu_estimate: float = 0.5):
        """
        Performs a full analysis of memory, compute, and performance.
        """
        print(f"--- Starting Analysis for Batch Size: {batch_size} ---")

        # --- Memory Calculation (in GB) ---
        params_mem = (self.total_params * 2) / (1024 ** 3)
        grads_mem = (self.total_params * 2) / (1024 ** 3)
        optim_mem = (self.total_params * 8) / (1024 ** 3)  # Adam is 8 bytes/param

        # Activations for input + all hidden layers
        activations_mem = (self.num_layers * batch_size * self.hidden_dim * 2) / (1024 ** 3)

        misc_overhead = 1.0  # GB
        total_mem = params_mem + grads_mem + optim_mem + activations_mem + misc_overhead

        print("\n--- Memory Breakdown ---")
        print(f"Total Model Parameters : {self.total_params / 1e6:.2f} M")
        print(f"  - Parameters         : {params_mem:.2f} GB")
        print(f"  - Gradients          : {grads_mem:.2f} GB")
        print(f"  - Optimizer (Adam)   : {optim_mem:.2f} GB")
        print(f"  - Activations        : {activations_mem:.2f} GB")
        print(f"  - Misc Overhead      : {misc_overhead:.2f} GB")
        print(f"------------------------------------")
        print(f"TOTAL ESTIMATED MEMORY : {total_mem:.2f} GB")

        if total_mem > self.chip_memory_gb:
            print(
                f"CONCLUSION: INFEASIBLE. Required memory ({total_mem:.2f} GB) exceeds chip capacity ({self.chip_memory_gb} GB).")
            return

        print(f"CONCLUSION: FEASIBLE. Fits within the {self.chip_memory_gb} GB chip memory.")

        # --- FLOPs and Performance ---
        flops_per_step = 6 * self.total_params * batch_size
        achieved_tflops = self.chip_tflops * mfu_estimate
        time_per_step_s = (flops_per_step / 1e12) / achieved_tflops if achieved_tflops > 0 else float('inf')

        print("\n--- Performance Estimation ---")
        print(f"FLOPs per Step         : {flops_per_step / 1e9:.2f} GFLOPs")
        print(f"Est. Achieved Perf.    : {achieved_tflops:.2f} TFLOPS (at {mfu_estimate * 100}% MFU)")
        print(f"Est. Time per Step     : {time_per_step_s * 1000:.2f} ms")
        print("-" * 40)


if __name__ == '__main__':
    # --- Configuration from the problem description ---
    CHIP_MEMORY = 8.0  # GB
    CHIP_PERFORMANCE = 100.0  # TFLOPS

    # SimpleMLP config
    NUM_LAYERS = 3
    HIDDEN_DIM = 10000

    estimator = MLPEstimator(
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        chip_memory_gb=CHIP_MEMORY,
        chip_tflops=CHIP_PERFORMANCE
    )

    # Analyze with a practical batch size
    estimator.analyze(batch_size=128)

    # Analyze with a larger batch size to improve utilization
    estimator.analyze(batch_size=4096)
