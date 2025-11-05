import math


class MFU_Estimator:
    """
    Estimates the training feasibility and MFU for a given model and hardware.
    """

    def __init__(self, chip_memory_gb: float, chip_tflops: float):
        self.chip_memory_gb = chip_memory_gb
        self.chip_tflops = chip_tflops
        self.total_params = 0

    def _calculate_params(self, input_dim, hidden_dim, num_hidden_layers):
        """Calculates parameters for a multi-layer MLP."""
        # Layer 1
        self.total_params += (input_dim * hidden_dim) + hidden_dim
        # Hidden Layers
        self.total_params += (num_hidden_layers - 1) * ((hidden_dim * hidden_dim) + hidden_dim)
        # Output Layer (assuming same as hidden)
        self.total_params += (hidden_dim * hidden_dim) + hidden_dim

    def analyze(self, seq_len: int, embedding_dim: int, hidden_dim: int, num_layers: int, throughput: int):
        """
        Performs a full analysis.
        """
        print(f"--- Analysis for SeqLen={seq_len}, EmbDim={embedding_dim}, HiddenDim={hidden_dim} ---")

        # 1. Calculate Model Parameters based on flattened input
        input_dim = seq_len * embedding_dim
        self._calculate_params(input_dim, hidden_dim, num_layers - 1)
        print(f"Total Model Parameters : {self.total_params / 1e9:.2f} B")

        # 2. Check Memory Feasibility
        params_mem_gb = (self.total_params * 2) / (1024 ** 3)
        print(f"Memory for Parameters (BF16): {params_mem_gb:.2f} GB")
        if params_mem_gb > self.chip_memory_gb:
            print(
                f"CONCLUSION: INFEASIBLE. Parameters alone ({params_mem_gb:.2f} GB) exceed chip memory ({self.chip_memory_gb} GB).")
            print("MFU calculation is a theoretical exercise for an impossible scenario.")
        else:
            print("CONCLUSION: Model parameters fit in memory. Continuing analysis.")

        # 3. Calculate MFU
        # For a flattened MLP, FLOPs per sample is ~6 * N
        flops_per_sample = 6 * self.total_params

        achieved_flops = flops_per_sample * throughput
        achieved_tflops = achieved_flops / 1e12

        mfu = (achieved_tflops / self.chip_tflops) * 100 if self.chip_tflops > 0 else 0

        print("\n--- MFU Calculation ---")
        print(f"Given Throughput       : {throughput} samples/sec")
        print(f"FLOPs per Sample       : {flops_per_sample / 1e9:.2f} GFLOPs")
        print(f"Achieved Performance   : {achieved_tflops:.2f} TFLOPS")
        print(f"Peak Chip Performance  : {self.chip_tflops:.2f} TFLOPS")
        print(f"------------------------------------")
        print(f"MODEL FLOPS UTILIZATION (MFU): {mfu:.2f}%")
        print("-" * 40)


if __name__ == '__main__':
    # --- Configuration from the problem description ---
    CHIP_MEMORY = 8.0  # GB
    CHIP_PERFORMANCE = 100.0  # TFLOPS

    # SimpleMLP config
    NUM_LAYERS = 3
    HIDDEN_DIM = 10000

    # Input Shape config (swapped from previous question)
    SEQ_LEN = 100
    EMBEDDING_DIM = 10000

    # Performance metric
    THROUGHPUT = 200  # samples/sec

    estimator = MFU_Estimator(
        chip_memory_gb=CHIP_MEMORY,
        chip_tflops=CHIP_PERFORMANCE
    )

    estimator.analyze(
        seq_len=SEQ_LEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        throughput=THROUGHPUT
    )

