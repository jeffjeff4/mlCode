import math


class SingleGPUEstimator:
    """
    Estimates the full resource requirements for training a large language model
    on a single GPU without any parallelism.
    """

    def __init__(self, model_params_b: float, hidden_dim: int, num_layers: int, num_heads: int, vocab_size: int,
                 intermediate_dim_multiplier: int = 4):
        self.total_params = model_params_b * 1e9
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.intermediate_dim = hidden_dim * intermediate_dim_multiplier

    def _to_gb(self, bytes_val: float) -> float:
        """Converts bytes to gigabytes."""
        return bytes_val / (1024 ** 3)

    def calculate_flops(self, global_batch_size: int, seq_len: int) -> dict:
        """
        1. Calculates the total computational FLOPs for one iteration.
        """
        total_flops = 6 * self.total_params * global_batch_size * seq_len
        return {"Total FLOPs per Iter (PetaFLOPs)": total_flops / 1e15}

    def _calculate_parameter_breakdown(self) -> dict:
        """Helper to show a detailed breakdown of model parameters."""
        # Parameters per layer
        wq_params = self.hidden_dim * self.hidden_dim
        wk_params = self.hidden_dim * self.hidden_dim
        wv_params = self.hidden_dim * self.hidden_dim
        wo_params = self.hidden_dim * self.hidden_dim
        attention_params_per_layer = wq_params + wk_params + wv_params + wo_params

        ffn_up_params = self.hidden_dim * self.intermediate_dim
        ffn_down_params = self.intermediate_dim * self.hidden_dim
        ffn_params_per_layer = ffn_up_params + ffn_down_params

        layernorm_params_per_layer = 2 * self.hidden_dim * 2  # Two LayerNorms per block

        params_per_layer = attention_params_per_layer + ffn_params_per_layer + layernorm_params_per_layer
        total_transformer_params = params_per_layer * self.num_layers

        # Other parameters
        embedding_params = self.vocab_size * self.hidden_dim
        output_head_params = self.hidden_dim * self.vocab_size

        total_calculated = total_transformer_params + embedding_params + output_head_params

        return {
            "Total Calculated Params (B)": total_calculated / 1e9,
            "Wq/Wk/Wv/Wo per layer (M)": attention_params_per_layer / 1e6,
            "FFN per layer (M)": ffn_params_per_layer / 1e6,
            "Total Transformer Blocks (B)": total_transformer_params / 1e9,
            "Embeddings + Output Head (B)": (embedding_params + output_head_params) / 1e9
        }

    def calculate_memory(self, global_batch_size: int, seq_len: int, model_precision_bytes: int = 2) -> dict:
        """
        2. Calculates the total memory consumption with a detailed breakdown.
        """
        # a) Model State Memory (Parameters, Gradients, Optimizer States)
        param_mem = self.total_params * model_precision_bytes
        grad_mem = self.total_params * model_precision_bytes
        optimizer_mem = self.total_params * 4 * 2  # Adam states (Momentum + Variance in FP32)
        model_state_mem = param_mem + grad_mem + optimizer_mem

        # b) Activation Memory
        # Main activation tensor that flows through the model
        main_activations = global_batch_size * seq_len * self.hidden_dim * self.num_layers * model_precision_bytes
        # Attention scores (the quadratic bottleneck)
        attention_scores = global_batch_size * self.num_heads * seq_len * seq_len * model_precision_bytes
        activation_mem = main_activations + attention_scores

        misc_overhead = 5 * (1024 ** 3)  # 5 GB for CUDA context, etc.
        total_mem = model_state_mem + activation_mem + misc_overhead

        return {
            "Parameter Breakdown": self._calculate_parameter_breakdown(),
            "Model Parameters": self._to_gb(param_mem),
            "Gradients": self._to_gb(grad_mem),
            "Optimizer States (Adam)": self._to_gb(optimizer_mem),
            "Subtotal Model State": self._to_gb(model_state_mem),
            "---": "---",
            "Main Activation Tensors": self._to_gb(main_activations),
            "Attention Score Tensors (S^2)": self._to_gb(attention_scores),
            "Subtotal Activations": self._to_gb(activation_mem),
            "---": "---",
            "Misc Overhead": self._to_gb(misc_overhead),
            "TOTAL ESTIMATED MEMORY": self._to_gb(total_mem)
        }

    def calculate_communication(self) -> dict:
        """3. Estimates data transfer volume."""
        return {"Inter-GPU Communication": "0 GB (only one GPU is used)"}

    def estimate_other_factors(self) -> dict:
        """4. Calculates metrics for other factors."""
        return {
            "a) The Memory Wall": "The primary blocker. Total memory must fit in one GPU's VRAM.",
            "b) GPU Memory Bandwidth": "The speed of moving data from HBM to compute cores can be a bottleneck, affecting MFU.",
            "c) CPU/IO Data Loading": "The GPU can be starved if the data pipeline is too slow.",
        }


if __name__ == '__main__':
    llama_7b_config = {'model_params_b': 7, 'hidden_dim': 4096, 'num_layers': 32, 'num_heads': 32, 'vocab_size': 32000}

    # --- SCENARIO CONFIG ---
    # A moderate batch size and sequence length
    GLOBAL_BATCH_SIZE = 8
    SEQUENCE_LENGTH = 2048

    estimator = SingleGPUEstimator(**llama_7b_config)

    print("--- Single GPU (No Parallelism) Estimates for Llama 7B ---")

    # 1. FLOPs
    flops_info = estimator.calculate_flops(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH)
    print("\n1. COMPUTATION (FLOPs):")
    for key, val in flops_info.items(): print(f"   - {key:<35}: {val:.2f}")

    # 2. Memory
    memory_info = estimator.calculate_memory(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH)
    print("\n2. MEMORY BREAKDOWN:")
    param_breakdown = memory_info.pop("Parameter Breakdown")
    print(f"   - DETAILED PARAMETER BREAKDOWN:")
    for key, val in param_breakdown.items(): print(f"     - {key:<30}: {val:.2f}")
    for key, val in memory_info.items(): print(f"   - {key:<35}: {val}")

    # 3. Communication
    comm_info = estimator.calculate_communication()
    print("\n3. DATA TRANSFER:")
    for key, val in comm_info.items(): print(f"   - {key:<35}: {val}")

    # 4. Other Factors
    other_info = estimator.estimate_other_factors()
    print("\n4. OTHER LIMITING FACTORS:")
    for key, val in other_info.items(): print(f"   - {key:<35}: {val}")
    print("\n=========================================================================\n")
