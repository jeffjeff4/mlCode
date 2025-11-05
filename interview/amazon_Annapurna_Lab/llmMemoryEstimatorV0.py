####please write code to estimate
####given a model, say llama 7b, estimate memory consumption, parameters, how much memory, if activation uses a lot memory, calculate it


import math


class LLMMemoryEstimator:
    """
    A class to estimate the memory consumption of a Large Language Model
    for both training and inference scenarios.
    """

    def __init__(self, model_params_b: float, hidden_dim: int, num_layers: int, num_heads: int):
        """
        Initializes the estimator with the model's architecture.

        Args:
            model_params_b (float): Number of model parameters in billions.
            hidden_dim (int): The size of the hidden dimension (d_model).
            num_layers (int): The number of transformer layers.
            num_heads (int): The number of attention heads.
        """
        self.params_count = model_params_b * 1e9
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

    def _to_gb(self, bytes_val: float) -> float:
        """Converts bytes to gigabytes."""
        return bytes_val / (1024 ** 3)

    def calculate_parameter_memory(self) -> dict:
        """
        Calculates the memory required to store the model's parameters at various precisions.
        """
        precisions = {
            'FP32': 4,
            'FP16/BF16': 2,
            'INT8': 1,
            'INT4': 0.5
        }
        memory_breakdown = {}
        for name, bytes_per_param in precisions.items():
            mem_bytes = self.params_count * bytes_per_param
            memory_breakdown[name] = self._to_gb(mem_bytes)
        return memory_breakdown

    def calculate_optimizer_memory(self, model_precision_bytes: int = 2) -> dict:
        """
        Calculates the memory for optimizer states (assuming AdamW).
        Adam optimizer stores 2 states (momentum, variance) typically in FP32 (4 bytes each).
        The gradients are stored at the model's precision.
        """
        grad_mem = self.params_count * model_precision_bytes
        momentum_mem = self.params_count * 4
        variance_mem = self.params_count * 4

        total_mem = grad_mem + momentum_mem + variance_mem

        return {
            'Gradients (e.g., FP16)': self._to_gb(grad_mem),
            'Momentum (FP32)': self._to_gb(momentum_mem),
            'Variance (FP32)': self._to_gb(variance_mem),
            'Total Optimizer State': self._to_gb(total_mem)
        }

    def calculate_activation_memory(self, batch_size: int, seq_len: int, activation_precision_bytes: int = 2) -> dict:
        """
        Calculates the memory required for activations during training.
        This includes a high-level approximation and a specific calculation for the attention scores.
        """
        # High-level approximation for the output of each layer
        approx_total_activations = batch_size * seq_len * self.hidden_dim * self.num_layers * activation_precision_bytes

        # Specific calculation for the attention score matrix (often a bottleneck)
        # Formula: B * H * S * S
        attention_scores = batch_size * self.num_heads * seq_len * seq_len * activation_precision_bytes

        return {
            'Total Activation (Approximation)': self._to_gb(approx_total_activations),
            'Attention Scores Only (Per Layer)': self._to_gb(attention_scores),
            'Attention Scores (All Layers)': self._to_gb(attention_scores * self.num_layers)
        }

    def calculate_kv_cache_memory(self, batch_size: int, seq_len: int, kv_cache_precision_bytes: int = 2) -> float:
        """
        Calculates the memory for the Key-Value (KV) cache during inference.
        Formula: B * S * N_layers * D_hidden * 2 (for K and V)
        """
        kv_cache_bytes = batch_size * seq_len * self.num_layers * self.hidden_dim * 2 * kv_cache_precision_bytes
        return self._to_gb(kv_cache_bytes)

    def estimate_training_memory(self, batch_size: int, seq_len: int, model_precision: str = 'FP16') -> dict:
        """
        Provides a full estimate for training memory requirements.
        """
        bytes_per_param = {'FP32': 4, 'FP16': 2}[model_precision]

        param_mem = self._to_gb(self.params_count * bytes_per_param)
        optim_mem = self.calculate_optimizer_memory(bytes_per_param)['Total Optimizer State']
        # Use a more realistic estimation for activations, often much larger than the simple formula
        # A common heuristic is that it's of a similar magnitude to optimizer states for large sequences.
        activ_mem = self.calculate_activation_memory(batch_size, seq_len, bytes_per_param)[
            'Total Activation (Approximation)']

        # Add a buffer for CUDA context, fragmentation, and other overhead
        misc_mem = 5.0

        total_mem = param_mem + optim_mem + activ_mem + misc_mem

        return {
            'Model Parameters': param_mem,
            'Optimizer States': optim_mem,
            'Activations (Approximation)': activ_mem,
            'Miscellaneous': misc_mem,
            'ESTIMATED TOTAL TRAINING MEMORY': total_mem
        }

    def estimate_inference_memory(self, batch_size: int, seq_len: int, model_precision: str = 'FP16') -> dict:
        """
        Provides a full estimate for inference memory requirements.
        """
        bytes_per_param = {'FP16': 2, 'INT8': 1, 'INT4': 0.5}.get(model_precision, 2)

        param_mem = self._to_gb(self.params_count * bytes_per_param)
        kv_cache_mem = self.calculate_kv_cache_memory(batch_size, seq_len, bytes_per_param)

        # Buffer for activations (much smaller than training) and other overhead
        misc_mem = 2.0

        total_mem = param_mem + kv_cache_mem + misc_mem

        return {
            'Model Parameters': param_mem,
            'KV Cache': kv_cache_mem,
            'Miscellaneous/Workspace': misc_mem,
            'ESTIMATED TOTAL INFERENCE MEMORY': total_mem
        }


def display_results(title: str, data: dict):
    """Helper function to print results in a readable format."""
    print(f"--- {title} ---")
    for key, value in data.items():
        if isinstance(value, float):
            print(f"{key:<35}: {value:.2f} GB")
        else:
            print(f"{key:<35}: {value}")
    print("-" * (len(title) + 6), "\n")


if __name__ == '__main__':
    # --- Configuration for Llama 7B ---
    llama_7b_config = {
        'model_params_b': 7,
        'hidden_dim': 4096,
        'num_layers': 32,
        'num_heads': 32
    }

    # --- Scenario Configuration ---
    # Typical scenario for fine-tuning or training
    training_batch_size = 4
    training_seq_len = 2048

    # Typical scenario for inference
    inference_batch_size = 1
    inference_seq_len = 2048

    # --- Create Estimator and Run Calculations ---
    estimator = LLMMemoryEstimator(**llama_7b_config)

    # 1. Calculate Parameter Memory at different precisions
    param_memory = estimator.calculate_parameter_memory()
    display_results("Model Parameter Memory", param_memory)

    # 2. Calculate a full training estimate
    training_estimate = estimator.estimate_training_memory(training_batch_size, training_seq_len,
                                                           model_precision='FP16')
    display_results(f"Full Training Estimate (Batch Size: {training_batch_size}, Seq Len: {training_seq_len})",
                    training_estimate)

    # 3. Calculate activation memory specifically to show its scale
    activation_details = estimator.calculate_activation_memory(training_batch_size, training_seq_len)
    display_results("Detailed Activation Memory Breakdown", activation_details)

    # 4. Calculate a full inference estimate
    inference_estimate = estimator.estimate_inference_memory(inference_batch_size, inference_seq_len,
                                                             model_precision='FP16')
    display_results(f"Inference Estimate (Batch Size: {inference_batch_size}, Seq Len: {inference_seq_len})",
                    inference_estimate)
