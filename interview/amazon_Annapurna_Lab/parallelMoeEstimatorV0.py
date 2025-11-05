import math


class MoEParallelEstimator:
    """
    Estimates per-GPU resources for training a Mixture of Experts (MoE) model
    using Expert Parallelism combined with Data Parallelism.
    """

    def __init__(self, base_model_params_b: float, hidden_dim: int, num_moe_layers: int,
                 num_experts_total: int, num_experts_per_token: int):
        self.base_params = base_model_params_b * 1e9
        self.hidden_dim = hidden_dim
        # MoE layers replace FFNs, so we use this to calculate expert size and count
        self.num_moe_layers = num_moe_layers
        self.num_experts_total = num_experts_total
        self.num_experts_per_token = num_experts_per_token  # The 'top_k' value

        # Calculate the size of one expert FFN
        # An FFN has an up-projection and a down-projection
        ffn_params = (self.hidden_dim * self.hidden_dim * 4) + (self.hidden_dim * 4 * self.hidden_dim)
        self.expert_params = ffn_params
        self.total_expert_params = self.num_experts_total * self.expert_params
        self.total_model_params = self.base_params + self.total_expert_params

    def _to_gb(self, bytes_val: float) -> float:
        """Converts bytes to gigabytes."""
        return bytes_val / (1024 ** 3)

    def calculate_per_gpu_flops(self, global_batch_size: int, seq_len: int, k: int) -> dict:
        """
        1. Calculates the computational FLOPs per GPU for one iteration.
        FLOPs depend on ACTIVE parameters, not total parameters.
        """
        if k <= 0: return {}

        # Computation for the dense base model (replicated, so work is split by data)
        base_flops = 6 * self.base_params * (global_batch_size / k) * seq_len

        # Computation for the sparsely activated experts
        active_expert_params = self.num_experts_per_token * self.expert_params
        expert_flops = self.num_moe_layers * (6 * active_expert_params * (global_batch_size / k) * seq_len)

        total_flops_per_gpu = base_flops + expert_flops

        return {
            "Total Active Params per Token (B)": (self.base_params + active_expert_params * self.num_moe_layers) / 1e9,
            "Total Model Params (B)": self.total_model_params / 1e9,
            "FLOPs per GPU per Iter (PetaFLOPs)": total_flops_per_gpu / 1e15
        }

    def calculate_per_gpu_memory(self, global_batch_size: int, seq_len: int, k: int,
                                 model_precision_bytes: int = 2) -> dict:
        """
        2. Calculates per-GPU memory consumption.
        Base model is replicated (DP), experts are sharded (EP).
        """
        if k <= 0: return {}

        # a) Replicated Components: The dense base model.
        base_param_mem = self.base_params * model_precision_bytes
        base_optimizer_mem = self.base_params * 4 * 2  # Adam states

        # b) Sharded Components: The experts.
        experts_per_gpu = self.num_experts_total / k
        expert_param_mem = experts_per_gpu * self.expert_params * model_precision_bytes
        expert_optimizer_mem = experts_per_gpu * self.expert_params * 4 * 2

        # c) Per-GPU Activations (based on local batch size and active model size).
        per_gpu_batch_size = global_batch_size / k
        # Simplified: activation memory is proportional to the number of dense layers + MoE layers
        activation_mem = per_gpu_batch_size * seq_len * self.hidden_dim * self.num_moe_layers * model_precision_bytes

        misc_overhead = 5 * (1e9)  # 5 GB
        total_mem = base_param_mem + base_optimizer_mem + expert_param_mem + expert_optimizer_mem + activation_mem + misc_overhead

        return {
            "Replicated Base Model (Params+Optim)": self._to_gb(base_param_mem + base_optimizer_mem),
            "Sharded Experts (Params+Optim)": self._to_gb(expert_param_mem + expert_optimizer_mem),
            "Activations (for local batch)": self._to_gb(activation_mem),
            "Misc Overhead": self._to_gb(misc_overhead),
            "TOTAL ESTIMATED MEMORY": self._to_gb(total_mem)
        }

    def calculate_per_gpu_communication(self, global_batch_size: int, seq_len: int, k: int,
                                        model_precision_bytes: int = 2) -> dict:
        """
        3. Estimates data transfer from All-to-All (token shuffling).
        """
        if k <= 1:
            return {"Total Communication per Iteration (GB)": 0}

        # The data being shuffled is the activation for each token.
        token_activation_size = self.hidden_dim * model_precision_bytes
        total_data_to_shuffle = global_batch_size * seq_len * token_activation_size

        # Effective volume per GPU using the 2*(k-1)/k model for All-to-All.
        comm_per_all_to_all = 2 * ((k - 1) / k) * total_data_to_shuffle

        # This happens twice per MoE layer (tokens to experts, results from experts).
        total_comm = self.num_moe_layers * 2 * comm_per_all_to_all

        return {
            "Total Communication per Iteration (GB)": self._to_gb(total_comm)
        }

    def estimate_other_factors(self, k: int, global_batch_size: int, seq_len: int,
                               capacity_factor: float = 1.25) -> dict:
        """
        4. Calculates metrics for other factors, especially load balancing.
        """
        if k <= 0: return {}

        avg_tokens_per_expert = (global_batch_size * seq_len) / self.num_experts_total
        expert_capacity = avg_tokens_per_expert * capacity_factor

        return {
            "a) Load Balancing Factor": "A key challenge. Uneven token routing creates stragglers.",
            "b) Expert Capacity": f"{expert_capacity:.2f} tokens",
            "c) Dropped Tokens Risk": "If routing exceeds capacity, tokens are dropped, affecting accuracy.",
            "d) Communication Pattern": "All-to-All is the primary bottleneck, sensitive to network bandwidth."
        }


if __name__ == '__main__':
    # Configuration for a Llama 7B-style model converted to an MoE.
    # The base model contains everything EXCEPT the FFNs which are replaced by experts.
    # A Llama 7B FFN is ~134M params per layer. Total FFN params = 32 * 134M = ~4.3B
    base_model_params = 7 - 4.3  # ~2.7B params for attention, embeddings, etc.

    estimator = MoEParallelEstimator(
        base_model_params_b=base_model_params,
        hidden_dim=4096,
        num_moe_layers=32,  # All 32 FFN layers are replaced with MoE layers
        num_experts_total=64,  # 64 experts in total
        num_experts_per_token=2  # Route each token to the top 2 experts
    )

    GLOBAL_BATCH_SIZE = 16
    SEQUENCE_LENGTH = 4096

    print("--- MoE (Expert Parallelism) Scaling Estimates for a 7B-Active-Param Model ---")

    for k_gpus in [1, 2, 4, 8]:
        print(f"\n========================= ESTIMATE FOR k = {k_gpus} GPU(s) =========================")

        flops_info = estimator.calculate_per_gpu_flops(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        memory = estimator.calculate_per_gpu_memory(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        comms = estimator.calculate_per_gpu_communication(GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH, k_gpus)
        other = estimator.estimate_other_factors(k_gpus, GLOBAL_BATCH_SIZE, SEQUENCE_LENGTH)

        print(f"1. PER-GPU COMPUTATION (FLOPs):")
        print(f"   - Total Model Params (Sparse)       : {flops_info['Total Model Params (B)']:.2f} B")
        print(f"   - Active Params per Token           : {flops_info['Total Active Params per Token (B)']:.2f} B")
        print(
            f"   - FLOPs per GPU per Iter            : {flops_info['FLOPs per GPU per Iter (PetaFLOPs)']:.2f} PetaFLOPs")

        print("\n2. PER-GPU MEMORY BREAKDOWN:")
        for key, val in memory.items(): print(f"   - {key:<35}: {val:.2f} GB")

        print("\n3. PER-GPU DATA TRANSFER (All-to-All Token Shuffling):")
        for key, val in comms.items(): print(f"   - {key:<35}: {val:.2f} GB")

        print("\n4. OTHER LIMITING FACTORS (MoE Specifics):")
        for key, val in other.items(): print(f"   - {key:<35}: {val}")

    print("=========================================================================\n")

