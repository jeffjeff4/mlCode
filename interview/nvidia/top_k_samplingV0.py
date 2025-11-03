import numpy as np


def top_k_sampling(logits, k):
    # 1️⃣ Get top-k indices
    top_k_indices = np.argpartition(logits, -k)[-k:]

    # 2️⃣ Get corresponding logits and normalize via softmax
    top_k_logits = logits[top_k_indices]
    exp_vals = np.exp(top_k_logits - np.max(top_k_logits))  # numerical stability
    probs = exp_vals / np.sum(exp_vals)

    # 3️⃣ Sample one token index from top-k based on probabilities
    sampled_idx = np.random.choice(top_k_indices, p=probs)

    return int(sampled_idx)


# ✅ Test
np.random.seed(0)
logits = np.array([1.0, 2.0, 3.0, 0.5])
print(top_k_sampling(logits, 2))  # likely 2 or 1, depending on sampling
