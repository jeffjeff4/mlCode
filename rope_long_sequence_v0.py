import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
max_position = 100  # Maximum sequence length to visualize
d_model = 64  # Embedding dimension


# 1. Generate Sinusoidal Positional Embeddings
def get_positional_embeddings(max_pos, dim):
    pos = np.arange(max_pos)[:, np.newaxis]
    i = np.arange(dim)[np.newaxis, :]

    # Base formula for frequency calculation: 1 / (10000^(2i/d_model))
    freqs = 1.0 / np.power(10000, (2 * (i // 2)) / np.float32(dim))

    pos_enc = pos * freqs
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # Apply sine to even indices
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # Apply cosine to odd indices
    return pos_enc, freqs[0]


embeddings, tracking_freqs = get_positional_embeddings(max_position, d_model)

# 2. Compute Position-to-Position Similarity Matrix (Correlation)
# Dot product of normalized embeddings represents cosine similarity/correlation
norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
correlation_matrix = np.dot(norm_embeddings, norm_embeddings.T)

# 3. Plotting the Figures
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Figure 1: Frequency across different embedding dimensions
# We track how the wave patterns span across early vs deep positions
axes[0].plot(embeddings[:, 2], label="Dimension 2 (High Freq / Close Range)", color='#1f77b4', lw=2)
axes[0].plot(embeddings[:, 16], label="Dimension 16 (Medium Freq)", color='#ff7f0e', lw=2)
axes[0].plot(embeddings[:, 48], label="Dimension 48 (Low Freq / Long Range)", color='#2ca02c', lw=2)
axes[0].set_title("Wave Frequency Variation across Embedding Dimensions", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Token Position Index", fontsize=10)
axes[0].set_ylabel("Embedding Value Amplitude", fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc="upper right")

# Figure 2: Position Correlation Matrix Heatmap
sns.heatmap(correlation_matrix, ax=axes[1], cmap="viridis", cbar_kws={'label': 'Correlation (Cosine Similarity)'})
axes[1].set_title("Position-to-Position Correlation Matrix", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Token Position Index", fontsize=10)
axes[1].set_ylabel("Token Position Index", fontsize=10)

# Highlighting Specific Correlations on the Heatmap
# 1) Close Positions (Diagonal band)
axes[1].add_patch(
    plt.Rectangle((10, 10), 15, 15, fill=False, edgecolor='red', lw=2.5, label='Close Position Correlation'))
# 2) Long Distance Positions (Corners away from diagonal)
axes[1].add_patch(
    plt.Rectangle((5, 80), 15, 15, fill=False, edgecolor='cyan', lw=2.5, label='Long Distance Correlation'))

plt.tight_layout()
plt.show()