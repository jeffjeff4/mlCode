import torch
import torch.nn as nn

#1. Absolute Position Embeddings
#(a) Learned Absolute Position Embedding
#Treat each position as an embedding (just like words).
#Example: BERT uses this.

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: shape [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_embed = self.pos_embedding(positions)  # [1, seq_len, d_model]
        return x + pos_embed

# Example
x = torch.randn(2, 5, 16)  # [batch=2, seq_len=5, d_model=16]
pe = LearnedPositionalEncoding(max_len=50, d_model=16)
print(pe(x).shape)  # [2, 5, 16]


#(b) Sinusoidal Absolute Position Embedding
#Fixed, not learned.
#Encodes position using sine and cosine functions at different frequencies.
#Original Transformer (Vaswani et al. 2017) used this.
#Formula:
#    𝑃𝐸(𝑝𝑜𝑠, 2𝑖) = sin(𝑝𝑜𝑠 / pow(10000, 2𝑖/𝑑),
#    𝑃𝐸(𝑝𝑜𝑠, 2𝑖+1) = cos(𝑝𝑜𝑠 / pow(10000, 2𝑖/𝑑)



import math

def sinusoidal_position_encoding(seq_len, d_model):
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
    i = torch.arange(d_model, dtype=torch.float).unsqueeze(0)    # [1, d_model]
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / d_model)
    angle_rads = pos * angle_rates
    # apply sin to even indices; cos to odd indices
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe

pe = sinusoidal_position_encoding(seq_len=10, d_model=16)
print(pe.shape)  # [10, 16]

#2. Relative Position Embeddings
#Instead of absolute positions, encode the relative distance between tokens.
#Used in Transformer-XL, T5, DeBERTa.
#Better for generalizing to longer sequences.
#Example: relative attention bias (T5 style)
#    Attention(𝑄, 𝐾) = 𝑄 𝐾^⊤ / sqrt(𝑑_k) + 𝑏_rel(𝑖 − 𝑗)
#where 𝑏_𝑟𝑒𝑙 depends on the distance between positions 𝑖 and 𝑗.


class RelativePositionalBias(nn.Module):
    def __init__(self, num_heads, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Embedding(2 * max_distance + 1, num_heads)

    def forward(self, seq_len):
        # distances matrix
        pos = torch.arange(seq_len)
        rel_dist = pos[None, :] - pos[:, None]  # [seq_len, seq_len]
        rel_dist = rel_dist.clamp(-self.max_distance, self.max_distance)
        rel_dist += self.max_distance  # shift to [0, 2*max_distance]
        # lookup bias
        bias = self.relative_bias(rel_dist)  # [seq_len, seq_len, num_heads]
        return bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]

# Example
bias_layer = RelativePositionalBias(num_heads=4, max_distance=10)
bias = bias_layer(seq_len=6)
print(bias.shape)  # [4, 6, 6]



#3. Rotary Position Embeddings (RoPE)
#Popular in GPT-NeoX, LLaMA, ChatGLM.
#Instead of adding embeddings, it rotates query & key vectors in complex plane according to position.
#Supports extrapolation beyond training length.
#Idea:
#For each 2D pair (𝑥_2𝑖, 𝑥_(2𝑖+1), apply a rotation:

def rotary_embedding(x, seq_len, base=10000):
    """
    x: [batch, seq_len, d_model]
    """
    d_model = x.shape[-1]
    pos = torch.arange(seq_len, device=x.device).float()
    i = torch.arange(d_model//2, device=x.device).float()
    theta = pos[:, None] / (base ** (2 * i / d_model))  # [seq_len, d_model/2]

    cos = torch.cos(theta).repeat_interleave(2, dim=1)  # [seq_len, d_model]
    sin = torch.sin(theta).repeat_interleave(2, dim=1)

    x1 = x * cos + torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1) * sin
    return x1

# Example
x = torch.randn(2, 5, 8)  # batch=2, seq_len=5, d_model=8
out = rotary_embedding(x, seq_len=5)
print(out.shape)  # [2, 5, 8]


#✅ Summary Table
#Method	                        Used in	                              Idea	                        Generalizes to longer seq?
#Learned Absolute	            BERT	                        Learn pos embedding table	        ❌ Limited to max length
#Sinusoidal Absolute	        Transformer (2017)	            Fixed sin/cos function	            ✅ Yes
#Relative Position Bias	        T5, Transformer-XL, DeBERTa	    Encode distance in attention	    ✅ Yes
#Rotary Positional Embedding	GPT-NeoX, LLaMA	                Rotate Q,K vectors by position	    ✅ Yes (good extrapolation)

