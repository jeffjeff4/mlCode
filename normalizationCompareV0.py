import numpy as np

#-----------------------------------------------------------
#1. Batch Normalization (BatchNorm)
#-----------------------------------------------------------

##Concept: Normalizes the inputs to each layer within a mini-batch. For each feature (or channel in CNNs), it computes the mean and variance across the batch dimension and then normalizes the data to have zero mean and unit variance. During inference, it uses learned running averages of the mean and variance from training.
##When to Use: Widely used in Convolutional Neural Networks (CNNs) and Feed-Forward Networks. It works best with larger batch sizes.
##
##Benefits:
##Reduces Internal Covariate Shift: Makes the distribution of inputs to a layer more stable, allowing higher learning rates.
##Speeds up Training: Networks converge much faster.
##Acts as a Regularizer: Reduces the need for other regularization techniques like Dropout, as it adds some noise due to batch-wise statistics.
##
##Formula:
##y= [(x−E[x]) / sqrt(Var[x]+ϵ)] * γ + β
##
##Where:
##x: input feature (scalar, or feature map element)
##textE[x]: mean of the mini-batch for that feature
##textVar[x]: variance of the mini-batch for that feature
##epsilon: small constant for numerical stability (e.g., 10−5)
##gamma: learnable scaling parameter (initialized to 1)
##beta: learnable shifting parameter (initialized to 0)

import torch
import torch.nn as nn

# Example for a fully connected layer (BatchNorm1d)
# input_features: number of features in the input tensor
batch_norm_1d = nn.BatchNorm1d(num_features=128) # For inputs like (batch_size, num_features)

# Example for a convolutional layer (BatchNorm2d)
# num_channels: number of channels in the input feature map
batch_norm_2d = nn.BatchNorm2d(num_features=64) # For inputs like (batch_size, channels, height, width)

# Example usage:
input_data_1d = torch.randn(32, 128) # Batch size 32, 128 features
output_1d = batch_norm_1d(input_data_1d)
print(f"BatchNorm1d output shape: {output_1d.shape}")

input_data_2d = torch.randn(16, 64, 32, 32) # Batch size 16, 64 channels, 32x32 image
output_2d = batch_norm_2d(input_data_2d)
print(f"BatchNorm2d output shape: {output_2d.shape}")

# In a model:
class ConvNetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 16 * 16, 10) # Assuming 32x32 input image, after 1 conv/pool

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

model = ConvNetWithBN()

batch_size = 32
n_channels = 3
height = 64
width = 16

#input_ids = np.random.randn(batch_size, n_channels, height, width)
input_ids = torch.randn(batch_size, n_channels, height, width)
input_ids = input_ids.float()
input_ids = torch.tensor(input_ids)
output = model(input_ids)

print("bn:")
print(model)
print("bn output[0, :] = ", output[0, :])
print("-----------------------------------------------------------\n")


#-----------------------------------------------------------
#2. Layer Normalization (LayerNorm)
#-----------------------------------------------------------
##Concept: Normalizes across the features of each individual sample within a layer. Unlike BatchNorm, it doesn't depend on the batch size, making it suitable for Recurrent Neural Networks (RNNs) and Transformers, where batch sizes can vary or sequences are involved.
##When to Use: Primarily used in RNNs, Transformers, and other sequence models. Also useful when batch sizes are very small.
##
##Benefits:
##Independent of Batch Size: Stable performance regardless of mini-batch size.
##Effective for RNNs/Transformers: Normalizes activations within a single sequence element or across the hidden states, which can vary greatly in value.
##Faster Training: Can lead to faster convergence in sequence models.
##
##Formula:
##y= (x−E[x]) / sqrt(Var[x]+ϵ) * γ+β
##
##The difference is the scope of textE[x] and textVar[x]: they are calculated over all features of a single sample, not across the batch.

import torch
import torch.nn as nn

# Example for a sequence model (e.g., in a Transformer)
# normalized_shape: the shape of the features to normalize over (e.g., embedding_dim)
layer_norm = nn.LayerNorm(normalized_shape=512) # For inputs like (batch_size, sequence_length, embedding_dim)

# Example usage:
input_data = torch.randn(16, 100, 512) # Batch size 16, sequence length 100, embedding dim 512
output = layer_norm(input_data)
print(f"LayerNorm output shape: {output.shape}")

# In a Transformer block (conceptual):
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.norm1 = nn.LayerNorm(d_model) # Normalizes embedding_dim
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model) # Normalizes embedding_dim

    def forward(self, x):
        # Apply attention and then LayerNorm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output) # Add & Norm (residual connection)
        
        # Apply Feed-Forward Network and then LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # Add & Norm (residual connection)
        return x

batch_size = 32
seq_len = 10
dimension_size = 512

transformer_block = TransformerBlock(d_model=dimension_size)

#input_ids = np.random.randn(batch_size, n_channels, height, width)
input_ids = torch.randn(batch_size, seq_len, dimension_size)
input_ids = input_ids.float()
input_ids = torch.tensor(input_ids)
output = transformer_block(input_ids)

print("layer norm:")
print(transformer_block)
print("layer norm output[0, :] = ", output[0, :])
print("-----------------------------------------------------------\n")



#-----------------------------------------------------------
#3. Instance Normalization (InstanceNorm)
#-----------------------------------------------------------
##Concept: Normalizes each feature map independently for each instance (image) in a batch. It's an extension of LayerNorm, but typically applied per channel, per sample, for image data.
##When to Use: Primarily used in generative models like Style Transfer (e.g., CycleGAN, style-transfer networks) where it's desirable to normalize the style of individual instances, preserving content.
##
##Benefits:
##Style Control: Helps to remove instance-specific variations (like contrast) while preserving content.
##Effective for Image Generation: Crucial for tasks where per-instance style is important.
##
##Formula: Similar to BatchNorm/LayerNorm, but textE[x] and textVar[x] are calculated over the spatial dimensions (H times W) for each channel of each individual sample.

import torch
import torch.nn as nn

# Example for 2D image data (InstanceNorm2d)
# num_features: number of channels in the input
instance_norm_2d = nn.InstanceNorm2d(num_features=64) # For inputs like (batch_size, channels, height, width)

# Example usage:
input_data = torch.randn(16, 64, 32, 32) # Batch size 16, 64 channels, 32x32 image
output = instance_norm_2d(input_data)
print(f"InstanceNorm2d output shape: {output.shape}")

# In a generative network:
class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, padding=4)
        self.in1 = nn.InstanceNorm2d(32) # InstanceNorm after convolution
        self.relu = nn.ReLU()
        # ... other layers

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        return x


style_net = StyleTransferNet()
print(style_net)

batch_size = 32
n_channels = 3
height = 64
width = 16

#input_ids = np.random.randn(batch_size, n_channels, height, width)
input_ids = torch.randn(batch_size, n_channels, height, width)
input_ids = input_ids.float()
input_ids = torch.tensor(input_ids)
output = style_net(input_ids)

print("Instance Normalization:")
print(style_net)
print("Instance Normalization output[0, :] = ", output[0, :])
print("-----------------------------------------------------------\n")



#-----------------------------------------------------------
# 4. Group Normalization (GroupNorm)
#-----------------------------------------------------------

#---------------------------
#part 1
#---------------------------
##Concept: Divides the channels of an input into groups and normalizes the features within each group independently. It's a compromise between BatchNorm (normalizes all channels across the batch) and InstanceNorm (normalizes each channel independently).
##When to Use: When BatchNorm isn't feasible due to very small batch sizes, or when it causes issues. It offers a good balance between the benefits of BatchNorm and the independence of LayerNorm/InstanceNorm.
##
##Benefits:
##Batch Size Independent: Like LayerNorm and InstanceNorm, it works well regardless of batch size.
##More Stable than BatchNorm with small batches.
##Flexible: The num_groups hyperparameter allows tuning how much shared information is used for normalization.
##Formula: Mean and variance are computed over the spatial dimensions and the channels within each group for each sample.


import torch
import torch.nn as nn

# Example of GroupNorm for a 2D input
# num_groups: number of groups to divide the channels into
# num_channels: total number of channels in the input
group_norm = nn.GroupNorm(num_groups=8, num_channels=64) # For 64 channels, 8 groups means 8 channels per group

# Example usage:
input_data = torch.randn(16, 64, 32, 32) # Batch size 16, 64 channels, 32x32 image
output = group_norm(input_data)
print(f"GroupNorm output shape: {output.shape}")

# In a CNN:
class ConvNetWithGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64) # 8 groups for 64 channels
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ... rest of the network

    def forward(self, x):
        x = self.pool(self.relu(self.gn1(self.conv1(x))))
        return x

model = ConvNetWithGN()

batch_size = 32
n_channels = 3
height = 64
width = 16

#input_ids = np.random.randn(batch_size, n_channels, height, width)
input_ids = torch.randn(batch_size, n_channels, height, width)
input_ids = input_ids.float()
input_ids = torch.tensor(input_ids)
output = model(input_ids)

print("group norm:")
print(model)
print("group norm output[0, :] = ", output[0, :])
print("-----------------------------------------------------------\n")

#---------------------------
#part 2
#---------------------------

##For Group Normalization (GroupNorm), the input x is typically a batch of samples (e.g., images), where normalization is applied independently to each sample in the batch. However, because its calculation does not depend on the batch dimension, it can also be conceptually applied to a single sample.
##
##Group Normalization Input Explained 🧠
##Let's break this down:
##The formula for normalization (be it Batch, Layer, Instance, or Group) fundamentally involves calculating mean (
##textE[x]) and variance (textVar[x]) over certain dimensions, and then using these statistics to normalize the input.
##
##For GroupNorm(num_groups, num_channels), the mean and variance are computed over the spatial dimensions (Height and Width) and over the channels within each group, for each individual sample in the batch.
##
##Consider an input tensor x with shape (N, C, H, W):
##N: Batch size (number of samples)
##C: Number of channels
##H: Height
##W: Width
##
##When GroupNorm is applied:
##The C channels are divided into num_groups groups.
##For each individual sample n in the batch:
##For each group of channels g:
##The mean and variance are calculated across all channels within that group g and across all spatial locations H and W.
##
##These calculated statistics are then used to normalize the values of that specific group of channels for that specific sample.
##Because the normalization statistics (mean and variance) are calculated per sample and per group of channels, the computation is independent of the batch size N. This is why GroupNorm works well even with batch sizes of 1.


import torch
import torch.nn as nn

# Assume an input tensor of shape (N, C, H, W)
N_batch_size = 4  # Number of samples in the batch
C_channels = 64   # Number of channels
H_height = 32     # Height of feature map
W_width = 32      # Width of feature map

num_groups = 8    # Divide 64 channels into 8 groups, so 8 channels per group

# 1. Input is a batch of samples (common scenario)
batch_input = torch.randn(N_batch_size, C_channels, H_height, W_width)
group_norm_layer = nn.GroupNorm(num_groups, C_channels)
output_batch = group_norm_layer(batch_input)

print(f"1. Input shape (batch): {batch_input.shape}")
print(f"   Output shape (batch): {output_batch.shape}")
print(f"   Output values (first sample, first group of channels):\n{output_batch[0, :8, 0, 0]}")
print("-" * 50)

# 2. Input is a single sample (batch size = 1)
single_sample_input = torch.randn(1, C_channels, H_height, W_width)
output_single_sample = group_norm_layer(single_sample_input)

print(f"2. Input shape (single sample): {single_sample_input.shape}")
print(f"   Output shape (single sample): {output_single_sample.shape}")
print(f"   Output values (first sample, first group of channels):\n{output_single_sample[0, :8, 0, 0]}")
print("-" * 50)

# 3. What if you manually try to process one sample from a batch through a new layer?
# The GroupNorm layer doesn't distinguish if the batch dimension is 1 or more,
# as it normalizes independently for each "N" unit.
# The internal logic is "per-sample, per-group".


##Output Interpretation:
##As you can see from the output shapes and the fact that the code runs without error for both batch and single-sample inputs, GroupNorm can handle both. The core reason is that the normalization statistics are computed independently for each sample within its designated channel groups.
##Therefore, GroupNorm is robust to batch size changes, a significant advantage over BatchNorm when dealing with small batches or variable batch sizes, for example, in tasks where memory constraints or architectural design (like certain GANs or segmentation networks) might limit batch size.

#-----------------------------------------------------------
# 5. Weight Normalization
#-----------------------------------------------------------

#-----------------------------------
# part 1
#-----------------------------------

##Concept: Decouples the magnitude of a weight vector from its direction. Instead of normalizing the activations, it normalizes the weights of a layer. It reparameterizes each weight vector
##mathbfw as
##w = g * v / ∣∣v∣∣,
##where g is a scalar magnitude parameter and v is a vector direction parameter.
##When to Use: Can be applied to any layer with weights (e.g., nn.Linear, nn.Conv2d). Often used in Generative Adversarial Networks (GANs) or when you need more explicit control over the scale of weights.
##
##Benefits:
##Faster Convergence: Can speed up optimization, similar to batch normalization, but without relying on batch statistics.
##Stable Training: Helps prevent exploding gradients by controlling the scale of weights.
##Less Memory Intensive: Doesn't store running statistics, making it more memory efficient than BatchNorm.


import torch
import torch.nn as nn
import torch.nn.utils as utils

# Define a simple linear layer
linear_layer = nn.Linear(10, 5)
print(f"Original linear layer weight shape: {linear_layer.weight.shape}")

# Apply weight normalization to the 'weight' parameter of the linear layer
# dim=0 means normalize across the output features (rows for weight matrix)
weight_normed_layer = utils.weight_norm(linear_layer, name='weight', dim=0)

# Now, the 'weight' parameter is replaced by 'weight_g' (magnitude) and 'weight_v' (direction)
print(f"Weight normalized layer: {weight_normed_layer}")
print(f"Magnitude parameter (weight_g) shape: {weight_normed_layer.weight_g.shape}")
print(f"Direction parameter (weight_v) shape: {weight_normed_layer.weight_v.shape}")

# Example usage (forward pass works as usual):
input_data = torch.randn(32, 10)
output = weight_normed_layer(input_data)
print(f"WeightNormedLayer output shape: {output.shape}")

# In a model:
class SimpleNetWithWN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = utils.weight_norm(nn.Linear(768, 256), name='weight', dim=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten for FC layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNetWithWN()

batch_size = 32
n_channels = 3
height = 16
width = 16

#input_ids = np.random.randn(batch_size, n_channels, height, width)
input_ids = torch.randn(batch_size, n_channels, height, width)
input_ids = input_ids.float()
input_ids = torch.tensor(input_ids)
output = model(input_ids)

print("weight norm:")
print(model)
print("weight norm output[0, :] = ", output[0, :])
print("-----------------------------------------------------------\n")


#-----------------------------------
# part 2
#-----------------------------------

##This equation,
##w = g * v / ∥v∥ ,
##describes Weight Normalization. It's a reparameterization technique used in deep learning to separate the magnitude of a weight vector from its direction.
##
##Breakdown of the Equation 🔬
##Let's break down each component:
##
##1. w: This represents a weight vector in a neural network layer. For instance, in a linear layer, if you have 10 input features and 5 output features, the weight matrix would be 5×10. Each row (or column, depending on convention) of this matrix is a weight vector corresponding to an output feature or an input feature.
##2. g: This is a scalar magnitude parameter. It's a single learnable number that controls the length or scale of the weight vector w. Think of it as a knob that adjusts how "strong" the connections are.
##3. v: This is a vector direction parameter. It's a learnable vector that determines the direction of the weight vector w. It has the same dimensionality as w.
##4. v / ∥v∥:
## : This term is the unit vector in the direction of v. ∥v∥ represents the Euclidean (L2) norm of the vector v. Dividing v by its norm effectively normalizes its length to 1, so it only retains its direction.
##
##In summary: The equation states that the original weight vector w is re-expressed as the product of a scalar magnitude g and a unit vector derived from v.
##
##Why is this done? (The Purpose of Weight Normalization) 🤔
##This reparameterization has several key benefits in deep learning:
##
##1. Decoupling Magnitude and Direction:
##    1) By separating g from v, the optimizer can adjust the magnitude (g) and the direction (v) of the weights independently.
##    2) This can lead to more stable and efficient optimization. For instance, updating the direction might not require also changing the scale, or vice-versa.
##
##2. Faster Convergence:
##    1) It helps accelerate the convergence of deep neural networks. Similar to Batch Normalization, it aims to reduce "internal covariate shift" by stabilizing the scale of activations. However, unlike BatchNorm, it doesn't rely on batch statistics, which can be unstable with small batch sizes.
##    2) The gradient computation becomes simpler and more stable because the scale is explicitly controlled.
##
##3. No Batch Dependence:
##    1) Unlike Batch Normalization, Weight Normalization does not use mini-batch statistics. This means its behavior is consistent regardless of the batch size. This makes it particularly useful in scenarios where batch sizes are very small (e.g., certain reinforcement learning tasks, GANs, or when memory is limited).
##
##4. Reduced Overhead:
##    It doesn't introduce extra layers or require storing running statistics (like BatchNorm does for inference), making it more memory-efficient and potentially faster during inference.
##
##5. Controlling Exploding/Vanishing Gradients:
##    By explicitly controlling the magnitude of the weights through g, it can help prevent weights from becoming too large (which can cause exploding gradients) or too small (which can cause vanishing gradients).
##
##In essence, Weight Normalization offers an alternative or complementary way to stabilize training and improve performance by providing a more explicit and decoupled control over the properties of the weight vectors.




#-----------------------------------------------------------
# 6. L2 Normalization (Unit Length Normalization)
#-----------------------------------------------------------

##Concept: This isn't an activation normalization like the others, but a common technique to normalize vectors (e.g., feature embeddings) to have a unit Euclidean norm (length of 1). It often used for outputs of specific layers or for input feature vectors.
##When to Use: Common in tasks involving similarity measures (e.g., in metric learning, recommender systems, or where feature vectors are compared using cosine similarity).
##
##Benefits:
##Consistent Scale: Ensures all vectors have the same magnitude, making distance/similarity calculations more meaningful.
##Focus on Direction: Emphasizes the direction of the vector over its magnitude.
##
##Formula:
##y= x / ∥x∥2 = x / sqrt[∑i (x_i)^2]
##Where x is the input vector.


import torch
import torch.nn.functional as F

# Example for L2 normalization
input_vector = torch.randn(5)
print(f"Original vector: {input_vector}")
print(f"Original L2 norm: {torch.norm(input_vector, p=2)}")

# Apply L2 normalization
# dim: the dimension along which to normalize (default is 1)
# For a 1D tensor, dim=0 works. For a batch of vectors (N, D), dim=1
normalized_vector = F.normalize(input_vector, p=2, dim=0)
print(f"Normalized vector: {normalized_vector}")
print(f"Normalized L2 norm: {torch.norm(normalized_vector, p=2)}")

# Example with a batch of vectors
batch_of_vectors = torch.randn(4, 10) # 4 samples, 10-dimensional vectors
print(f"\nOriginal batch of vectors:\n{batch_of_vectors}")
print(f"Original L2 norms per vector: {torch.norm(batch_of_vectors, p=2, dim=1)}")

normalized_batch = F.normalize(batch_of_vectors, p=2, dim=1) # Normalize each row (vector)
print(f"Normalized batch of vectors:\n{normalized_batch}")
print(f"Normalized L2 norms per vector: {torch.norm(normalized_batch, p=2, dim=1)}")

# In a model (e.g., for learned embeddings before comparison):
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        # Normalize embeddings to unit length for cosine similarity
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1) # Normalize last dimension
        return normalized_embeddings

model = EmbeddingModel(vocab_size=10000, embedding_dim=256)
input_ids = torch.randint(0, 10000, (32, 5)) # Batch of 32 sequences, length 5
output = model(input_ids)
print(f"\nEmbedding model output shape: {output.shape}")
print(f"L2 norm of first embedding: {torch.norm(output[0, 0], p=2)}")


#-----------------------------------------------------------
# conclusion
#-----------------------------------------------------------

##Choosing the Right Normalization Method
##The choice of normalization method depends heavily on your network architecture, data type, and the specific task:
##
##    1. Batch Normalization: Still a strong default for CNNs, especially with sufficiently large batch sizes.
##
##    2. Layer Normalization: Preferred for sequence models (RNNs, Transformers) and situations with highly varying or small batch sizes.
##
##    3. Instance Normalization: Best for style transfer and certain generative tasks where per-instance normalization is desired.
##
##    4. Group Normalization: A good alternative when BatchNorm struggles with small batch sizes, offering a configurable trade-off between BatchNorm and InstanceNorm.
##
##    5. Weight Normalization: Can be used as an alternative or complement to activation normalizations, especially in GANs or for fine-grained control over weight scale.
##
##    6. L2 Normalization: Useful for processing feature vectors or embeddings where their direction, rather than their raw magnitude, is most important for downstream tasks.
##
##Understanding these differences helps you make informed decisions to build more stable and powerful deep learning models!


#------------------------------------------------------------------------------
# differences among layer norm, weighted norm, L2 norm
#------------------------------------------------------------------------------
##Feature	            |    Layer Normalization (LayerNorm)             |   	Weight Normalization (WeightNorm)                   |   	L2 Normalization
##What it acts on	    |   Activations of a layer	                     |       Weights of a layer	                                |       Any vector (e.g., embeddings, features)
##Primary Goal  	    |   Stabilize training dynamics of activations	 |       Improve optimization of weights	                |       Ensure unit length for a vector
##Scope of Stats        |	Per sample, across features/channels	     |       Per weight vector (no batch stats)	                |        Per individual vector
##Parameters            |	Learnable gamma (scale) and beta (shift)     |    	Learnable g (magnitude) and mathbfv (direction)     |	    No learnable parameters (just a mathematical operation)
##Batch Dependence      |	Independent of batch size	                 |      Independent of batch size	                        |       Independent of batch size (per vector)
##Typical Use           |	RNNs, Transformers, small batches	         |       GANs, specific layers where BatchNorm is not ideal	|       Embeddings, similarity measures, regularization