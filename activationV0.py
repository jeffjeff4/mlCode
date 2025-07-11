import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create a Sigmoid activation function
sigmoid_activation = nn.Sigmoid()

# Generate some input values
x = torch.linspace(-5, 5, 100)

# Apply the activation function
y_sigmoid = sigmoid_activation(x)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_sigmoid.numpy(), label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Example with a tensor
input_tensor = torch.tensor([-3.0, 0.0, 3.0])
output_tensor = sigmoid_activation(input_tensor)
print(f"Sigmoid input: {input_tensor}")
print(f"Sigmoid output: {output_tensor}\n")


# Create a Tanh activation function
tanh_activation = nn.Tanh()

# Apply the activation function
y_tanh = tanh_activation(x)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_tanh.numpy(), label='Tanh', color='orange')
plt.title('Tanh Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Example with a tensor
input_tensor = torch.tensor([-3.0, 0.0, 3.0])
output_tensor = tanh_activation(input_tensor)
print(f"Tanh input: {input_tensor}")
print(f"Tanh output: {output_tensor}\n")

# Create a ReLU activation function
relu_activation = nn.ReLU()

# Apply the activation function
y_relu = relu_activation(x)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_relu.numpy(), label='ReLU', color='green')
plt.title('ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Example with a tensor
input_tensor = torch.tensor([-3.0, 0.0, 3.0])
output_tensor = relu_activation(input_tensor)
print(f"ReLU input: {input_tensor}")
print(f"ReLU output: {output_tensor}\n")


# Create a Leaky ReLU activation function
leaky_relu_activation = nn.LeakyReLU(negative_slope=0.01) # alpha = 0.01

# Apply the activation function
y_leaky_relu = leaky_relu_activation(x)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_leaky_relu.numpy(), label='Leaky ReLU ($\\alpha=0.01$)', color='red')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Example with a tensor
input_tensor = torch.tensor([-3.0, 0.0, 3.0])
output_tensor = leaky_relu_activation(input_tensor)
print(f"Leaky ReLU input: {input_tensor}")
print(f"Leaky ReLU output: {output_tensor}\n")


# Create an ELU activation function
elu_activation = nn.ELU(alpha=1.0) # alpha = 1.0 is default

# Apply the activation function
y_elu = elu_activation(x)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_elu.numpy(), label='ELU ($\\alpha=1.0$)', color='purple')
plt.title('ELU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Example with a tensor
input_tensor = torch.tensor([-3.0, 0.0, 3.0])
output_tensor = elu_activation(input_tensor)
print(f"ELU input: {input_tensor}")
print(f"ELU output: {output_tensor}\n")


# Create a GELU activation function
gelu_activation = nn.GELU()

# Apply the activation function
y_gelu = gelu_activation(x)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), y_gelu.numpy(), label='GELU', color='darkblue')
plt.title('GELU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Example with a tensor
input_tensor = torch.tensor([-3.0, 0.0, 3.0])
output_tensor = gelu_activation(input_tensor)
print(f"GELU input: {input_tensor}")
print(f"GELU output: {output_tensor}\n")


# Create a Softmax activation function
softmax_activation = nn.Softmax(dim=0) # dim=0 means apply Softmax across the first dimension (features/classes)

# Example input tensor (logits for 3 classes)
input_tensor_softmax = torch.tensor([1.0, 2.0, 3.0]) # A single sample with 3 logit scores

# Apply the activation function
output_tensor_softmax = softmax_activation(input_tensor_softmax)
print(f"Softmax input (logits): {input_tensor_softmax}")
print(f"Softmax output (probabilities): {output_tensor_softmax}")
print(f"Sum of Softmax output probabilities: {torch.sum(output_tensor_softmax):.4f}\n")

# Example for a batch of samples (common in training)
batch_logits = torch.tensor([[1.0, 2.0, 3.0], # Sample 1
                             [3.0, 1.0, 2.0]]) # Sample 2
softmax_batch = nn.Softmax(dim=1) # dim=1 means apply Softmax across the second dimension (classes for each sample)
output_batch_softmax = softmax_batch(batch_logits)
print(f"Softmax batch input (logits):\n{batch_logits}")
print(f"Softmax batch output (probabilities):\n{output_batch_softmax}")
print(f"Sum of probabilities for sample 1: {torch.sum(output_batch_softmax[0]):.4f}")
print(f"Sum of probabilities for sample 2: {torch.sum(output_batch_softmax[1]):.4f}\n")

