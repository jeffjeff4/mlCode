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

#---------------------------------------------------------
##从图中可以看出：
##
##Sigmoid
##函数的输出范围是(0, 1)。
##
##饱和区域：
##
##当x值非常大（例如x5）时，e−x趋近于0，因此f(x)趋近于1。
##
##当x值非常小（例如 $x \ < -5$）时，e−x趋近于infty，因此f(x)趋近于0。
##
##在这两个区域，函数曲线变得非常平坦，输出几乎不随输入的变化而变化。这被称为饱和。
##
##导数特性：
##
##Sigmoid
##函数的导数f′(x)的最大值是0.25，发生在x = 0处。
##当x远离0时（无论是正向还是负向），f′(x)的值会迅速趋近于0。这意味着在饱和区域，梯度非常小。
##
##梯度消失问题(Vanishing Gradient Problem)
##在深度神经网络中，我们通过反向传播(backpropagation)
##算法来计算损失函数相对于模型参数（权重和偏置）的梯度，然后利用这些梯度来更新参数。
##
##反向传播的链式法则表明，一个参数的梯度是其后续所有层梯度相乘的结果。
##例如，考虑一个简单的三层神经网络：
##L
##rightarrow
##textLayer_3
##rightarrow
##textLayer_2
##rightarrow
##textLayer_1
##rightarrow
##textInput
##
##要计算
##textLayer_1
##中某个参数的梯度，需要乘以
##textLayer_1
##的激活函数导数，再乘以
##textLayer_2
##的激活函数导数，再乘以
##textLayer_3
##的激活函数导数，最后乘以损失相对于
##textLayer_3
##输出的导数。
##
##简化来看，对于一个深度网络中的某个权重W_k(位于第k层)，其梯度fracpartialLpartialW_k
##大致正比于：
##
##$$\frac
##{\partial
##L}{\partial
##W_k} \approx \frac
##{\partial
##L}{\partial \text
##{output}} \times \prod_
##{i = k} ^ {\text
##{last_layer}} f'(z_i) \times \text{some_other_terms}$$
##
##其中f′(z_i)是第i层激活函数的导数。
##
##梯度消失的例子：
##
##假设我们有一个深度网络，所有隐藏层都使用Sigmoid激活函数。
##我们知道Sigmoid的导数f′(x)的最大值是0.25。
##
##如果一个神经元的输入x落在Sigmoid的饱和区域（例如，一个非常大的正数或非常小的负数），那么该层的梯度
##f′(x)将会非常接近0(例如，0.01或0.001)。
##
##现在，想象一下这个非常小的梯度在反向传播过程中，一层一层地与前面层的梯度相乘：
##
##第一层的梯度：
##textgradient_output
##timesf
##′
##(z_textlast)
##timesf
##′
##(z_textlast−1)
##times
##dots
##timesf
##′
##(z_1)
##
##如果每一层的激活函数导数都很小（例如，所有都小于0.25），那么：
##0.25 times 0.25 times 0.25 times dots times0 .25 = (0.25)
##textnum_layers
##
##这个值会随着网络深度的增加呈指数级减小。
##
##举例说明：
##假设：网络有10层隐藏层。每层Sigmoid激活函数的导数平均为0.1(因为很多神经元可能处于饱和区)。
##
##损失函数对最后一层输出的梯度是1。
##
##那么，计算第一层参数的梯度时，近似会乘以(0.1)^10=0.0000000001。
##
##这导致：
##靠近输入层（前几层）的权重梯度变得非常小，接近于零。
##参数几乎不更新： 梯度更新公式是
##W_textnew = W_textold−
##textlearning_rate
##times
##textgradient。如果梯度非常小，即使学习率很大，参数也几乎不会发生变化。
##
##训练停滞： 靠近输入层的神经元（它们负责提取更基础的特征）无法有效地学习和更新，导致整个网络的训练停滞不前，模型无法收敛或收敛速度极其缓慢，性能无法提升。
##
##总结
##Sigmoid函数的饱和特性导致其导数在输入值远离0时趋近于0。在深度网络中，这些小梯度在反向传播过程中层层相乘，导致梯度呈指数级衰减，使得靠近输入层的参数几乎无法更新，从而引发梯度消失问题，严重阻碍了深度网络的训练。
##
##这就是为什么在深度学习中，ReLU及其变体（LeakyReLU, ELU, GELU等）取代Sigmoid和Tanh
##成为主流激活函数的原因。ReLU在正区间导数为常数1，有效解决了梯度消失问题；在负区间导数为
##0（或一个很小的常数），减轻了饱和效应，并引入了稀疏性。


import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid function
def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

x = np.linspace(-10, 10, 200)
y_sigmoid = sigmoid(x)
y_sigmoid_prime = sigmoid_prime(x)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y_sigmoid)
plt.title('Sigmoid Function f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)

plt.subplot(1, 2, 2)
plt.plot(x, y_sigmoid_prime)
plt.title("Derivative of Sigmoid Function f'(x)")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(0.25, color='red', linestyle=':', linewidth=0.8, label='Max derivative = 0.25') # Max at x=0
plt.legend()

plt.tight_layout()
plt.show()
#---------------------------------------------------------

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

