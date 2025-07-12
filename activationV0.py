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

# Generate some input values
x = torch.linspace(-5, 5, 100)

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

#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Tanh function
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Derivative of Tanh function
def tanh_prime(x):
    return 1 - tanh(x)**2

x = np.linspace(-10, 10, 200)
y_tanh = tanh(x)
y_tanh_prime = tanh_prime(x)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y_tanh)
plt.title('Tanh Function f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(-1, color='gray', linestyle='--', linewidth=0.8)

plt.subplot(1, 2, 2)
plt.plot(x, y_tanh_prime)
plt.title("Derivative of Tanh Function f'(x)")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axhline(1, color='red', linestyle=':', linewidth=0.8, label='Max derivative = 1') # Max at x=0
plt.legend()

plt.tight_layout()
plt.show()

##From the plots, we can observe:
##
##Tanh function's output range is (-1, 1). This is a key advantage over Sigmoid, as its output is zero-centered, which can sometimes lead to faster convergence during training.
##
##Saturation Regions:
##When x is a very large positive number (e.g., x3), e^−x  becomes very small, and ex dominates. So, frac (e^x − e^−x) / (e^x + e^−x) approaches
##frac e^x / e^x=1. The function output saturates at 1.
##
##When x is a very large negative number (e.g., $x \< -3$), e^x becomes very small, and e^−x dominates. So, frac (e^x − e^−x) / (e^x + e^−x) approaches
##frac (−e^−x) / (e^−x) =−1. The function output saturates at -1.
##
##In these saturated regions (where the curve becomes flat), the output changes very little even with significant changes in input.
##
##Derivative Characteristics:
##The derivative f′(x)=1−tanh^2(x) has a maximum value of 1, which occurs at x=0 (since tanh(0)=0).
##
##As x moves away from 0 (either positively or negatively), the value of
##tanh(x) approaches 1 or -1. Consequently, tanh^2(x) approaches 1, and thus f′(x) (which is 1−tanh^2 (x)) approaches 1−1=0.
##
##Why Tanh Suffers from Vanishing Gradients
##The mechanism for vanishing gradients with Tanh is fundamentally the same as with Sigmoid:
##
##Small Derivatives in Saturated Regions: If the pre-activation value (z=
##sumw_ix_i+b) for a neuron falls into the saturated regions of the Tanh function (i.e., z is a large positive or large negative number), the derivative of the Tanh function f′(z) will be very close to zero.
##
##Multiplication of Small Gradients in Backpropagation: In a deep neural network, when backpropagating the error, the gradient for weights in earlier layers is computed by multiplying the gradients of all subsequent layers' activation functions.
##
##For example, consider the gradient flow through multiple layers:
##fracpartialLpartialW_1
##propto
##fracpartialLpartialtextoutput
##timesf
##′
## (z_textLayer_N)
##timesf
##′
## (z_textLayer_N−1)
##times
##dots
##timesf
##′
## (z_textLayer_1).
##
##Even though Tanh's maximum derivative is 1 (compared to Sigmoid's 0.25), if inputs to a few consecutive layers fall into the saturated regions, their corresponding derivatives will be very small (e.g., 0.05, 0.01, etc.).
##
##Multiplying these small numbers across many layers leads to an exponentially decreasing product. For instance, if you have 10 layers, and each contributes a derivative of, say, 0.1 (due to saturation), the overall multiplier for the first layer's gradient would be (0.1)^10 =0.0000000001.
##
##Near-Zero Weight Updates: When the gradients become extremely small, the updates to the weights and biases in the earlier layers of the network become negligible.
##
##W_textnew=W_textold−
##textlearning_rate
##times
##textgradient
##
##If gradient is close to zero, W_textnew will be almost identical to W_textold.
##
##Stalled Learning: This means that the early layers of the deep network, which are responsible for learning fundamental, low-level features from the input data, effectively stop learning. The network cannot properly extract useful representations from the raw input, and the overall training process stalls or becomes extremely slow, leading to poor model performance.
##
##Tanh vs. Sigmoid: A Small Advantage, But Still Prone to Vanishing Gradients
##While Tanh is generally preferred over Sigmoid for hidden layers because its zero-centered output can help with faster convergence (by avoiding a "zig-zagging" effect in gradient descent when all gradients are positive), it does not fundamentally solve the vanishing gradient problem. Both functions squash their inputs into a bounded range, and in those saturation regions, their derivatives become negligible.
##
##This limitation is precisely why Rectified Linear Units (ReLUs) and their variants (Leaky ReLU, ELU, GELU) became so popular in deep learning. ReLUs have a constant derivative (1) for positive inputs, which largely mitigates the vanishing gradient problem in that range.

#-------------------------------------------------------
# Generate some input values
x = torch.linspace(-5, 5, 100)

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
#-------------------------------------------------------



#-------------------------------------------------------
# Generate some input values
x = torch.linspace(-5, 5, 100)

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

#-------------------------------------------------------




#-------------------------------------------------------
# Generate some input values
x = torch.linspace(-5, 5, 100)

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

#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def elu(x, alpha=1.0):
    """
    Exponential Linear Unit (ELU) activation function.

    Args:
        x (numpy.ndarray or float): Input value(s).
        alpha (float): Hyperparameter, typically 1.0.

    Returns:
        numpy.ndarray or float: Output of the ELU function.
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# Generate input values
x = np.linspace(-5, 5, 100)

# Calculate ELU output
y_elu = elu(x)

# Plotting the ELU function
plt.figure(figsize=(7, 4))
plt.plot(x, y_elu, label='ELU ($\\alpha=1.0$)', color='purple')
plt.title('ELU Activation Function')
plt.xlabel('x')
plt.ylabel('ELU(x)')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()

#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def elu_derivative(x, alpha=1.0):
    """
    Derivative of the Exponential Linear Unit (ELU) activation function.

    Args:
        x (numpy.ndarray or float): Input value(s).
        alpha (float): Hyperparameter, typically 1.0.

    Returns:
        numpy.ndarray or float: Output of the ELU derivative.
    """
    return np.where(x > 0, 1, alpha * np.exp(x))


# Generate input values for derivative
x_deriv = np.linspace(-5, 5, 100)

# Calculate ELU derivative output
y_elu_deriv = elu_derivative(x_deriv)

# Plotting the ELU derivative function
plt.figure(figsize=(7, 4))
plt.plot(x_deriv, y_elu_deriv, label='ELU Derivative ($\\alpha=1.0$)', color='orange')
plt.title('ELU Activation Function Derivative')
plt.xlabel('x')
plt.ylabel("ELU'(x)")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()

#-------------------------------------------------------
# Generate some input values
x = torch.linspace(-5, 5, 100)

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


#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm # For accurate CDF for comparison
from scipy.special import erf # For accurate error function

def gelu_accurate(x):
    """
    GELU activation function using the accurate CDF of the standard normal distribution.
    Requires scipy.stats.norm.cdf.
    """
    return x * norm.cdf(x)

def gelu_approximate(x):
    """
    GELU activation function using the tanh approximation.
    This is the commonly used approximation in deep learning frameworks due to computational efficiency.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Generate input values
x = np.linspace(-5, 5, 100)

# Calculate GELU using both methods
y_gelu_accurate = gelu_accurate(x)
y_gelu_approximate = gelu_approximate(x)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, y_gelu_accurate, label='GELU (Accurate CDF)', color='blue')
plt.plot(x, y_gelu_approximate, label='GELU (Tanh Approximation)', color='red', linestyle='--')
plt.title('GELU Activation Function: Accurate vs. Approximation')
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()

#-------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # For accurate CDF and PDF


def gelu_derivative_accurate(x):
    """
    Derivative of GELU activation function using accurate CDF and PDF.
    """
    return norm.cdf(x) + x * norm.pdf(x)


def gelu_approximate(x):
    """
    GELU activation function using the tanh approximation.
    (Repeated for context in derivative calculation)
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative_approximate(x):
    """
    Derivative of GELU activation function using the tanh approximation.
    This involves the derivative of tanh(A(x)) where A(x) is the argument.
    """
    sqrt_2_pi = np.sqrt(2 / np.pi)
    const = 0.044715

    # A(x) argument for tanh
    arg_tanh = sqrt_2_pi * (x + const * x ** 3)

    # Derivative of A(x)
    d_arg_tanh = sqrt_2_pi * (1 + 3 * const * x ** 2)

    # tanh(A(x))
    tanh_val = np.tanh(arg_tanh)

    # sech^2(A(x)) = 1 - tanh^2(A(x))
    sech_squared = 1 - tanh_val ** 2

    # Derivative of gelu_approximate(x)
    # Using product rule: 0.5 * [ (1 * (1 + tanh_val)) + (x * sech_squared * d_arg_tanh) ]
    derivative = 0.5 * (1 + tanh_val) + 0.5 * x * sech_squared * d_arg_tanh
    return derivative


# Generate input values
x_deriv = np.linspace(-5, 5, 100)

# Calculate derivatives
y_gelu_deriv_accurate = gelu_derivative_accurate(x_deriv)
y_gelu_deriv_approximate = gelu_derivative_approximate(x_deriv)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_deriv, y_gelu_deriv_accurate, label='GELU Derivative (Accurate)', color='blue')
plt.plot(x_deriv, y_gelu_deriv_approximate, label='GELU Derivative (Approximate)', color='red', linestyle='--')
plt.title('GELU Activation Function Derivative: Accurate vs. Approximation')
plt.xlabel('x')
plt.ylabel("GELU'(x)")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()

#-------------------------------------------------------
# Generate some input values
x = torch.linspace(-10000, 10000, 10000)

# Create a Softmax activation function
softmax_activation = nn.Softmax(dim=0) # dim=0 means apply Softmax across the first dimension (features/classes)

# Example input tensor (logits for 3 classes)
input_tensor_softmax = torch.tensor(x) # A single sample with 3 logit scores

#-------------------------------------
# Apply the activation function
output_tensor_softmax = softmax_activation(input_tensor_softmax)

y = output_tensor_softmax.numpy()
# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Softmax', color='blue')
plt.xlabel('x')
plt.ylabel("Softmax'(x)")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()

#-------------------------------
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


#-------------------------------------------------------

import numpy as np

def softmax(z):
    """
    Softmax activation function.

    Args:
        z (numpy.ndarray): A 1D NumPy array of raw scores (logits).

    Returns:
        numpy.ndarray: A 1D NumPy array representing a probability distribution.
    """
    # Subtract the maximum value from z for numerical stability
    # This does not change the output of softmax, but prevents overflow for large z values
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Example usage
scores = np.array([1.0, 2.0, 3.0])
probabilities = softmax(scores)
print(f"Input Scores (z): {scores}")
print(f"Softmax Probabilities: {probabilities}")
print(f"Sum of Probabilities: {np.sum(probabilities):.4f}")

scores_large = np.array([100.0, 101.0, 102.0])
probabilities_large = softmax(scores_large)
print(f"\nInput Scores (large values): {scores_large}")
print(f"Softmax Probabilities (with stability trick): {probabilities_large}")
print(f"Sum of Probabilities: {np.sum(probabilities_large):.4f}")

scores_negative = np.array([-1.0, -2.0, -3.0])
probabilities_negative = softmax(scores_negative)
print(f"\nInput Scores (negative values): {scores_negative}")
print(f"Softmax Probabilities: {probabilities_negative}")
print(f"Sum of Probabilities: {np.sum(probabilities_negative):.4f}")

#-------------------------------------------------------

import numpy as np


def softmax(z):
    """
    Softmax activation function (re-defined for completeness).
    """
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def softmax_derivative(z):
    """
    Calculates the Jacobian matrix of the Softmax function.

    Args:
        z (numpy.ndarray): A 1D NumPy array of raw scores (logits).

    Returns:
        numpy.ndarray: A 2D NumPy array (Jacobian matrix) where
                       softmax_jacobian[i, k] = d(Softmax(z_i)) / d(z_k).
    """
    s = softmax(z)
    K = len(z)
    jacobian_matrix = np.zeros((K, K))

    for i in range(K):
        for k in range(K):
            if i == k:
                jacobian_matrix[i, k] = s[i] * (1 - s[i])
            else:
                jacobian_matrix[i, k] = -s[i] * s[k]

    return jacobian_matrix


# Example usage
scores_deriv = np.array([1.0, 2.0, 3.0])
softmax_output = softmax(scores_deriv)
jacobian = softmax_derivative(scores_deriv)

print(f"\nSoftmax Input (z): {scores_deriv}")
print(f"Softmax Output (S): {softmax_output}")
print(f"\nSoftmax Jacobian Matrix:")
print(jacobian)


# Verify one element: d(S_0)/d(z_0) = S_0 * (1 - S_0)
print(f"S_0 * (1 - S_0): {softmax_output[0] * (1 - softmax_output[0])}")
print(f"Jacobian[0,0]: {jacobian[0,0]}")

# Verify another element: d(S_0)/d(z_1) = -S_0 * S_1
print(f"-S_0 * S_1: {-softmax_output[0] * softmax_output[1]}")
print(f"Jacobian[0,1]: {jacobian[0,1]}")

#-------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Softmax activation function.
    """
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)  # Ensure sum over last axis


def softmax_derivative_element(s_i, s_k, i_equals_k):
    """
    Calculates a single element of the Softmax derivative.
    """
    if i_equals_k:
        return s_i * (1 - s_i)
    else:
        return -s_i * s_k


# --- Setup for plotting ---
num_classes = 3
# Vary the first input (z_0) over a range
z0_values = np.linspace(-5, 5, 100)
# Keep other inputs constant for simplicity
z1_fixed = 0.0
z2_fixed = 0.0

# Store derivative values for each output S_i with respect to z_0
deriv_s0_wrt_z0 = []
deriv_s1_wrt_z0 = []
deriv_s2_wrt_z0 = []

# Loop through z0_values to calculate derivatives
for z0_val in z0_values:
    # Create the input vector for softmax at current z0_val
    z_vector = np.array([z0_val, z1_fixed, z2_fixed])

    # Calculate softmax output for this z_vector
    s_output = softmax(z_vector)  # This will be [S0, S1, S2]

    # Calculate derivatives with respect to z_0
    deriv_s0_wrt_z0.append(softmax_derivative_element(s_output[0], s_output[0], True))  # dS0/dz0
    deriv_s1_wrt_z0.append(softmax_derivative_element(s_output[1], s_output[0], False))  # dS1/dz0
    deriv_s2_wrt_z0.append(softmax_derivative_element(s_output[2], s_output[0], False))  # dS2/dz0

# Convert lists to numpy arrays for plotting
deriv_s0_wrt_z0 = np.array(deriv_s0_wrt_z0)
deriv_s1_wrt_z0 = np.array(deriv_s1_wrt_z0)
deriv_s2_wrt_z0 = np.array(deriv_s2_wrt_z0)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(z0_values, deriv_s0_wrt_z0, label='$\\frac{\\partial S_0}{\\partial z_0} = S_0(1-S_0)$', color='blue')
plt.plot(z0_values, deriv_s1_wrt_z0, label='$\\frac{\\partial S_1}{\\partial z_0} = -S_1 S_0$', color='red')
plt.plot(z0_values, deriv_s2_wrt_z0, label='$\\frac{\\partial S_2}{\\partial z_0} = -S_2 S_0$', color='green')

plt.title('Softmax Derivative with respect to $z_0$ (while $z_1=0, z_2=0$)')
plt.xlabel('Input $z_0$')
plt.ylabel('Derivative Value')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.ylim(-0.25, 0.25)  # Adjust y-limit for better visualization of the derivatives
plt.show()