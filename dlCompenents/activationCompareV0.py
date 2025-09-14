#--------------------------------------------------------------------
# 4 questions to compare activation functions:
# 1. bias shift, sigmoid does not have it
# 2. satuation: sigmoid, tanh do not have it
# 3. smooth: only elu, gelu have it
# 4. non-monotonicity: only gelu has it
# 5. nature drop out (drop out probability depends on x pdf and cdf): only gelu has it
#
#
#                       bias shift    |    satuation    |    smooth    |    non-monotonicity    |    nature drop out
#    sigmoid     |       y            |       y         |      n       |        n               |           n
#    tanh        |       n            |       y         |      n       |        n               |           n
#    relu        |       n            |  y (hard 0)     |      n       |        n               |           n
#    leaky relu  |       n            |       n         |      n       |        n               |           n
#    elu         |       n            |       n         |      y       |        n               |           y
#    gelu        |       n            |       n         |      y       |        y               |           y
#
#--------------------------------------------------------------------


import numpy as np

import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# part 0: compare relu and elu
#--------------------------------------------------------------------

#----------------------------------------
# question 0: bias shift
#----------------------------------------

##elu. " helps reduce the "bias shift" issue, pushing mean activations towards zero", what do you mean? please 举例说明
##You're asking for clarification on how ELU (Exponential Linear Unit) helps reduce the "bias shift" issue by pushing mean activations towards zero. Let's break this down with an example.
##What is "Bias Shift"? 🤨
##In neural networks, as data flows through layers, the mean (average) of the activations can drift away from zero. This is known as bias shift.
##Think of it this way:
##Ideal Scenario: For optimal training, we often want the inputs to each layer (and thus the activations coming from the previous layer) to be zero-centered, meaning their average is close to zero. This is similar to how we often normalize input data to have a mean of zero.
##Problem: If the mean activation is significantly positive or negative, it can push subsequent layers into saturated regions of activation functions (like the tails of Sigmoid or Tanh, or the zero region of ReLU for negative inputs). This leads to vanishing gradients, where the gradients become very small, and the network learns very slowly or stops learning altogether.
##ReLU, for example, outputs 0 for all negative inputs. If a neuron's inputs are often negative, or if the initial weights lead to negative sums, a significant portion of its activations will be exactly zero. This tends to make the mean activation positive (or at least non-negative), causing a bias shift away from zero.

import numpy as np


def relu(x):
    return np.maximum(0, x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# Hypothetical pre-activation values for a batch
inputs = np.array([-3, -2, -1, 0, 1, 2, 3])

print(f"Original Inputs: {inputs}")
print(f"Mean of Inputs: {np.mean(inputs):.2f}")
print("-" * 30)

# --- ReLU Activations ---
relu_activations = relu(inputs)
mean_relu_activations = np.mean(relu_activations)

print(f"ReLU Activations: {relu_activations}")
print(f"Mean of ReLU Activations: {mean_relu_activations:.2f}")  # Always non-negative
print("-" * 30)

# --- ELU Activations (alpha=1.0) ---
elu_activations = elu(inputs, alpha=1.0)
mean_elu_activations = np.mean(elu_activations)

print(f"ELU Activations: {elu_activations}")
print(f"Mean of ELU Activations: {mean_elu_activations:.2f}")  # Can be negative, closer to zero
print("-" * 30)

# --- Let's try a scenario with more negative inputs ---
inputs_more_negative = np.array([-5, -4, -3, -2, -1, 0, 1])

relu_activations_neg = relu(inputs_more_negative)
mean_relu_activations_neg = np.mean(relu_activations_neg)

elu_activations_neg = elu(inputs_more_negative, alpha=1.0)
mean_elu_activations_neg = np.mean(elu_activations_neg)

print(f"Original Inputs (more negative): {inputs_more_negative}")
print(f"Mean of Inputs (more negative): {np.mean(inputs_more_negative):.2f}")
print("-" * 30)

print(f"ReLU Activations (more negative): {relu_activations_neg}")
print(f"Mean of ReLU Activations (more negative): {mean_relu_activations_neg:.2f}")
print("-" * 30)

print(f"ELU Activations (more negative): {elu_activations_neg}")
print(f"Mean of ELU Activations (more negative): {mean_elu_activations_neg:.2f}")
print("-" * 30)


##Output Interpretation:
##In the first example (inputs = [-3, -2, -1, 0, 1, 2, 3]):
##Mean of inputs: 0.00
##Mean of ReLU Activations: 1.29 (positive shift)
##Mean of ELU Activations: -0.08 (much closer to zero)
##
##In the second example (inputs_more_negative = [-5, -4, -3, -2, -1, 0, 1]):
##Mean of inputs_more_negative: -2.00
##Mean of ReLU Activations: 0.14 (still a positive shift from the input mean, and generally positive)
##Mean of ELU Activations: -1.63 (maintains a negative mean, closer to the input mean, helping to keep activations centered).
##As you can see, ReLU consistently produces non-negative activations, leading to a positive mean. ELU, by allowing negative outputs, can produce a mean closer to zero or even negative, thereby counteracting the "bias shift" that would otherwise push the mean activations away from zero. This property can lead to faster convergence and better performance in deep neural networks because it helps maintain a more "healthy" flow of information (gradients) through the layers.


#----------------------------------------
# question 1: satuaration
#----------------------------------------
##The Core Idea: What is "Saturation"?
##In the context of activation functions, saturation means that as the input to the function becomes very large (positive or negative), the output of the function changes very little, or even flattens out.
##
##Sigmoid and Tanh: These functions saturate on both ends. For very large positive inputs, they approach 1 (or
##alpha); for very large negative inputs, they approach 0 (or −
##alpha).
##
##ReLU: It saturates for negative inputs, clamping them all to 0. This is a "hard" saturation. For positive inputs, it doesn't saturate; it's linear.
##
##ELU: It doesn't saturate for positive inputs (it's linear, f(x)=x). But for negative inputs, it smoothly saturates to −
##alpha (e.g., -1 if alpha=1). This is a "soft" saturation.
##
##How ELU's Negative Saturation Provides Robustness to Noise 🛡️
##Let's illustrate with an example where activations might contain noise.
##
##Imagine a neuron in a hidden layer receives an input from the previous layer. This input, let's call it x, might not be perfectly clean; it could contain some small random fluctuations or "noise."
##
##Scenario: You have a set of pre-activation values (the output of the weighted sum of inputs to a neuron, before applying the activation function). Let's say these values are slightly negative, and some are affected by noise.
##
##Consider these pre-activation values:
##clean_value = -2.0
##noisy_value_1 = -2.0 + 0.5 = -1.5 (positive noise spike)
##noisy_value_2 = -2.0 - 0.5 = -2.5 (negative noise spike)
##very_noisy_value = -2.0 - 3.0 = -5.0 (larger negative noise spike)

import numpy as np

def relu(x):
    return np.maximum(0, x)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# Define the alpha for ELU
alpha_val = 1.0

print(f"--- Comparing ReLU vs. ELU's response to negative values/noise (alpha={alpha_val}) ---")

# Example values
values = {
    "clean_value": -2.0,
    "noisy_value_1 (less negative)": -1.5,
    "noisy_value_2 (more negative)": -2.5,
    "very_noisy_value (far negative)": -5.0
}

for name, x_val in values.items():
    relu_out = relu(x_val)
    elu_out = elu(x_val, alpha_val)

    print(f"\nInput '{name}': {x_val:.2f}")
    print(f"  ReLU Output: {relu_out:.4f}")
    print(f"  ELU Output:  {elu_out:.4f}")

# Plotting to visualize the saturation effect
x_plot = np.linspace(-5, 0, 100)  # Focus on negative region
y_relu_plot = relu(x_plot)
y_elu_plot = elu(x_plot, alpha_val)

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_relu_plot, label='ReLU', color='blue')
plt.plot(x_plot, y_elu_plot, label=f'ELU ($\\alpha={alpha_val}$)', color='orange', linestyle='--')
plt.axhline(-alpha_val, color='red', linestyle=':', label='ELU Saturation Point ($-\\alpha$)')  # Mark saturation point
plt.title('ReLU vs. ELU in the Negative Region')
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.grid(True)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()


##Interpretation of Outputs and Plot 📈
##Input Value (x),         |   ReLU Output,   |     ELU Output (withalpha=1.0),   |     Observation
##
##clean_value = -2.0       |   0.0000         |      -0.8647                      |     ELU provides a non-zero, negative output.
##noisy_value_1 = -1.5     |   0.0000         |      -0.7769                      |     eLU still outputs 0. ELU's output changed from -0.8647 to -0.7769. The change in ELU's output is relatively small compared to the input change, and it's still negative.
##noisy_value_2 = -2.5     |   0.0000         |      -0.9179                      |     ReLU still outputs 0. ELU's output changed from -0.8647 to -0.9179. Again, a relatively small change, and moving closer to the saturation point.
##very_noisy_value = -5.0  |   0.0000         |      -0.9933                      |     ReLU outputs 0. ELU's output is now very close to its saturation limit of -1.0. Even a large negative "noise" value of -5.0 results in an output that's almost identical to -4.0 or -3.0. The output is stable near −alpha.
##
##匯出到試算表
##What this demonstrates:
##ReLU's "Hard Zero": For any negative input, no matter how small or large, ReLU outputs a flat 0. This means if a neuron's input is, for instance, -0.1 or -100.0, its activation is always 0. This "hard zero" can lead to "dying ReLUs" where neurons become inactive and stop learning. It also means any fine-grained differences or signals in the negative input range are completely lost.
##ELU's "Soft Saturation":
##For small negative inputs (e.g., -0.1, -0.5), ELU still provides distinct negative values and has a non-zero gradient, preventing the "dying ReLU" problem.
##For very large negative inputs (e.g., -3.0, -5.0, -10.0), ELU's output rapidly approaches its saturation value of −alpha (e.g., -1.0). This means that if an input is subject to significant negative noise, pushing it far into the negative region, ELU's output will still be constrained to a narrow range around −alpha. The output of the neuron becomes relatively insensitive to these large, noisy negative fluctuations.
##
##In essence, the saturation in the negative region means that large, noisy negative inputs produce similar, bounded negative outputs. This dampens the impact of extreme negative noise, making the neuron's response more stable and robust to outliers or errors in the input data or internal activations. This robustness helps in smoother learning and potentially faster convergence, especially in deeper networks where noise can accumulate.


#--------------------------------------------------------------------
# part 1: gelu
#--------------------------------------------------------------------

#----------------------------------------
# question 0: Introduces randomness (Gaussian distribution) to activations, which can lead to better performance in large-scale models. It can be thought of as applying a "stochastic regularizer" that gates the input based on its magnitude.
#----------------------------------------
##GELU's Core Idea: "Stochastic Gating"
##The GELU activation function is defined as textGELU(x)=x cdot Phi(x), where Phi(x) is the cumulative distribution function (CDF) of the standard normal distribution.
##The "randomness" isn't explicitly injected in a training step like dropout. Instead, the connection to randomness comes from the motivation behind GELU. The original paper argues that GELU can be seen as a smooth approximation of a stochastic neuron behavior.
##
##Analogy: Adaptive Dropout 🧠
##Imagine a neuron's input x. Instead of simply passing x or setting it to 0 (like ReLU), or randomly dropping it with a fixed probability (like traditional Dropout), imagine a more nuanced "dropping" mechanism:
##Traditional Dropout: For a given neuron, its output x is randomly set to 0 with a fixed probability p (e.g., 0.5), or kept as x with probability 1-p. The decision to drop is independent of the value of x.
##Output (stochastic): x * Bernoulli(1-p)
##Output (expectation): x * (1-p)
##
##GELU's Intuition (Stochastic Regularizer): What if the probability of keeping the input x (i.e., multiplying it by 1) depends on x itself?
##If x is large and positive: It's likely a significant signal, so we want to keep it.
##If x is small (close to zero or negative): It might be noise or less important, so we might want to "drop" it (set it to zero) with a higher probability.
##This is where the Gaussian distribution comes in. The CDF Phi(x) represents the probability that a standard normal random variable is less than or equal to x.
##
##If x is very small (e.g., -3), Phi(x) is close to 0.
##If x is 0, Phi(x) is 0.5.
##If x is very large (e.g., 3), Phi(x) is close to 1.
##
##So, Phi(x) can be interpreted as the probability of keeping the input x (or passing it through) in a conceptual "stochastic gate."
##The "Stochastic Gate" Analogy for GELU:
##Consider a conceptual random variable M sim textBernoulli(Phi(x)). This means M is 1 with probability Phi(x) and 0 with probability 1−Phi(x).
##The output of such a stochastic gate would be x cdot M.
##The expected value of this stochastic operation is:
##E[x⋅M]=x⋅E[M]=x⋅(1⋅Φ(x)+0⋅(1−Φ(x)))=x⋅Φ(x)
##This expected value x cdot Phi(x) is precisely the GELU function!
##
##Therefore, GELU is a deterministic activation function that in expectation behaves like a stochastic regularization process where the probability of "dropping" an activation (setting it to 0) depends on its magnitude.
##Larger positive x: Phi(x) is close to 1, so the probability of "keeping" x is high. The output is close to x.
##x close to 0: Phi(x) is close to 0.5, so there's a 50% chance of "keeping" x and a 50% chance of "dropping" it (conceptually). This means small inputs are attenuated.
##Negative x: Phi(x) is close to 0, so the probability of "keeping" x is low. The output is attenuated towards 0.

import numpy as np
from scipy.stats import norm


def gelu(x):
    """GELU activation function."""
    return x * norm.cdf(x)


# Example input values
inputs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

print("--- GELU as a Magnitude-Dependent Gate ---")
print(f"{'Input (x)':<15} | {'Phi(x) (Prob. of Keeping)':<25} | {'GELU(x) (Output)':<15}")
print("-" * 65)

for x_val in inputs:
    phi_val = norm.cdf(x_val)  # Probability of keeping 'x'
    gelu_val = gelu(x_val)  # Actual GELU output

    print(f"{x_val:<15.2f} | {phi_val:<25.4f} | {gelu_val:<15.4f}")


##Output and Explanation:
##--- GELU as a Magnitude-Dependent Gate ---
##Input (x)       | Phi(x) (Prob. of Keeping) | GELU(x) (Output)
##-----------------------------------------------------------------
##-3.00           | 0.0013                    | -0.0040        <- Very small chance of 'keeping', output heavily attenuated
##-1.00           | 0.1587                    | -0.1587        <- Low chance of 'keeping', output attenuated
##0.00            | 0.5000                    | 0.0000         <- 50% chance of 'keeping', output is 0
##1.00            | 0.8413                    | 0.8413         <- High chance of 'keeping', output close to input
##3.00            | 0.9987                    | 2.9960         <- Very high chance of 'keeping', output almost exactly input
##
##As the example shows:
##For small or negative inputs (-3.0, -1.0, 0.0),
##Phi(x) is small or 0.5. This means the corresponding GELU output is heavily attenuated or near zero. It's like these less significant signals are "dropped" or down-weighted more aggressively.
##
##For large positive inputs (1.0, 3.0),
##Phi(x) is close to 1. This means the input x is largely "kept" and passed through, as it's likely a strong, important signal.
##
##Why this is a "Stochastic Regularizer" and leads to better performance:
##Adaptive Gating/Regularization: Unlike traditional dropout which drops neurons randomly with a fixed probability, GELU's gating is adaptive to the input's magnitude. Larger inputs are more likely to be preserved, while smaller or negative inputs are more likely to be attenuated. This makes the regularization more intelligent.
##
##Smoothness: The Phi(x) function is smooth and differentiable everywhere. This means the "gating" is not a harsh, step-like function (like ReLU's sudden zero-out for negatives). Smoothness helps with gradient flow during backpropagation, leading to more stable and efficient training, especially in very deep models.
##
##Combination of Benefits: GELU effectively combines the non-linearity of activation functions with the regularization effect of dropout. It reduces the reliance of the network on specific input values by implicitly "fuzzing" or attenuating less significant activations, which helps prevent overfitting and improves generalization.
##
##Empirical Success: This underlying principle is believed to be a key reason why GELU has been so successful in large-scale models, particularly Transformer architectures (like BERT, GPT, etc.), where such nuanced handling of activation magnitudes and implicit regularization prove highly beneficial for learning complex patterns.

#----------------------------------------
# question 1: Highly effective in state-of-the-art models like Transformers and large language models.
#----------------------------------------
##gelu, "Highly effective in state-of-the-art models like Transformers and large language models.", 请举例说明
##
##GELU is highly effective in state-of-the-art models like Transformers and large language models (LLMs) primarily because of its smoothness, non-monotonicity, and probabilistic interpretation. These characteristics contribute to better gradient flow, more robust learning, and improved ability to capture complex patterns in high-dimensional data, which are crucial for the immense scale and intricacy of Transformer architectures.
##
##Why GELU Excels in State-of-the-Art Models (e.g., Transformers) 🚀
##Let's illustrate with the most prominent examples:
##BERT (Bidirectional Encoder Representations from Transformers):
##Context: BERT revolutionized Natural Language Processing (NLP) by learning deep bidirectional representations from unlabeled text. It's built on the Transformer encoder architecture.
##GELU's Role: Within each Transformer block of BERT, specifically in the feed-forward networks (FFNs), GELU is used as the activation function between the two linear layers.
##Benefit: BERT operates on rich, contextual word embeddings that often contain complex and nuanced information. GELU's smoothness (unlike ReLU's sharp cutoff at zero) ensures that gradients flow more consistently and smoothly through these deep networks, even for negative pre-activation values. This helps prevent "dead neurons" (a problem where ReLU neurons get stuck outputting zero) and allows the model to learn more effectively from small or negative activations. Its non-monotonicity provides additional flexibility to model complex relationships in the high-dimensional latent spaces common in NLP.
##
##GPT (Generative Pre-trained Transformer) Series (e.g., GPT-2, GPT-3, GPT-4):
##Context: These are large language models primarily used for text generation, summarization, and understanding, also based on the Transformer architecture (specifically the decoder part).
##GELU's Role: Similar to BERT, GELU is a standard component in the FFNs within the Transformer blocks of GPT models.
##Benefit: For LLMs with billions of parameters and vast training datasets, stable and efficient training is paramount. GELU's properties, particularly its ability to maintain a mean activation closer to zero (mitigating "bias shift"), help stabilize training dynamics in these incredibly deep and wide networks. This stability is critical for the convergence of such large models and for preventing gradient issues that could derail learning over many layers and billions of computations. The "stochastic regularizer" intuition also suggests it contributes to better generalization by adaptively weighting features based on their significance.
##
##Vision Transformers (ViT):
##Context: Transformers, initially designed for NLP, have also found great success in computer vision tasks like image classification. Vision Transformers (ViT) process images as sequences of patches.
##GELU's Role: Again, GELU is commonly used in the FFNs within the Transformer encoder blocks of ViTs.
##Benefit: Just as with textual data, image patches can lead to diverse activation patterns. GELU's smooth and probabilistic approach to activation helps the model learn robust features from visual data, which can often contain subtle patterns or noise. The gentle "gating" based on magnitude (where important features are fully passed and less important ones are attenuated) makes ViTs more effective at capturing complex spatial relationships without over-emphasizing noise.
##
##Why GELU is Preferred Over Alternatives in these Models ✨
##Compared to ReLU: While ReLU is computationally simpler, its hard zero for negative inputs can lead to dead neurons and non-zero-centered outputs, which can slow down training for very deep models. GELU's smoothness and negative outputs address these issues, promoting more stable gradients.
##Compared to Sigmoid/Tanh: Sigmoid and Tanh suffer from severe vanishing gradients in deep networks because they saturate on both ends. GELU (like ReLU) avoids this for positive inputs and has a soft saturation for negative values, providing a better balance between non-linearity and gradient flow.
##Computational Cost: Although slightly more computationally intensive than ReLU (due to involving the erf or tanh approximation), the benefits in training stability and final model performance in large-scale, complex architectures often outweigh this minor overhead, especially with highly optimized GPU implementations.
##
##In essence, GELU's nuanced non-linearity and implicitly regularizing behavior provide a more robust and efficient activation mechanism, allowing state-of-the-art models like Transformers and LLMs to learn from massive datasets and achieve their impressive performance.

#----------------------------------------
# question 2: gelu, "Its non-monotonicity provides additional flexibility to model complex relationships in the high-dimensional latent spaces common in NLP."
#----------------------------------------
##What is Non-Monotonicity? 📈📉
##A function is monotonic if it either always increases or always decreases.
##
##Monotonically increasing: As the input increases, the output never decreases.
##
##Monotonically decreasing: As the input increases, the output never increases.
##
##Non-monotonicity means the function's output can both increase and decrease as its input increases. It has "dips" or "peaks" that are not simply constant plateaus.
##
##Let's look at the plots we generated earlier (or imagine them):
##
##ReLU and Leaky ReLU are monotonic (always increasing or flat).
##
##ELU is also monotonic (always increasing, though it saturates to a constant for large negative values).
##
##GELU, however, is non-monotonic. While it generally increases, it has a slight "dip" for negative inputs before increasing again.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # For accurate CDF


def gelu_accurate(x):
    """GELU activation function using accurate CDF."""
    return x * norm.cdf(x)


# Generate input values, focusing on the region where non-monotonicity is visible
x_values = np.linspace(-4, 2, 200)  # Extend to visualize the dip
y_gelu = gelu_accurate(x_values)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_gelu, label='GELU', color='blue', linewidth=2)
plt.title('GELU Activation Function (Highlighting Non-Monotonicity)')
plt.xlabel('Input (x)')
plt.ylabel('GELU(x)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='gray', linestyle='-', linewidth=0.7)
plt.axvline(0, color='gray', linestyle='-', linewidth=0.7)
plt.legend()
plt.show()

##Why Non-Monotonicity Helps in High-Dimensional Latent Spaces (NLP) 🤖
##High-dimensional latent spaces in NLP (like the embeddings learned by Transformers for words, phrases, or contextual representations) are incredibly complex. They capture nuanced semantic, syntactic, and contextual relationships.
##
##1. Increased Modeling Capacity:
##Monotonic functions can only create boundaries that are somewhat simpler. For example, a linear combination of monotonic functions might struggle to approximate functions that have complex "curves" or "swings."
##Non-monotonic functions, by allowing both increases and decreases in output over their domain, provide the neural network with greater expressive power and flexibility. They can learn to model more intricate and non-linear relationships that monotonic functions might miss or approximate poorly. This means the model can capture more subtle interactions between features in the latent space.
##
##2. Handling Complex Feature Interactions:
##In NLP, a word's meaning is highly context-dependent. A slight change in an embedding dimension might correspond to a shift in sentiment, part of speech, or semantic role. These relationships are rarely simple monotonic mappings.
##GELU's non-monotonicity allows it to differentiate between inputs that are close to zero in a more sophisticated way than ReLU. It can effectively "gate" or down-weight features that are slightly negative or small positive differently from those that are strongly positive. This is crucial when dealing with subtle signals or noise in high-dimensional representations. For example, a specific combination of features that falls into GELU's "dip" region might be indicative of a nuanced linguistic phenomenon that a monotonic function would simply treat as "slightly positive" or "zero."
##
##3. Better Gradient Flow for Nuanced Signals:
##The smooth, non-monotonic curve means that gradients are always flowing (even for negative inputs, unlike ReLU's zero gradient), but their magnitude can vary in a more complex way. This helps the model learn from all parts of the input distribution, including those subtle, noisy, or ambiguous signals that might exist in dense embeddings. This improved gradient flow contributes to better optimization in very deep Transformer models.
##
##In essence, the non-monotonicity of GELU allows the model to learn more sophisticated transformations of its high-dimensional inputs, enabling it to better capture the intricate, non-linear dependencies present in language data, which is critical for state-of-the-art NLP performance.
