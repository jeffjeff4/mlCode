####python code, forward/backward passes for a simple mlp
####
####This document outlines the core formulas for a simple 2-layer Multi-Layer Perceptron (MLP) with one hidden layer.
####
####1. Notation
####
####$X$: Input matrix, shape (N, d_in) where N is batch size, d_in is input features.
####
####$y$: True labels, shape (N, d_out).
####
####$W_1$, $b_1$: Weights and biases for the hidden layer.
####
####$W_1$ shape: (d_in, d_h)
####
####$b_1$ shape: (1, d_h)
####
####$W_2$, $b_2$: Weights and biases for the output layer.
####
####$W_2$ shape: (d_h, d_out)
####
####$b_2$ shape: (1, d_out)
####
####$z_1$, $a_1$: Pre-activation and post-activation of the hidden layer.
####
####$z_1$ shape: (N, d_h)
####
####$a_1$ shape: (N, d_h)
####
####$z_2$, $\hat{y}$: Pre-activation and post-activation (prediction) of the output layer.
####
####$z_2$ shape: (N, d_out)
####
####$\hat{y}$ shape: (N, d_out)
####
####$\sigma(z)$: Activation function (e.g., Sigmoid, ReLU).
####
####$\sigma'(z)$: Derivative of the activation function.
####
####$L$: Loss function (e.g., Mean Squared Error, MSE).
####
####2. Forward Pass (FP)
####
####The forward pass computes the prediction $\hat{y}$ from the input $X$.
####
####Hidden Layer (Layer 1):
####
####$z_1 = X \cdot W_1 + b_1$
####
####$a_1 = \sigma(z_1)$
####
####Output Layer (Layer 2):
####
####$z_2 = a_1 \cdot W_2 + b_2$
####
####$\hat{y} = \sigma(z_2)$
####(Note: The output activation might be different, e.g., Softmax for classification, or linear for regression. We'll assume $\sigma$ for simplicity.)
####
####Loss Calculation:
####
####$L = \text{MSE}(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$
####
####3. Backward Pass (BP) / Backpropagation
####
####The backward pass computes the gradient (derivative) of the Loss $L$ with respect to each parameter. This is an application of the chain rule, starting from the end and moving backward.
####
####Notation: $\frac{\partial L}{\partial W_2}$ is the gradient of the Loss with respect to $W_2$. We'll call this grad_W2.
####
####Start:
####
####Gradient of Loss w.r.t. Prediction ($\hat{y}$):
####
####$\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y} - y)$
####
####(This is the derivative of the MSE loss)
####
####Step 1: Gradients at Output Layer (Layer 2)
####
####Gradient w.r.t. $z_2$:
####
####$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2}$
####
####$\frac{\partial \hat{y}}{\partial z_2} = \sigma'(z_2)$
####
####$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \odot \sigma'(z_2)$
####(Note: $\odot$ is element-wise multiplication)
####
####Gradient w.r.t. $W_2$:
####
####$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial W_2}$
####
####$\frac{\partial z_2}{\partial W_2} = a_1^T$
####
####$\frac{\partial L}{\partial W_2} = a_1^T \cdot \frac{\partial L}{\partial z_2}$
####(Note: This is a matrix multiplication)
####
####Gradient w.r.t. $b_2$:
####
####$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial b_2}$
####
####$\frac{\partial z_2}{\partial b_2} = 1$
####
####$\frac{\partial L}{\partial b_2} = \sum_{\text{batch}} \left( \frac{\partial L}{\partial z_2} \right)$
####(Note: We sum the gradients across the batch)
####
####Step 2: Propagate Gradients to Hidden Layer (Layer 1)
####
####Gradient w.r.t. $a_1$:
####
####$\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1}$
####
####$\frac{\partial z_2}{\partial a_1} = W_2^T$
####
####$\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} \cdot W_2^T$
####(Note: This is a matrix multiplication that "propagates" the error backward)
####
####Gradient w.r.t. $z_1$:
####
####$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1}$
####
####$\frac{\partial a_1}{\partial z_1} = \sigma'(z_1)$
####
####$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1)$
####(Note: $\odot$ is element-wise multiplication)
####
####Gradient w.r.t. $W_1$:
####
####$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$
####
####$\frac{\partial z_1}{\partial W_1} = X^T$
####
####$\frac{\partial L}{\partial W_1} = X^T \cdot \frac{\partial L}{\partial z_1}$
####(Note: This is a matrix multiplication)
####
####Gradient w.r.t. $b_1$:
####
####$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \cdot \frac{\partial z_1}{\partial b_1}$
####
####$\frac{\partial z_1}{\partial b_1} = 1$
####
####$\frac{\partial L}{\partial b_1} = \sum_{\text{batch}} \left( \frac{\partial L}{\partial z_1} \right)$
####(Note: We sum the gradients across the batch)
####
####4. Summary of Gradient Formulas (for Gradient Descent)
####
####To update the weights, you would calculate these gradients and then apply the rule:
####W = W - learning_rate * grad_W
####
####$\delta_2 = \frac{\partial L}{\partial \hat{y}} \odot \sigma'(z_2)$
####
####$\frac{\partial L}{\partial W_2} = a_1^T \cdot \delta_2$
####
####$\frac{\partial L}{\partial b_2} = \sum(\delta_2)$
####
####(Where $\delta_2$ and $\delta_1$ are the "error signals" at the output and hidden layers, respectively.)




import numpy as np


# --- 1. Activation Function and its Derivative ---
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    # Note: 'x' here is assumed to be the *output* of sigmoid
    # (i.e., sig(z)), so we calculate sig(z) * (1 - sig(z))
    return x * (1 - x)


# --- 2. Loss Function and its Derivative ---
def mse_loss(y_pred, y_true):
    """Mean Squared Error loss."""
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_derivative(y_pred, y_true):
    """Derivative of the MSE loss w.r.t. y_pred."""
    # dLoss/dy_pred = 2 * (y_pred - y_true) / N
    # We'll use the 2/N part in the gradient calculation
    return 2 * (y_pred - y_true) / y_true.shape[0]


# --- 3. The MLP Class ---
class SimpleMLP:
    """A simple 2-layer MLP (Input -> Hidden -> Output)"""

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        # W1: [input_size, hidden_size]
        # b1: [1, hidden_size]
        # W2: [hidden_size, output_size]
        # b2: [1, output_size]
        np.random.seed(42)  # for reproducibility
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Cache for storing intermediate values for backprop
        self.cache = {}
        # Gradients
        self.grads = {}

    def forward(self, X):
        """
        Performs the forward pass and caches intermediate values.
        X -> z1 -> a1 -> z2 -> y_pred
        """
        # Layer 1 (Hidden)
        # z1 = X @ W1 + b1
        z1 = np.dot(X, self.W1) + self.b1
        # a1 = sigmoid(z1)
        a1 = sigmoid(z1)

        # Layer 2 (Output)
        # z2 = a1 @ W2 + b2
        z2 = np.dot(a1, self.W2) + self.b2
        # y_pred = sigmoid(z2)  (Using sigmoid for 0-1 output)
        y_pred = sigmoid(z2)

        # Store all intermediate values in cache for backward pass
        self.cache['X'] = X  # (N, input_size)
        self.cache['z1'] = z1  # (N, hidden_size)
        self.cache['a1'] = a1  # (N, hidden_size)
        self.cache['z2'] = z2  # (N, output_size)
        self.cache['y_pred'] = y_pred  # (N, output_size)

        return y_pred

    def backward(self, y_true):
        """
        Performs the backward pass (backpropagation) to compute gradients.
        This is the core of the chain rule.
        """

        # --- Retrieve values from cache ---
        X = self.cache['X']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        y_pred = self.cache['y_pred']

        # N = batch size
        N = X.shape[0]

        # --- Start from the end: Gradient of Loss w.r.t. y_pred ---
        # dLoss/dy_pred
        dLoss_dy_pred = mse_loss_derivative(y_pred, y_true)  # (N, output_size)

        # --- Step 1: Gradients at Output Layer ---
        # Propagate to z2: dLoss/dz2
        # dLoss/dz2 = dLoss/dy_pred * dy_pred/dz2
        # dy_pred/dz2 is the derivative of sigmoid(z2)
        dy_pred_dz2 = sigmoid_derivative(y_pred)  # (N, output_size)
        dLoss_dz2 = dLoss_dy_pred * dy_pred_dz2  # (N, output_size)

        # Gradients for W2 and b2
        # dLoss/dW2 = dLoss/dz2 * dz2/dW2 = a1.T @ dLoss_dz2
        dLoss_dW2 = np.dot(a1.T, dLoss_dz2)  # (hidden_size, output_size)

        # dLoss/db2 = dLoss/dz2 * dz2/db2 = sum(dLoss_dz2)
        dLoss_db2 = np.sum(dLoss_dz2, axis=0, keepdims=True)  # (1, output_size)

        # --- Step 2: Gradients at Hidden Layer ---
        # Propagate to a1: dLoss/da1
        # dLoss/da1 = dLoss/dz2 * dz2/da1 = dLoss_dz2 @ W2.T
        dLoss_da1 = np.dot(dLoss_dz2, self.W2.T)  # (N, hidden_size)

        # Propagate to z1: dLoss/dz1
        # dLoss/dz1 = dLoss/da1 * da1/dz1
        # da1/dz1 is the derivative of sigmoid(z1)
        da1_dz1 = sigmoid_derivative(a1)  # (N, hidden_size)
        dLoss_dz1 = dLoss_da1 * da1_dz1  # (N, hidden_size)

        # Gradients for W1 and b1
        # dLoss/dW1 = dLoss/dz1 * dz1/dW1 = X.T @ dLoss_dz1
        dLoss_dW1 = np.dot(X.T, dLoss_dz1)  # (input_size, hidden_size)

        # dLoss/db1 = dLoss/dz1 * dz1/db1 = sum(dLoss_dz1)
        dLoss_db1 = np.sum(dLoss_dz1, axis=0, keepdims=True)  # (1, hidden_size)

        # --- Store gradients ---
        self.grads['dW1'] = dLoss_dW1
        self.grads['db1'] = dLoss_db1
        self.grads['dW2'] = dLoss_dW2
        self.grads['db2'] = dLoss_db2

    def update_weights(self, learning_rate):
        """
        Updates the model's weights and biases using gradient descent.
        """
        self.W1 -= learning_rate * self.grads['dW1']
        self.b1 -= learning_rate * self.grads['db1']
        self.W2 -= learning_rate * self.grads['dW2']
        self.b2 -= learning_rate * self.grads['db2']


# --- 4. Training Loop ---
if __name__ == "__main__":

    # XOR problem: a classic non-linear problem
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y_true = np.array([[0],
                       [1],
                       [1],
                       [0]])

    # --- Hyperparameters ---
    input_size = 2
    hidden_size = 4  # Number of neurons in the hidden layer
    output_size = 1
    learning_rate = 0.1
    epochs = 20000

    # --- Initialize Model ---
    mlp = SimpleMLP(input_size, hidden_size, output_size)

    print("Starting training...")

    # --- Training ---
    for epoch in range(epochs):
        # 1. Forward Pass
        # Calculate predictions
        y_pred = mlp.forward(X)

        # 2. Calculate Loss
        loss = mse_loss(y_pred, y_true)

        # 3. Backward Pass (Backpropagation)
        # Compute gradients
        mlp.backward(y_true)

        # 4. Update Weights
        # Apply gradient descent
        mlp.update_weights(learning_rate)

        # Print loss
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    print("Training complete.")

    # --- Final Predictions ---
    print("\nFinal predictions:")
    final_predictions = mlp.forward(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Output: {final_predictions[i][0]:.4f} (True: {y_true[i][0]})")
