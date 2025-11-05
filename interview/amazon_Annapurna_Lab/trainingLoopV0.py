import numpy as np


class SimpleNN:
    """
    A simple two-layer feed-forward neural network.
    Architecture: Input Layer -> Hidden Layer (with sigmoid) -> Output Layer (linear)
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the weights and biases for the network with random values.

        Args:
            input_size (int): The number of features in the input data.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of neurons in the output layer.
        """
        # np.random.randn generates samples from the standard normal distribution
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Cache for storing intermediate values needed for backpropagation
        self.cache = {}

    def _sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_derivative(self, Z):
        """Derivative of the sigmoid function."""
        s = self._sigmoid(Z)
        return s * (1 - s)

    def forward(self, X):
        """
        Performs the forward pass of the network.

        Args:
            X (np.ndarray): The input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: The output of the network of shape (batch_size, output_size).
        """
        # Layer 1 (Hidden)
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._sigmoid(Z1)

        # Layer 2 (Output)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = Z2  # Linear activation for the output layer

        # Store values in cache for the backward pass
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2

    def backward(self, dL_dA2):
        """
        Performs the backward pass (backpropagation) to compute gradients.

        Args:
            dL_dA2 (np.ndarray): The gradient of the loss with respect to the network's output.
        """
        m = dL_dA2.shape[0]  # Number of examples in the batch

        # Retrieve cached values
        X, Z1, A1, Z2, A2 = self.cache["X"], self.cache["Z1"], self.cache["A1"], self.cache["Z2"], self.cache["A2"]

        # Gradients for Layer 2 (Output)
        # For a linear activation, dZ2 = dL_dA2 * 1
        dL_dZ2 = dL_dA2
        self.dL_dW2 = (1 / m) * np.dot(A1.T, dL_dZ2)
        self.dL_db2 = (1 / m) * np.sum(dL_dZ2, axis=0, keepdims=True)

        # Propagate gradient back to the hidden layer
        dL_dA1 = np.dot(dL_dZ2, self.W2.T)

        # Gradients for Layer 1 (Hidden)
        dL_dZ1 = dL_dA1 * self._sigmoid_derivative(Z1)
        self.dL_dW1 = (1 / m) * np.dot(X.T, dL_dZ1)
        self.dL_db1 = (1 / m) * np.sum(dL_dZ1, axis=0, keepdims=True)

    def update_parameters(self, learning_rate):
        """
        Updates the model's parameters using the calculated gradients.
        This is the Stochastic Gradient Descent (SGD) update rule.
        """
        self.W1 -= learning_rate * self.dL_dW1
        self.b1 -= learning_rate * self.dL_db1
        self.W2 -= learning_rate * self.dL_dW2
        self.b2 -= learning_rate * self.dL_db2


def mean_squared_error_loss(Y_true, Y_pred):
    """
    Calculates the Mean Squared Error (MSE) loss and its derivative.

    Returns:
        tuple: A tuple containing the loss (float) and the gradient of the loss (np.ndarray).
    """
    m = Y_true.shape[0]
    loss = (1 / m) * np.sum((Y_pred - Y_true) ** 2)

    # Gradient of MSE loss w.r.t. Y_pred
    grad = (2 / m) * (Y_pred - Y_true)

    return loss, grad


def train(model, X_train, Y_train, epochs, learning_rate, batch_size):
    """
    The main training loop for the neural network.

    Args:
        model (SimpleNN): The neural network model to train.
        X_train (np.ndarray): The training input data.
        Y_train (np.ndarray): The training labels.
        epochs (int): The number of passes through the entire dataset.
        learning_rate (float): The step size for parameter updates.
        batch_size (int): The number of samples to process in each step.
    """
    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        epoch_loss = 0

        # Shuffle data at the beginning of each epoch
        permutation = np.random.permutation(num_samples)
        X_shuffled = X_train[permutation]
        Y_shuffled = Y_train[permutation]

        for i in range(0, num_samples, batch_size):
            # 1. Get mini-batch
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            # 2. Forward Pass: Get predictions
            Y_pred = model.forward(X_batch)

            # 3. Calculate Loss and its gradient
            loss, dL_dA2 = mean_squared_error_loss(Y_batch, Y_pred)
            epoch_loss += loss

            # 4. Backward Pass: Calculate gradients for weights and biases
            model.backward(dL_dA2)

            # 5. Update Parameters (Gradient Descent)
            model.update_parameters(learning_rate)

        avg_epoch_loss = epoch_loss / (num_samples / batch_size)
        if (epoch % 100) == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.6f}")

