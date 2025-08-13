from linecache import cache

import  numpy as np

num_samples = 6
num_features = 4
num_hidden_dim = 7

# --- activation function ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# --- loss function ---
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


# --- network parameters ---
def initialize_parameters(n_input, n_hidden, n_output):
    np.random.seed(42)
    params = {
        "W1": np.random.randn(n_hidden, n_input) * 0.01,
        "b1": np.zeros((n_hidden, 1)),
        "W2": np.random.randn(n_output, n_hidden) * 0.01,
        "b2": np.zeros((n_output, 1)),
    }
    return params

# --- forward propagation ---
def forward(X, params):
    Z1 = params['W1'] @ X + params['b1']
    A1 = relu(Z1)
    Z2 = params['W2'] @ A1 + params['b2']
    A2 = Z2

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# --- backward propagation ---
def backward(y_true, cache, params):
    m = y_true.shape[1]
    dZ2 = mse_loss_derivative(cache['A2'], y_true)
    dW2 = dZ2 @ cache['A1'].T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = params['W2'].T @ dZ2
    dZ1 = dA1 * relu_derivative(cache["Z1"])
    dW1 = dZ1 @ cache['X'].T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

# --- parameter update ---
def update_parameters(params, grads, learning_rate=0.1):
    for key in params:
        params[key] -= learning_rate * grads['d' + key]
    return params

# --- training example ---
if __name__ == "__main__":
    # example data: 2 input features, 1 hidden layer (4 neurons), 1 output
    #X = np.array([[0.5, -1.5], [1.0, 2.0]])
    #y = np.array([[1.0, 0.0]])

    # initialize parameters
    #params = initialize_parameters(n_input=2, n_hidden=4, n_output=1)

    #----------------------------------------------------------------------
    X = np.random.rand(num_samples, num_features).T
    y0 = np.random.rand(num_samples)
    y1 = np.array([y0])

    # initialize parameters
    params = initialize_parameters(n_input=num_features, n_hidden=num_hidden_dim, n_output=1)

    for idx in range(1000):
        y_pred, cache = forward(X, params)
        loss = mse_loss(y_pred, y1)
        grads = backward(y1, cache, params)
        params = update_parameters(params, grads, learning_rate=0.05)

        if idx % 100 == 0:
            print(f"step {idx}, loss: {loss:.4f}")


####1. forward
####
####       X (n_input × m)
####         │
####         │  W1, b1
####         ▼
####    Z1 = W1·X + b1
####         │
####         │ ReLU
####         ▼
####    A1 = ReLU(Z1)
####         │
####         │  W2, b2
####         ▼
####    Z2 = W2·A1 + b2
####         │
####         │ Linear output
####         ▼
####    A2 = Z2   ←────────────── y_true
####         │                      │
####         │                      │ Loss L = MSE(A2, y_true)
####         │                      ▼
####         │              dA2 = 2(A2 - y)/m
####         │
####         ▼
####
####
####2. Backward pass (gradients in red)
####
####       X
####       ▲ \
####       │  \
#### dW1   │   \  dZ1
####       │    \
####       │     A1
####       │    ▲ \
####       │    │  \
####       │    │   \  dZ2
####       │    │    \
####       │    │     A2
####       │    │     ▲
####       │    │     │ dA2 = 2(A2 - y)/m
####       │    │
####       │   dA1 = W2ᵀ·dZ2
####       │
####dW1 = (dZ1)·Xᵀ / m
####db1 = sum(dZ1) / m
####dW2 = (dZ2)·A1ᵀ / m
####db2 = sum(dZ2) / m
####
####
####
####3. Forward and backward flow together
####   (Black = forward, Red = backward)
####X - -------------------------▶ (forward)
####│                            ▲
####│ W1, b1                      │ dZ1
####▼                            │
####Z1 - -------▶ ReLU - -------▶ A1
####│                            ▲
####│                            │ dA1 = W2ᵀ·dZ2
####│ W2, b2                      │
####▼                            │
####Z2 - -------▶ (linear) - ---▶ A2 - -------▶ Loss
####L
####▲
####│ dA2 = 2(A2 - y) / m

