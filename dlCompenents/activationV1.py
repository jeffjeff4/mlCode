import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 500)
functions = {
    "Sigmoid": lambda x: 1/(1+np.exp(-x)),
    "Tanh": np.tanh,
    "ReLU": lambda x: np.maximum(0, x),
    "Leaky ReLU": lambda x: np.maximum(0.01*x, x),
    "ELU": lambda x: np.where(x>0, x, 0.1*(np.exp(x)-1)),
    "GELU": lambda x: 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))
}

plt.figure(figsize=(12, 6))
for name, func in functions.items():
    plt.plot(x, func(x), label=name, lw=2)
plt.title("Activation Functions Comparison")
plt.legend()
plt.grid()
plt.show()