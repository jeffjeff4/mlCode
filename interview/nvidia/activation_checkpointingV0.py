def activation_checkpointing(layers, input_tensor):
    """
    Simulate activation checkpointing by recomputing intermediate outputs
    instead of storing them.

    Args:
        layers (list[callable]): list of layer functions
        input_tensor: initial input
    Returns:
        final output after passing through all layers
    """
    x = input_tensor
    for layer in layers:
        # Instead of storing x, we recompute it on the fly (simulation)
        x = layer(x)
    return x


# ✅ Test
layers = [lambda x: x + 1, lambda x: x * 2, lambda x: x ** 2]
print(activation_checkpointing(layers, 1))  # ((1+1)*2)^2 = 16
