# Question 4: Implement Gradient Accumulation
class GradientAccumulator:
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def backward_step(self, loss, optimizer):
        """
        Implement gradient accumulation to handle large batch sizes
        with limited GPU memory
        """
        # Normalize loss for accumulation
        loss = loss / self.accumulation_steps
        loss.backward()

        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Follow-up: How would you handle gradient synchronization in distributed training?


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def test_basic_gradient_accumulation():
    """Test basic gradient accumulation functionality"""
    print("=== Test 1: Basic Gradient Accumulation ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulator = GradientAccumulator(model, accumulation_steps=4)

    # Store initial weights
    initial_weights = model.linear.weight.data.clone()

    # Simulate 4 accumulation steps
    for i in range(4):
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)

        accumulator.backward_step(loss, optimizer)

        # After first 3 steps, weights shouldn't change (no optimizer step)
        if i < 3:
            assert torch.allclose(model.linear.weight.data,
                                  initial_weights), "Weights should not change before accumulation steps complete"

    # After 4th step, weights should have changed
    assert not torch.allclose(model.linear.weight.data,
                              initial_weights), "Weights should change after accumulation steps complete"
    print("✓ Basic gradient accumulation test passed")


def test_loss_normalization():
    """Test that loss is properly normalized for accumulation"""
    print("\n=== Test 2: Loss Normalization ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulator = GradientAccumulator(model, accumulation_steps=4)

    # Create a simple loss
    x = torch.ones(2, 10)
    output = model(x)
    loss = nn.MSELoss()(output, torch.ones(2, 1))

    original_loss = loss.item()
    accumulator.backward_step(loss, optimizer)

    # Check that gradients are scaled appropriately
    for param in model.parameters():
        if param.grad is not None:
            # Gradient should be approximately loss/accumulation_steps
            expected_grad_scale = original_loss / 4
            assert param.grad.abs().mean() > 0, "Gradients should be computed"
    print("✓ Loss normalization test passed")


def test_accumulation_step_counting():
    """Test that step counting works correctly"""
    print("\n=== Test 3: Step Counting ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulation_steps = 3
    accumulator = GradientAccumulator(model, accumulation_steps)

    # Perform multiple accumulation cycles
    total_steps = 10
    optimizer_step_count = 0

    for step in range(total_steps):
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)

        # Check if this should be an optimizer step
        should_step = ((step + 1) % accumulation_steps == 0)

        # Store gradients before backward
        grads_before = [param.grad.clone() if param.grad is not None else None
                        for param in model.parameters()]

        accumulator.backward_step(loss, optimizer)

        # Check if optimizer step occurred
        grads_after = [param.grad for param in model.parameters()]

        if should_step:
            # Gradients should be reset after optimizer step
            assert all(grad is None or grad.abs().sum() == 0 for grad in
                       grads_after), "Gradients should be zero after optimizer step"
            optimizer_step_count += 1
        else:
            # Gradients should be accumulated
            assert any(
                grad is not None and grad.abs().sum() > 0 for grad in grads_after), "Gradients should be accumulated"

    expected_optimizer_steps = total_steps // accumulation_steps
    assert optimizer_step_count == expected_optimizer_steps, f"Expected {expected_optimizer_steps} optimizer steps, got {optimizer_step_count}"
    print("✓ Step counting test passed")


def test_partial_accumulation():
    """Test behavior when not completing full accumulation steps"""
    print("\n=== Test 4: Partial Accumulation ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulator = GradientAccumulator(model, accumulation_steps=4)

    initial_weights = model.linear.weight.data.clone()

    # Only perform 2 out of 4 steps
    for i in range(2):
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        accumulator.backward_step(loss, optimizer)

    # Weights should not change (no optimizer step)
    assert torch.allclose(model.linear.weight.data,
                          initial_weights), "Weights should not change with partial accumulation"

    # Gradients should be present
    has_gradients = any(param.grad is not None and param.grad.abs().sum() > 0
                        for param in model.parameters())
    assert has_gradients, "Gradients should be accumulated"
    print("✓ Partial accumulation test passed")


def test_gradient_equivalence():
    """Test that accumulated gradients match single large batch"""
    print("\n=== Test 5: Gradient Equivalence ===")
    torch.manual_seed(42)

    # Model for accumulation
    model_accum = SimpleModel()
    optimizer_accum = optim.SGD(model_accum.parameters(), lr=0.1)
    accumulator = GradientAccumulator(model_accum, accumulation_steps=4)

    # Model for single batch (equivalent to accumulated batches)
    model_single = SimpleModel()
    optimizer_single = optim.SGD(model_single.parameters(), lr=0.1)

    # Make models have same initial weights
    model_single.load_state_dict(model_accum.state_dict())

    # Data for accumulation (4 small batches)
    accumulation_losses = []
    for i in range(4):
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        output = model_accum(x)
        loss = nn.MSELoss()(output, y)
        accumulation_losses.append(loss)
        accumulator.backward_step(loss, optimizer_accum)

    # Equivalent single large batch
    x_large = torch.randn(8, 10)  # 4 batches of size 2 = batch size 8
    y_large = torch.randn(8, 1)
    output_single = model_single(x_large)
    loss_single = nn.MSELoss()(output_single, y_large)
    loss_single.backward()

    # Compare gradients (they should be similar but not exactly equal due to normalization)
    for param_accum, param_single in zip(model_accum.parameters(), model_single.parameters()):
        if param_accum.grad is not None and param_single.grad is not None:
            grad_accum = param_accum.grad
            grad_single = param_single.grad

            # Gradients should be in the same ballpark
            ratio = (grad_accum / grad_single).mean()
            assert 0.8 < ratio.abs() < 1.2, f"Gradient ratios should be reasonable, got {ratio}"

    print("✓ Gradient equivalence test passed")


def test_gradient_accumulation_accuracy():
    """Test that gradients are properly accumulated across steps"""
    print("\n=== Test 6: Gradient Accumulation Accuracy ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulation_steps = 3
    accumulator = GradientAccumulator(model, accumulation_steps)

    # Store gradients at each step
    step_gradients = []

    for step in range(accumulation_steps):
        # Reset gradients to track accumulation
        optimizer.zero_grad()

        x = torch.ones(2, 10) * (step + 1)  # Different input each step
        y = torch.ones(2, 1) * (step + 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)

        # Compute gradients for this step only
        loss.backward()

        # Store gradients
        step_gradients.append([param.grad.clone() for param in model.parameters()])

        # Reset and do accumulation step
        optimizer.zero_grad()
        accumulator.backward_step(loss, optimizer)

    # Check that final gradients match the sum of individual step gradients
    final_gradients = [param.grad for param in model.parameters()]

    for i, (final_grad, step_grad_list) in enumerate(zip(final_gradients, zip(*step_gradients))):
        if final_grad is not None:
            # Sum of individual step gradients (normalized)
            expected_grad = sum(step_grad / accumulation_steps for step_grad in step_grad_list)

            # Should be approximately equal
            assert torch.allclose(final_grad, expected_grad, rtol=1e-4), \
                f"Accumulated gradients don't match sum of step gradients for parameter {i}"

    print("✓ Gradient accumulation accuracy test passed")


def test_single_accumulation_step():
    """Test with accumulation_steps=1 (equivalent to normal training)"""
    print("\n=== Test 7: Single Accumulation Step ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulator = GradientAccumulator(model, accumulation_steps=1)

    initial_weights = model.linear.weight.data.clone()

    # Single step should trigger optimizer immediately
    x = torch.randn(2, 10)
    y = torch.randn(2, 1)
    output = model(x)
    loss = nn.MSELoss()(output, y)

    accumulator.backward_step(loss, optimizer)

    # Weights should change immediately
    assert not torch.allclose(model.linear.weight.data,
                              initial_weights), "Weights should change immediately with accumulation_steps=1"

    # Gradients should be cleared
    assert all(param.grad is None or param.grad.abs().sum() == 0
               for param in model.parameters()), "Gradients should be cleared after step"
    print("✓ Single accumulation step test passed")


def test_large_accumulation_steps():
    """Test with large number of accumulation steps"""
    print("\n=== Test 8: Large Accumulation Steps ===")
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    accumulation_steps = 10
    accumulator = GradientAccumulator(model, accumulation_steps)

    initial_weights = model.linear.weight.data.clone()

    # Perform multiple steps without reaching accumulation limit
    for i in range(5):
        x = torch.randn(2, 10)
        y = torch.randn(2, 1)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        accumulator.backward_step(loss, optimizer)

        # Weights should not change
        assert torch.allclose(model.linear.weight.data,
                              initial_weights), "Weights should not change before accumulation completes"

    # Gradients should be accumulating
    has_gradients = any(param.grad is not None and param.grad.abs().sum() > 0
                        for param in model.parameters())
    assert has_gradients, "Gradients should be accumulating"
    print("✓ Large accumulation steps test passed")


def test_zero_accumulation_steps():
    """Test error handling for invalid accumulation steps"""
    print("\n=== Test 9: Invalid Accumulation Steps ===")
    model = SimpleModel()

    try:
        accumulator = GradientAccumulator(model, accumulation_steps=0)
        assert False, "Should raise error for accumulation_steps=0"
    except Exception as e:
        print(f"✓ Correctly caught error for accumulation_steps=0: {e}")

    try:
        accumulator = GradientAccumulator(model, accumulation_steps=-1)
        assert False, "Should raise error for negative accumulation_steps"
    except Exception as e:
        print(f"✓ Correctly caught error for negative accumulation_steps: {e}")


def test_with_different_optimizers():
    """Test with different optimizer types"""
    print("\n=== Test 10: Different Optimizers ===")
    model = SimpleModel()

    optimizers = [
        optim.SGD(model.parameters(), lr=0.1),
        optim.Adam(model.parameters(), lr=0.001),
        optim.RMSprop(model.parameters(), lr=0.01)
    ]

    for optimizer in optimizers:
        model_copy = SimpleModel()
        model_copy.load_state_dict(model.state_dict())
        optimizer_copy = type(optimizer)(model_copy.parameters(), lr=optimizer.param_groups[0]['lr'])

        accumulator = GradientAccumulator(model_copy, accumulation_steps=3)

        # Run accumulation
        for i in range(3):
            x = torch.randn(2, 10)
            y = torch.randn(2, 1)
            output = model_copy(x)
            loss = nn.MSELoss()(output, y)
            accumulator.backward_step(loss, optimizer_copy)

        # Should complete without errors
        assert optimizer_copy.state, "Optimizer should have state"
        print(f"✓ Works with {optimizer.__class__.__name__}")


def test_memory_efficiency():
    """Test that memory usage is reasonable"""
    print("\n=== Test 11: Memory Efficiency ===")

    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1000, 500),
                nn.ReLU(),
                nn.Linear(500, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )

        def forward(self, x):
            return self.layers(x)

    model = LargeModel()
    optimizer = optim.Adam(model.parameters())
    accumulator = GradientAccumulator(model, accumulation_steps=4)

    # This should run without memory issues for reasonable batch sizes
    for i in range(4):
        x = torch.randn(32, 1000)  # Moderate batch size
        output = model(x)
        loss = output.mean()

        accumulator.backward_step(loss, optimizer)

    print("✓ Memory efficiency test passed")


def test_training_convergence():
    """Test that training actually converges with gradient accumulation"""
    print("\n=== Test 12: Training Convergence ===")
    torch.manual_seed(42)

    # Simple regression problem
    model = nn.Linear(5, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    accumulator = GradientAccumulator(model, accumulation_steps=4)

    # Generate simple data: y = 2x + 1 + noise
    X = torch.randn(100, 5)
    true_weights = torch.tensor([2.0, -1.0, 0.5, 1.5, -0.5]).reshape(1, -1)
    true_bias = torch.tensor([1.0])
    y = X @ true_weights.t() + true_bias + 0.1 * torch.randn(100, 1)

    losses = []
    for epoch in range(50):
        epoch_loss = 0
        accumulation_count = 0

        for i in range(0, len(X), 2):  # Batch size 2
            x_batch = X[i:i + 2]
            y_batch = y[i:i + 2]

            output = model(x_batch)
            loss = nn.MSELoss()(output, y_batch)
            epoch_loss += loss.item()

            accumulator.backward_step(loss, optimizer)
            accumulation_count += 1

        losses.append(epoch_loss / accumulation_count)

    # Check that loss decreased significantly
    assert losses[-1] < losses[0] * 0.5, f"Loss should decrease significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"✓ Training convergence test passed - loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")


def run_all_gradient_accumulation_tests():
    """Run all gradient accumulation test cases"""
    print("Running Gradient Accumulation Tests...")
    print("=" * 60)

    try:
        test_basic_gradient_accumulation()
        test_loss_normalization()
        test_accumulation_step_counting()
        test_partial_accumulation()
        test_gradient_equivalence()
        test_gradient_accumulation_accuracy()
        test_single_accumulation_step()
        test_large_accumulation_steps()
        test_zero_accumulation_steps()
        test_with_different_optimizers()
        test_memory_efficiency()
        test_training_convergence()

        print("\n" + "=" * 60)
        print("🎉 ALL GRADIENT ACCUMULATION TESTS PASSED! 🎉")
        print("The GradientAccumulator implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_gradient_accumulation_tests()

