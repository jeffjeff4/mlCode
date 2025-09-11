import numpy as np
from scipy.optimize import minimize_scalar

def win_rate(bid, k=1.0, b0=1.0):
    """Logistic win rate function: P(win | bid)."""
    return 1 / (1 + np.exp(-k * (bid - b0)))

def expected_margin(bid, valuation, k=1.0, b0=1.0):
    """Calculate expected margin for a bid."""
    return (valuation - bid) * win_rate(bid, k, b0)

def optimize_bid(valuation, k=1.0, b0=1.0):
    """Find the bid that maximizes expected margin."""
    # Objective function to minimize (negative expected margin)
    objective = lambda bid: -expected_margin(bid, valuation, k, b0)
    # Optimize within reasonable bid range (0 to valuation)
    result = minimize_scalar(objective, bounds=(0, valuation), method='bounded')
    return result.x, -result.fun  # Optimal bid and expected margin

# Example
valuation = 2.0  # Value of impression
k, b0 = 2.0, 1.0  # Parameters for win rate function
optimal_bid, max_margin = optimize_bid(valuation, k, b0)
print(f"Optimal Bid: ${optimal_bid:.2f}, Expected Margin: ${max_margin:.2f}")

# Test different bids
bids = [0.5, 1.0, 1.5]
for bid in bids:
    margin = expected_margin(bid, valuation, k, b0)
    print(f"Bid: ${bid:.2f}, Win Rate: {win_rate(bid, k, b0):.2f}, Expected Margin: ${margin:.2f}")


    