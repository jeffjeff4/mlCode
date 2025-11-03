"""
Plain-Python educational simulation of FSDP (Fully Sharded Data Parallel).
- No torch/numpy required.
- Single process simulates multiple ranks (world_size).
- Model: simple linear model y = dot(w, x)
- We show: shard params, local forward, gather preds, compute grads locally,
  reduce-scatter semantics (here we simply sum grads across replica axis if simulated),
  local optimizer step on shards, and full-state consolidation.
"""

from typing import List, Dict, Tuple
import random
import copy
import torch
import torch.distributed as dist


# -------------------------
# Utilities for vectors
# -------------------------
def vec_dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

def vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x+y for x,y in zip(a,b)]

def vec_scale(a: List[float], s: float) -> List[float]:
    return [x*s for x in a]

# -------------------------
# Sharding helpers
# -------------------------
def shard_indices(param_len: int, world_size: int) -> List[Tuple[int,int]]:
    """
    Return list of (start, end) indices (end exclusive) per rank.
    Simple equal (floor) partition; last gets remainder.
    """
    base = param_len // world_size
    rem = param_len % world_size
    idxs = []
    cur = 0
    for r in range(world_size):
        extra = 1 if r < rem else 0
        s = cur
        e = cur + base + extra
        idxs.append((s,e))
        cur = e
    return idxs

# -------------------------
# Simple "Model" container
# -------------------------
class SimpleLinearModel:
    def __init__(self, w: List[float]):
        # full parameter vector (conceptual)
        self.w = list(w)

    def forward_full(self, x: List[float]) -> float:
        return vec_dot(self.w, x)

    def get_param_len(self) -> int:
        return len(self.w)


class ManualMLP:
    """
    A simple MLP with fully manual forward/backward passes.
    This does *not* inherit from nn.Module.
    """

    def __init__(self, input_size, hidden_size, output_size, rank, world_size):
        # Seed to ensure all processes *could* initialize the same,
        # but we will broadcast from rank 0 to guarantee it.
        torch.manual_seed(42)

        # --- 1. Initialize Parameters ---
        self.W1 = torch.randn(input_size, hidden_size) * 0.1
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(hidden_size, output_size) * 0.1
        self.b2 = torch.zeros(output_size)
        self.params = [self.W1, self.b1, self.W2, self.b2]

        # --- 2. Manually Broadcast Weights (Part of "Manual DDP") ---
        # This is a critical step. We must ensure all models
        # start with the exact same weights from rank 0.
        for p in self.params:
            dist.broadcast(p, src=0)

        # --- 3. Buffers for intermediate values (for backward pass) ---
        self.x_input = None
        self.z1 = None
        self.a1 = None
        self.z2 = None  # (This is also y_pred)

        # --- 4. Buffers for gradients ---
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        self.grads = [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]

    def forward(self, x):
        """
        Manual forward pass.
        We save intermediate values needed for the backward pass.
        """
        self.x_input = x

        # Layer 1: z1 = X @ W1 + b1
        self.z1 = torch.matmul(x, self.W1) + self.b1

        # Activation 1: a1 = ReLU(z1)
        self.a1 = torch.maximum(torch.tensor(0.0), self.z1)

        # Layer 2: z2 = a1 @ W2 + b2
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2

        return self.z2  # z2 is our prediction, y_pred

    def backward(self, y_true):
        """
        Manual backward pass (Chain Rule).
        This computes the gradients *only* for the local batch.
        """
        N = y_true.shape[0]  # Batch size
        y_pred = self.z2

        # --- Start at the end: Gradient of Loss w.r.t. Prediction ---
        # Loss = (1/N) * sum( (y_pred - y_true)^2 )
        # dL/dy_pred = (2/N) * (y_pred - y_true)
        grad_y_pred = (2.0 / N) * (y_pred - y_true)

        # --- Step 1: Gradients for Layer 2 (W2, b2) ---
        # y_pred (or z2) = a1 @ W2 + b2
        grad_z2 = grad_y_pred  # Gradient just passes through

        # dL/dW2 = (dL/dz2) * (dz2/dW2) = grad_z2 * a1
        self.grad_W2 = torch.matmul(self.a1.T, grad_z2)

        # dL/db2 = (dL/dz2) * (dz2/db2) = grad_z2 * 1
        self.grad_b2 = torch.sum(grad_z2, axis=0)

        # --- Step 2: Propagate Gradients to Layer 1 (W1, b1) ---

        # dL/da1 = (dL/dz2) * (dz2/da1) = grad_z2 @ W2.T
        grad_a1 = torch.matmul(grad_z2, self.W2.T)

        # dL/dz1 = (dL/da1) * (da1/dz1)
        # da1/dz1 is the derivative of ReLU
        relu_deriv = (self.z1 > 0).float()
        grad_z1 = grad_a1 * relu_deriv  # Element-wise

        # dL/dW1 = (dL/dz1) * (dz1/dW1) = grad_z1 * x_input
        self.grad_W1 = torch.matmul(self.x_input.T, grad_z1)

        # dL/db1 = (dL/dz1) * (dz1/db1) = grad_z1 * 1
        self.grad_b1 = torch.sum(grad_z1, axis=0)

        # Store grads in a list for easier averaging
        self.grads = [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]

    def average_gradients(self, world_size):
        """
        This is the **MANUAL DDP** step.
        We use all_reduce (sum) and then divide by world_size
        to get the average gradient across all processes.
        """
        for grad_tensor in self.grads:
            # Sum all gradients from all processes
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM)
            # Divide by the number of processes to get the average
            grad_tensor /= world_size

    def update_weights(self, lr):
        """
        Manual optimizer step (SGD).
        This is identical on all processes because they all
        have the same averaged gradients.
        """
        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2


# -------------------------
# FSDP Rank (simulated)
# Each rank stores only its shard of parameters (w_shard)
# -------------------------
class FSDPRank:
    #def __init__(self, rank_id: int, world_size: int, full_model: SimpleLinearModel):
    def __init__(self, rank_id: int, world_size: int, full_model: ManualMLP):

        self.rank = rank_id
        self.world_size = world_size
        self.full_len = full_model.get_param_len()
        self.shard_map = shard_indices(self.full_len, world_size)
        self.start, self.end = self.shard_map[rank_id]
        # local shard (copy of the slice)
        self.local_w = full_model.w[self.start:self.end]
        # local grad placeholder (same shape as local_w)
        self.local_grad = [0.0] * len(self.local_w)
        # simple SGD opt state: learning rate
        self.lr = 0.1

    def local_forward_contrib(self, x: List[float]) -> float:
        """Compute local contribution to dot(w, x) using only local shard."""
        x_slice = x[self.start:self.end]
        return vec_dot(self.local_w, x_slice)

    def compute_local_gradients(self, x: List[float], global_pred: float, y: float):
        """
        Compute gradient of loss L = (pred - y)^2 wrt local params:
        grad_w_i = 2*(pred - y) * x_i
        Here global_pred should be full dot(w,x), not just local.
        """
        err = global_pred - y
        x_slice = x[self.start:self.end]
        self.local_grad = [2.0 * err * xi for xi in x_slice]

    def apply_local_step(self):
        """SGD update on local shard."""
        self.local_w = [w - self.lr * g for w, g in zip(self.local_w, self.local_grad)]

    def replace_shard(self, new_shard: List[float]):
        self.local_w = list(new_shard)

    def get_shard(self) -> List[float]:
        return list(self.local_w)

    def zero_local_grad(self):
        self.local_grad = [0.0] * len(self.local_w)

# -------------------------
# Simulated communication primitives
# (Since we're single-process simulating, these are simple aggregations)
# -------------------------
def all_gather_preds(local_contribs: List[float]) -> float:
    """All-gather and sum local contributions to get full prediction."""
    return sum(local_contribs)

def reduce_scatter_sum(sharded_grads_across_replicas: List[List[List[float]]]) -> List[List[float]]:
    """
    Simulate reduce-scatter across data-parallel replicas but here simplified:
    Input shape simulated as [replica_id][rank_id][shard_len]
    For single replica case, we just sum across replicas per rank and return list per rank.
    Output: list of grads per rank (already reduced and assigned to corresponding rank)
    """
    num_replicas = len(sharded_grads_across_replicas)
    world_size = len(sharded_grads_across_replicas[0])
    out = []
    for r in range(world_size):
        # sum replica contributions for rank r
        sum_shard = None
        for rep in range(num_replicas):
            if sum_shard is None:
                sum_shard = list(sharded_grads_across_replicas[rep][r])
            else:
                sum_shard = [a+b for a,b in zip(sum_shard, sharded_grads_across_replicas[rep][r])]
        out.append(sum_shard)
    return out

def consolidate_full_state(ranks: List[FSDPRank]) -> List[float]:
    """Gather each rank's shard to build the full parameter vector in order."""
    full = []
    for r in ranks:
        full.extend(r.get_shard())
    return full

# -------------------------
# Putting it together: one training step simulation
# -------------------------
#def fsdp_step_simulation(full_model: SimpleLinearModel, ranks: List[FSDPRank],
#                         x: List[float], y: float, num_replicas: int = 1):
def fsdp_step_simulation(full_model: ManualMLP, ranks: List[FSDPRank],
                         x: List[float], y: float, num_replicas: int = 1):

    """
    Simulate one step:
    - local forward contributions
    - gather preds (all-gather sum)
    - compute local grads (each rank)
    - simulate reduce-scatter (sum across replicas) -> get reduced grads per rank
    - local update on each rank with reduced gradient
    - (optionally consolidate full state for logging)
    """
    world_size = len(ranks)

    # 1) each rank computes local forward contribution
    local_contribs = [r.local_forward_contrib(x) for r in ranks]
    # 2) all-gather (sum) to get global prediction
    pred = all_gather_preds(local_contribs)
    loss = (pred - y)**2

    # 3) each rank computes local gradients (w.r.t its shard), using global pred
    for r in ranks:
        r.compute_local_gradients(x, pred, y)

    # If simulating multiple data-parallel replicas, we would have gradient shards per replica.
    # For simplicity, assume num_replicas replicas with identical grads here (or could simulate noise).
    # Build sharded_grads_across_replicas: [replica][rank][shard]
    sharded_grads_across_replicas = []
    for rep_id in range(num_replicas):
        rep_list = []
        for r in ranks:
            # Could add tiny noise per replica to simulate difference; here identical
            rep_list.append(list(r.local_grad))
        sharded_grads_across_replicas.append(rep_list)

    # 4) reduce-scatter-sum: sum grads across replicas and produce per-rank reduced grad
    reduced_per_rank = reduce_scatter_sum(sharded_grads_across_replicas)

    # 5) each rank replaces its local_grad with reduced gradient and applies optimizer step
    for idx, r in enumerate(ranks):
        r.local_grad = reduced_per_rank[idx]
        r.apply_local_step()

    return pred, loss

# -------------------------
# Demo / Example usage
# -------------------------
def demo():
    random.seed(0)
    # full model w dimension
    D = 7
    # initialize full parameter vector
    full_w = [random.uniform(-1,1) for _ in range(D)]
    full_model = SimpleLinearModel(full_w)
    print("Initial full w:", [round(v,3) for v in full_model.w])

    # world_size: number of FSDP shards (ranks)
    world_size = 3
    ranks = [FSDPRank(r, world_size, full_model) for r in range(world_size)]
    for i,r in enumerate(ranks):
        s,e = r.start, r.end
        print(f"Rank {i} holds indices [{s},{e}) -> shard:", [round(v,3) for v in r.get_shard()])

    # a single training sample (x,y)
    x = [random.uniform(-1,1) for _ in range(D)]
    # target computed from original full model plus some noise
    y = vec_dot(full_model.w, x) + 0.5  # want the model to fit y

    print("x:", [round(v,3) for v in x], "y:", round(y,3))

    # Run a few training steps (simulate single replica)
    for step in range(5):
        pred, loss = fsdp_step_simulation(full_model, ranks, x, y, num_replicas=1)
        full_now = consolidate_full_state(ranks)
        print(f"Step {step}: pred={pred:.4f}, loss={loss:.6f}")
        print(" Full w:", [round(v,4) for v in full_now])

    print("Final consolidated params:", [round(v,4) for v in consolidate_full_state(ranks)])

if __name__ == "__main__":
    demo()
