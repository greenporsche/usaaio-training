# 02 — Advanced Optimization

> Beyond SGD: second-order methods, meta-learning, and curriculum learning for competition-level AI.

---

## Discovery

### Why Adam Isn't Always Enough

Adam (and SGD with momentum) are **first-order methods** — they only use the gradient $\nabla L$, which tells you the direction of steepest descent but nothing about the **curvature** of the loss landscape.

Consider a narrow ravine in the loss landscape. First-order methods oscillate back and forth across the ravine while making slow progress along it. **Second-order methods** use curvature information (the Hessian $H$) to take a direct path.

### When You Need More Than Adam

- **PINNs** — the compound loss landscape is often ill-conditioned. L-BFGS converges where Adam struggles.
- **Fine-tuning** — near a minimum, second-order methods converge quadratically vs. linearly.
- **Few-shot learning** — meta-learning (MAML) requires differentiating through optimization steps.
- **Non-stationary objectives** — curriculum learning adapts the training distribution over time.

---

## Intuition

### Second-Order Methods

**Newton's method** uses the Hessian to compute the optimal step:

$$\theta_{k+1} = \theta_k - H^{-1} \nabla L(\theta_k)$$

where $H = \nabla^2 L$ is the Hessian matrix (matrix of second derivatives).

**Why it works:** Near a minimum, the loss is approximately quadratic: $L(\theta) \approx L(\theta^*) + \frac{1}{2}(\theta - \theta^*)^T H (\theta - \theta^*)$. Newton's method finds the minimum of this quadratic in one step.

**Why it's impractical:** For a model with $n$ parameters, the Hessian is $n \times n$. Inverting it costs $O(n^3)$. For a model with 1M parameters, that's $10^{18}$ operations per step.

### L-BFGS: Practical Second-Order

L-BFGS (Limited-memory BFGS) approximates $H^{-1}$ using only the last $m$ gradient differences. Memory: $O(mn)$ instead of $O(n^2)$.

```python
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=20,          # inner iterations per step
    history_size=10,      # m: number of stored gradient pairs
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer.zero_grad()
    loss = compute_loss(model, data)
    loss.backward()
    return loss

# Each call does multiple inner iterations
optimizer.step(closure)
```

**Key differences from Adam:**
- Requires a `closure` function that recomputes the loss
- `closure` is called multiple times per `step()` (line search)
- Typically needs full-batch or large-batch gradients
- Learning rate of 1.0 is standard (the Hessian approximation handles scaling)

**When to use L-BFGS:**
- PINNs (after initial Adam training)
- Small models where full-batch is feasible
- When Adam plateaus and you need to squeeze out more accuracy

**Common PINN training pattern:**
```python
# Phase 1: Adam for initial convergence
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(5000):
    # ... Adam training ...

# Phase 2: L-BFGS for final refinement
optimizer_lbfgs = torch.optim.LBFGS(model.parameters())
for epoch in range(100):
    optimizer_lbfgs.step(closure)
```

### Natural Gradient

The natural gradient uses the **Fisher information matrix** $F$ instead of the Hessian:

$$\theta_{k+1} = \theta_k - \eta F^{-1} \nabla L(\theta_k)$$

The Fisher information matrix measures how much the output distribution changes when parameters change. This makes the update invariant to the parameterization of the model.

**Practical approximations:**
- **K-FAC** (Kronecker-Factored Approximate Curvature): approximates $F$ as a Kronecker product of smaller matrices
- **Adam** is loosely related: the second moment acts as a diagonal approximation to $F$

---

## Mastery

### Meta-Learning: MAML

**Model-Agnostic Meta-Learning** learns an initialization $\theta$ that can be quickly adapted to new tasks.

**Inner loop** (task-specific adaptation):
$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{task}_i}(\theta)$$

**Outer loop** (meta-update across tasks):
$$\theta \leftarrow \theta - \beta \nabla_\theta \sum_{i} \mathcal{L}_{\text{task}_i}(\theta'_i)$$

The outer gradient involves **differentiating through the inner optimization step**, which requires second-order gradients.

```python
import torch
import torch.nn.functional as F
from torch import autograd

def maml_step(model, tasks, inner_lr=0.01, outer_lr=0.001):
    meta_loss = 0.0

    for support_x, support_y, query_x, query_y in tasks:
        # Inner loop: adapt to this task
        # Create a copy of parameters for the inner loop
        fast_weights = [p.clone() for p in model.parameters()]

        # Forward pass with current weights
        pred = functional_forward(model, support_x, fast_weights)
        inner_loss = F.mse_loss(pred, support_y)

        # Compute gradients for inner update
        grads = autograd.grad(inner_loss, fast_weights, create_graph=True)

        # Update fast weights
        fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

        # Evaluate adapted model on query set
        query_pred = functional_forward(model, query_x, fast_weights)
        task_loss = F.mse_loss(query_pred, query_y)
        meta_loss += task_loss

    # Outer loop: update original parameters
    meta_loss /= len(tasks)
    meta_loss.backward()
    # Update with outer optimizer
```

**Key insight:** `create_graph=True` in the inner loop is essential — without it, the outer loop cannot compute gradients through the adaptation step.

### Curriculum Learning

**Idea:** Present training examples in order of increasing difficulty.

```python
def difficulty_score(sample):
    """Lower score = easier sample."""
    # Could be: loss on a pretrained model, data complexity, etc.
    return score

# Sort by difficulty
sorted_indices = sorted(range(len(dataset)), key=lambda i: difficulty_score(dataset[i]))

# Train in phases
easy = sorted_indices[:len(sorted_indices)//3]
medium = sorted_indices[:2*len(sorted_indices)//3]
hard = sorted_indices  # all data

for phase, indices in [(easy, 100), (medium, 100), (hard, 200)]:
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    for epoch in range(phase[1]):
        train_one_epoch(model, loader)
```

**For PINNs:** Start training with smooth, low-frequency solutions and gradually introduce higher-frequency components or sharper gradients.

### Learning Rate Schedules (Competition-Relevant)

```python
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=2
)

# One-cycle policy (fast convergence)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=1000
)

# Reduce on plateau (adaptive)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

---

## Connection

### Optimization in the Competition Context

| Scenario | Best Approach |
|----------|--------------|
| Standard training from scratch | Adam + cosine schedule |
| PINN refinement | Adam → L-BFGS two-phase |
| Few-shot adaptation in exam | MAML-style inner loop |
| Training with limited time | OneCycleLR for fast convergence |
| Loss plateaus during training | ReduceLROnPlateau or switch to L-BFGS |

### Exam Implications

Round 2 may ask you to:
- Implement an L-BFGS training loop with a closure
- Explain why second-order methods help for PINNs
- Implement a MAML inner loop with `create_graph=True`
- Design a curriculum for a specific training scenario

---

## Summary

| Method | Key Idea | When to Use |
|--------|----------|-------------|
| Newton's | $\theta - H^{-1}\nabla L$ | Theory only (impractical) |
| L-BFGS | Approximate $H^{-1}$ from gradient history | PINN refinement, small models |
| Natural gradient | $\theta - F^{-1}\nabla L$ | Distribution-sensitive optimization |
| MAML | Learn adaptable initialization | Few-shot, task adaptation |
| Curriculum | Easy → hard training order | Difficult convergence landscapes |
