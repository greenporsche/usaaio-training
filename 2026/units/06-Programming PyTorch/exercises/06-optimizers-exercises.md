# Optimizers Exercises

**Topic**: SGD, Adam, learning rate scheduling, weight decay, gradient clipping
**Difficulty**: Intermediate

---

## Exercise 1: Predict the Parameter Update

Given the following setup, compute the parameter value after one step of SGD.

```python
import torch
import torch.nn as nn

# Simple linear model: y = w*x + b
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

x = torch.tensor([3.0])
y_true = torch.tensor([10.0])

# Forward
y_pred = w * x + b                           # 2*3 + 1 = 7
loss = (y_pred - y_true) ** 2                 # (7 - 10)^2 = 9

# Backward
loss.backward()

# SGD step with lr=0.01
# What are w and b after the update?
```

<details>
<summary>Solution</summary>

**Compute gradients:**

$\mathcal{L} = (wx + b - y)^2$

$\frac{\partial \mathcal{L}}{\partial w} = 2(wx + b - y) \cdot x = 2(7 - 10)(3) = -18$

$\frac{\partial \mathcal{L}}{\partial b} = 2(wx + b - y) \cdot 1 = 2(7 - 10) = -6$

**SGD update: $\theta \leftarrow \theta - \eta \cdot \nabla$**

$w_{\text{new}} = 2.0 - 0.01 \times (-18) = 2.0 + 0.18 = 2.18$

$b_{\text{new}} = 1.0 - 0.01 \times (-6) = 1.0 + 0.06 = 1.06$

Verify:
```python
optimizer = torch.optim.SGD([w, b], lr=0.01)
optimizer.step()
print(w.item(), b.item())    # 2.18, 1.06
```

**Key insight**: The negative gradient points "uphill." SGD subtracts it, moving "downhill." With lr=0.01, we take a small step. The gradient magnitude tells us the loss is more sensitive to w (gradient = -18) than to b (gradient = -6), because w is multiplied by x=3.
</details>

---

## Exercise 2: SGD with Momentum — Trace Two Steps

Trace the momentum updates for a 1D parameter over 2 steps. Initial: $v_0 = 0$, $\theta_0 = 5.0$, $\eta = 0.1$, $\beta = 0.9$.

Step 1 gradient: $g_1 = 4.0$
Step 2 gradient: $g_2 = -2.0$

Compute $v_1, \theta_1, v_2, \theta_2$.

<details>
<summary>Solution</summary>

**Step 1:**
$$v_1 = \beta v_0 + g_1 = 0.9(0) + 4.0 = 4.0$$
$$\theta_1 = \theta_0 - \eta v_1 = 5.0 - 0.1(4.0) = 4.6$$

**Step 2:**
$$v_2 = \beta v_1 + g_2 = 0.9(4.0) + (-2.0) = 3.6 - 2.0 = 1.6$$
$$\theta_2 = \theta_1 - \eta v_2 = 4.6 - 0.1(1.6) = 4.44$$

Note how momentum carries forward information from step 1: even though the gradient in step 2 is negative (-2.0), the velocity is still positive (1.6) because the previous gradient was large and positive. The parameter continues moving in the same direction, but slower. This is momentum's "inertia" — it smooths out oscillations.

**Key insight**: Without momentum, the step would be $\theta_2 = 4.6 - 0.1(-2.0) = 4.8$ (moving back up). With momentum, $\theta_2 = 4.44$ (continuing downward, but decelerating). Momentum prevents the parameter from oscillating back and forth.
</details>

---

## Exercise 3: Adam vs SGD Decision

For each scenario, recommend SGD or Adam and explain why.

1. Training a ResNet-50 on ImageNet for state-of-the-art accuracy
2. Fine-tuning a pretrained BERT model on a small dataset
3. Training a simple MLP on tabular data for a Kaggle competition
4. USAAIO Round 2: training any model in 4 hours under time pressure
5. Training a GAN (known to be unstable)

<details>
<summary>Solution</summary>

1. **ResNet on ImageNet**: **SGD with momentum** (lr=0.1, momentum=0.9, weight_decay=1e-4, cosine schedule). SGD generalizes better on large-scale image classification — this is well-established empirically and is what most published papers use.

2. **Fine-tuning BERT**: **AdamW** (lr=2e-5 to 5e-5, weight_decay=0.01). Transformers were trained with Adam; fine-tuning works best with the same optimizer family. The small learning rate prevents catastrophic forgetting. AdamW properly decouples weight decay.

3. **MLP on tabular data**: **Adam** (lr=1e-3). For smaller models and quick iteration, Adam converges faster and requires less hyperparameter tuning.

4. **USAAIO Round 2**: **Adam** (lr=1e-3). Under time pressure, Adam is the safest choice — it works well with default hyperparameters and converges quickly. You do not have time to tune SGD's learning rate and schedule.

5. **GAN training**: **Adam** with $\beta_1 = 0.0$ or $0.5$ (not the default 0.9). GANs are adversarial and unstable; the default momentum in Adam can cause oscillation. Lower $\beta_1$ reduces this.

**Key insight**: Adam is the "safe default" — it works well with minimal tuning. SGD can achieve better final performance with careful tuning (learning rate, schedule, momentum), but requires more effort. For competitions with time constraints, start with Adam.
</details>

---

## Exercise 4: Learning Rate Finder

Implement a simple learning rate finder that trains for one epoch while exponentially increasing the learning rate from `lr_min` to `lr_max`, and records the loss at each step.

```python
def lr_find(model, loader, criterion, lr_min=1e-7, lr_max=10, num_steps=100):
    """
    Learning rate finder.
    Returns:
        lrs: list of learning rates tested
        losses: list of corresponding losses
    """
    # YOUR CODE HERE
    pass

# Usage:
# lrs, losses = lr_find(model, train_loader, criterion)
# Plot lrs (log scale) vs losses to find the optimal LR
# Pick the LR where loss is decreasing fastest (steepest slope)
```

<details>
<summary>Solution</summary>

```python
import math
from copy import deepcopy

def lr_find(model, loader, criterion, lr_min=1e-7, lr_max=10, num_steps=100):
    # Save original state to restore later
    original_state = deepcopy(model.state_dict())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)

    # Exponential LR increase: lr = lr_min * (lr_max/lr_min)^(step/num_steps)
    mult = (lr_max / lr_min) ** (1.0 / num_steps)

    lrs = []
    losses = []
    best_loss = float('inf')
    lr = lr_min

    model.train()
    batch_iter = iter(loader)

    for step in range(num_steps):
        # Get next batch (cycle if needed)
        try:
            x, y = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            x, y = next(batch_iter)

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward + backward
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Record
        lrs.append(lr)
        losses.append(loss.item())

        # Stop if loss is exploding
        if loss.item() > 4 * best_loss:
            break
        best_loss = min(best_loss, loss.item())

        # Increase LR
        lr *= mult

    # Restore model to original state
    model.load_state_dict(original_state)

    return lrs, losses
```

**Key insight**: The LR finder sweeps through learning rates in a single epoch. The optimal LR is where the loss decreases fastest — typically about 10x smaller than where the loss starts exploding. This technique (from Leslie Smith / fast.ai) saves hours of manual LR tuning.
</details>

---

## Exercise 5: Compare Optimizer Trajectories

On the Rosenbrock function $f(x, y) = (1-x)^2 + 100(y - x^2)^2$, trace 500 steps of SGD, SGD+Momentum, and Adam starting from $(x_0, y_0) = (-1.5, 1.5)$.

```python
def rosenbrock(xy):
    x, y = xy[0], xy[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

# YOUR CODE: implement optimization for all three optimizers
# Record the trajectory (x, y) at each step
# The minimum is at (1, 1) with f(1, 1) = 0
```

Which optimizer reaches closest to (1, 1) in 500 steps?

<details>
<summary>Solution</summary>

```python
import torch

def optimize_rosenbrock(optimizer_class, lr, num_steps=500, **kwargs):
    xy = torch.tensor([-1.5, 1.5], requires_grad=True)
    optimizer = optimizer_class([xy], lr=lr, **kwargs)
    trajectory = [xy.detach().clone().numpy()]

    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = (1 - xy[0]) ** 2 + 100 * (xy[1] - xy[0] ** 2) ** 2
        loss.backward()
        optimizer.step()
        trajectory.append(xy.detach().clone().numpy())

    final_loss = (1 - xy[0]) ** 2 + 100 * (xy[1] - xy[0] ** 2) ** 2
    return trajectory, final_loss.item()

# SGD (needs very small LR to avoid divergence)
traj_sgd, loss_sgd = optimize_rosenbrock(torch.optim.SGD, lr=1e-4)

# SGD + Momentum
traj_mom, loss_mom = optimize_rosenbrock(
    torch.optim.SGD, lr=1e-4, momentum=0.9
)

# Adam (works well with default-ish LR)
traj_adam, loss_adam = optimize_rosenbrock(torch.optim.Adam, lr=1e-2)

print(f"SGD:      final loss = {loss_sgd:.6f}")
print(f"Momentum: final loss = {loss_mom:.6f}")
print(f"Adam:     final loss = {loss_adam:.6f}")
```

**Typical results**: Adam converges much faster on Rosenbrock because the adaptive learning rates handle the very different curvatures along the x and y directions. SGD struggles because the same learning rate is applied to both dimensions, but the optimal step size is very different for each.

**Key insight**: The Rosenbrock function has a narrow curved valley — the gradient along the valley is small, but the gradient perpendicular to the valley is large. Adam's per-parameter adaptive rates naturally handle this. SGD requires careful LR tuning. In neural networks, similar asymmetries exist across parameters.
</details>
