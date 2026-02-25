# Optimizers

**Prerequisites**: Calculus (gradients, gradient descent), loss functions
**USAAIO Relevance**: Understanding optimizer behavior helps debug training issues. Competition problems may ask you to implement training with specific optimizers or compare convergence behavior.

---

## Discovery

### Navigating the Loss Landscape

Once you have a loss function and its gradient (via autograd), you need an algorithm to update parameters. The simplest is gradient descent:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

But vanilla gradient descent is slow. The loss landscape of neural networks is complex — elongated valleys, saddle points, flat regions. Modern optimizers add two key ideas:

1. **Momentum**: Keep a running average of past gradients to smooth out oscillations and accelerate through narrow valleys
2. **Adaptive learning rates**: Scale the step size per-parameter based on historical gradient magnitudes

> **Socratic question**: Why not just use a very large learning rate to converge faster? Because the loss landscape is not smooth — a large step can overshoot the minimum, and for non-convex objectives, it can even diverge. The right learning rate depends on the local curvature, which varies across parameters.

### The Optimizer Interface

All PyTorch optimizers follow the same pattern:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# In training loop:
optimizer.zero_grad()    # Clear old gradients
loss.backward()          # Compute new gradients
optimizer.step()         # Update parameters using gradients
```

The optimizer holds references to `model.parameters()` and updates them in-place.

---

## Intuition

### SGD with Momentum

```
Without momentum:              With momentum (beta=0.9):
     ↗ ↘                            ────→
   ↗     ↘                         ────→
  ↗       ↘  oscillation          ────→  smooth, fast
   ↗     ↘                         ────→
     ↗ ↘                            ────→
```

Momentum accumulates a velocity vector:

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

The velocity averages recent gradients. If gradients consistently point in one direction, velocity builds up. If they oscillate, velocity dampens.

### Adam: Adaptive Moment Estimation

Adam combines momentum with per-parameter adaptive learning rates:

```
Parameter 1 (large gradients):  step = small  (scale down)
Parameter 2 (small gradients):  step = large  (scale up)
Parameter 3 (medium gradients): step = medium
```

Each parameter gets its own effective learning rate based on the first moment (mean) and second moment (variance) of its gradients.

### Learning Rate Schedules

```
Constant:        Step Decay:          Cosine Annealing:
LR │────────     LR │────┐            LR │╲
   │             │     └──┐             │  ╲
   │             │        └──           │   ╲
   │             │           └──        │    ╲____
   └────── epoch    └────── epoch       └────── epoch
```

Starting with a high learning rate and decreasing it over time is almost always beneficial. The model makes big moves early (explore) and fine-tunes later (exploit).

### Weight Decay vs L2 Regularization

```
L2 regularization:                Weight decay (decoupled):
L_total = L + λ||θ||²            θ_new = θ - η(∇L + λθ)
∇L_total = ∇L + 2λθ              = (1 - ηλ)θ - η∇L
θ_new = θ - η(∇L + 2λθ)

For SGD: equivalent.
For Adam: NOT equivalent! Adam scales ∇L by 1/√v, but doesn't
  scale the regularization term. AdamW decouples them correctly.
```

Use `AdamW` (not `Adam` with `weight_decay`) for proper weight decay.

---

## Math

### SGD Update

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

### SGD with Momentum

$$v_t = \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

### Adam

First moment (mean of gradients):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

Second moment (mean of squared gradients):
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias correction (compensate for zero initialization):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update:
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### AdamW (Decoupled Weight Decay)

$$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

The weight decay $\lambda$ is applied directly to the parameters, not through the gradient.

### Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t \pi}{T_{\max}}\right)$$

This smoothly decreases the learning rate from $\eta_{\max}$ to $\eta_{\min}$ over $T_{\max}$ steps.

---

## Code

### SGD

```python
import torch
import torch.nn as nn

model = nn.Linear(784, 10)

# Basic SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# SGD with momentum (standard choice)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with momentum and weight decay
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
)
```

### Adam and AdamW

```python
# Adam (default choice for most tasks)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Adam with weight decay (NOT recommended — use AdamW instead)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# AdamW (proper decoupled weight decay — preferred)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

### Per-Parameter Learning Rates

```python
# Different learning rates for different parts of the model
optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-4},    # Lower LR for pretrained
    {'params': model.classifier.parameters(), 'lr': 1e-3}, # Higher LR for new head
], lr=1e-3)  # Default LR for any group not specified
```

### Learning Rate Schedulers

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# StepLR: multiply LR by gamma every step_size epochs
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1          # LR *= 0.1 every 30 epochs
)

# CosineAnnealingLR: smooth cosine decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# ReduceLROnPlateau: reduce when metric stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# Training loop with scheduler
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)

    # Step the scheduler
    scheduler.step()                             # For StepLR, CosineAnnealing
    # scheduler.step(val_loss)                   # For ReduceLROnPlateau (needs metric)

    # Check current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr:.6f}")
```

### Gradient Clipping

```python
# Clip gradient norm (prevents exploding gradients in RNNs/Transformers)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Clip gradient values
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Implementing SGD from Scratch

```python
def manual_sgd_step(model, lr):
    """One step of SGD, implemented manually."""
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad

# Usage:
optimizer_manual = None  # No torch optimizer needed
loss.backward()
manual_sgd_step(model, lr=0.01)
# Don't forget to zero gradients:
model.zero_grad()
```

### Implementing Adam from Scratch

```python
class ManualAdam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]   # First moments
        self.v = [torch.zeros_like(p) for p in self.params]   # Second moments

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is None:
                    continue
                g = param.grad

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

### Comparing Optimizers

```python
import torch
import torch.nn as nn
from copy import deepcopy

# Create identical models for fair comparison
base_model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

models = {
    'SGD': deepcopy(base_model),
    'SGD+Momentum': deepcopy(base_model),
    'Adam': deepcopy(base_model),
}

optimizers = {
    'SGD': torch.optim.SGD(models['SGD'].parameters(), lr=0.01),
    'SGD+Momentum': torch.optim.SGD(models['SGD+Momentum'].parameters(),
                                     lr=0.01, momentum=0.9),
    'Adam': torch.optim.Adam(models['Adam'].parameters(), lr=1e-3),
}

# Train and compare convergence
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    for name in models:
        model = models[name]
        optimizer = optimizers[name]
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
```
