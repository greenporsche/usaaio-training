# Loss Functions

**Prerequisites**: Probability (MLE, cross-entropy), calculus (gradients), tensors
**USAAIO Relevance**: Round 2 problems often require custom compound losses (e.g., PDE residual + boundary + initial conditions). You must understand what each built-in loss computes and how to write your own.

---

## Discovery

### How to Tell a Network It Is Wrong

A neural network makes predictions. The loss function measures how far those predictions are from the truth. Training minimizes this loss. The choice of loss function determines:

1. **What the model optimizes for** — MSE penalizes large errors quadratically, L1 penalizes linearly
2. **The gradient signal** — some losses provide stronger gradients when the model is confident and wrong
3. **Numerical stability** — naive implementations overflow; PyTorch's built-in losses handle this

> **Socratic question**: Why not just count the number of wrong predictions (0-1 loss)? Because counting is not differentiable — the gradient is zero everywhere except at the decision boundary, where it is undefined. We need smooth, differentiable surrogates.

### The reduction Parameter

All PyTorch losses accept `reduction`:
- `'mean'` (default): Average over all elements → scalar
- `'sum'`: Sum over all elements → scalar
- `'none'`: No reduction → tensor of per-element losses

Use `'none'` when you need per-sample losses (e.g., for sample weighting or curriculum learning).

---

## Intuition

### Regression Losses

```
MSE Loss (L2):                     L1 Loss:
Error │    ╱                       Error │    ╱
      │   ╱                              │   ╱
      │  ╱                               │  ╱
      │ ╱     ← quadratic                │ ╱    ← linear
      │╱                                 │╱
──────┼────── residual            ──────┼────── residual
      │╲                                 │╲
      │ ╲                                │ ╲

MSE amplifies large errors.         L1 is robust to outliers.
Gradient = 2(y_hat - y)             Gradient = sign(y_hat - y)
```

### Classification Losses

For multi-class classification with $C$ classes:

```
CrossEntropyLoss pipeline:

Raw logits     LogSoftmax           NLLLoss
[2.1, 0.5, -1.0]  →  [-0.14, -1.84, -3.34]  →  pick -log(p_true)
                         ↑ sum to 0 in log space
```

**Critical**: `nn.CrossEntropyLoss` applies softmax internally. Do NOT apply softmax before passing logits to this loss.

### Binary vs Multi-Class

```
Binary (2 classes):                Multi-class (C classes):
┌──────────────────────┐           ┌──────────────────────┐
│ Model outputs: 1 logit│           │ Model outputs: C logits│
│ Loss: BCEWithLogitsLoss│          │ Loss: CrossEntropyLoss│
│ (applies sigmoid)     │           │ (applies softmax)    │
└──────────────────────┘           └──────────────────────┘
```

---

## Math

### Mean Squared Error (MSE)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Gradient with respect to prediction:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

### Cross-Entropy Loss

For a single sample with true class $c$ and predicted logits $z \in \mathbb{R}^C$:

$$\mathcal{L}_{\text{CE}} = -\log \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}} = -z_c + \log \sum_{j=1}^{C} e^{z_j}$$

This is numerically stable because PyTorch uses the log-sum-exp trick:

$$\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \quad m = \max_j z_j$$

### Binary Cross-Entropy with Logits

For a single sample with true label $y \in \{0, 1\}$ and predicted logit $z$:

$$\mathcal{L}_{\text{BCE}} = -[y \cdot \log \sigma(z) + (1-y) \cdot \log(1 - \sigma(z))]$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

### Custom Compound Loss (USAAIO Pattern)

For PINNs, the loss combines multiple terms:

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{PDE}} + \lambda_2 \mathcal{L}_{\text{BC}} + \lambda_3 \mathcal{L}_{\text{IC}}$$

where each $\lambda_i$ is a weighting coefficient balancing the relative importance of each constraint.

---

## Code

### MSELoss

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss()                    # reduction='mean' by default

y_pred = torch.tensor([2.5, 0.0, 2.1])
y_true = torch.tensor([3.0, -0.5, 2.0])

loss = criterion(y_pred, y_true)
print(loss)                                 # tensor(0.1167)
# = ((2.5-3)^2 + (0-(-0.5))^2 + (2.1-2)^2) / 3 = (0.25 + 0.25 + 0.01) / 3

# Per-element losses
criterion_none = nn.MSELoss(reduction='none')
losses = criterion_none(y_pred, y_true)
print(losses)                               # tensor([0.2500, 0.2500, 0.0100])
```

### CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()

# logits: raw scores, NOT probabilities
logits = torch.tensor([[2.0, 1.0, 0.1],    # sample 0: class 0 most likely
                        [0.5, 2.5, 0.3]])   # sample 1: class 1 most likely
# shape: (2, 3) — 2 samples, 3 classes

labels = torch.tensor([0, 1])               # true classes
# shape: (2,) — dtype must be long

loss = criterion(logits, labels)
print(loss)                                  # scalar

# WRONG: do NOT apply softmax first!
# probs = torch.softmax(logits, dim=-1)     # <-- NEVER do this before CE loss
# loss = criterion(probs, labels)            # <-- WRONG, double softmax
```

### BCEWithLogitsLoss

```python
criterion = nn.BCEWithLogitsLoss()

logits = torch.tensor([0.5, -1.0, 2.0])    # raw logits (not probabilities)
labels = torch.tensor([1.0, 0.0, 1.0])     # binary labels (float, not long!)

loss = criterion(logits, labels)
print(loss)                                  # scalar

# With class weights for imbalanced data
pos_weight = torch.tensor([3.0])            # Positive class is 3x rarer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### NLLLoss (with LogSoftmax)

```python
# NLLLoss expects LOG-probabilities, not raw logits
log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
log_probs = log_softmax(logits)             # Apply log-softmax manually
labels = torch.tensor([0, 1])

loss = nll_loss(log_probs, labels)
# This is EQUIVALENT to CrossEntropyLoss()(logits, labels)
```

### Custom Loss Function (as a plain function)

```python
def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    """
    Focal Loss: down-weights easy examples, focuses on hard ones.
    Used in object detection (RetinaNet).
    """
    ce_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)                 # p_t = probability of true class
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()

# Usage
logits = torch.randn(32, 10)                # (B, C)
labels = torch.randint(0, 10, (32,))        # (B,)
loss = focal_loss(logits, labels)
```

### Custom Loss as nn.Module

```python
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese networks.
    Pulls similar pairs together, pushes dissimilar pairs apart.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # output1, output2: (B, D) embeddings
        # label: (B,) — 1 if same class, 0 if different
        dist = torch.nn.functional.pairwise_distance(output1, output2)  # (B,)

        loss = label * dist ** 2 + \
               (1 - label) * torch.clamp(self.margin - dist, min=0) ** 2
        return loss.mean()

criterion = ContrastiveLoss(margin=2.0)
```

### Compound Loss (USAAIO PINNs Style)

```python
class PINNLoss(nn.Module):
    """
    Compound loss for Physics-Informed Neural Networks.
    L = lambda_pde * L_pde + lambda_bc * L_bc + lambda_ic * L_ic
    """
    def __init__(self, lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
        super().__init__()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

    def forward(self, pde_residual, bc_pred, bc_true, ic_pred, ic_true):
        loss_pde = (pde_residual ** 2).mean()       # PDE should be zero
        loss_bc = ((bc_pred - bc_true) ** 2).mean()  # Boundary conditions
        loss_ic = ((ic_pred - ic_true) ** 2).mean()  # Initial conditions

        total = (self.lambda_pde * loss_pde +
                 self.lambda_bc * loss_bc +
                 self.lambda_ic * loss_ic)

        return total, {
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item(),
            'total': total.item(),
        }
```

### Label Smoothing

```python
# Built-in support in CrossEntropyLoss
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Converts hard labels [0, 0, 1, 0] to soft labels [0.025, 0.025, 0.925, 0.025]
# Prevents overconfident predictions
```
