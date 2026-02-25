# Batch Normalization and Dropout

**Prerequisites**: Forward/backward propagation (Study Guides 02–03), activation functions (Study Guide 04)
**USAAIO Relevance**: Round 1 asks you to compute batch norm by hand (given a batch of values, compute normalized output). Round 2 requires implementing batch norm and dropout from scratch. Understanding training vs. inference mode is critical.

---

## Discovery

### The Problem: Internal Covariate Shift

As training progresses, the distribution of inputs to each layer changes because earlier layers' weights change. Layer 5 learns to expect certain input statistics, but layer 4 keeps shifting those statistics.

> **Socratic question**: If you train a model to normalize images with pixel values in [0, 1], what happens if you feed it images with pixels in [0, 255]?
>
> *The model fails — it was trained on different input statistics. The same thing happens internally: each layer is like a mini-model that expects its inputs to have certain statistical properties.*

Ioffe and Szegedy (2015) proposed **batch normalization**: explicitly normalize each layer's inputs to have zero mean and unit variance, then let the network learn to undo the normalization if needed.

### The Problem: Overfitting

Deep networks have millions of parameters — far more than training samples in many cases. They can memorize the training data perfectly, but fail on new data.

Srivastava et al. (2014) proposed **dropout**: during training, randomly zero out neurons with some probability. This forces the network to be robust — it cannot rely on any single neuron.

> **Analogy**: Dropout is like training an ensemble of networks that share weights. At test time, you average all the ensemble's predictions (which is approximated by the full network without dropout).

---

## Intuition

### Batch Normalization Step by Step

For a batch of activations $\{x_1, x_2, \ldots, x_m\}$ at some layer:

```
Step 1: Compute batch mean
  μ = (x₁ + x₂ + ... + xₘ) / m

Step 2: Compute batch variance
  σ² = Σ(xᵢ - μ)² / m

Step 3: Normalize
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)     ← now has mean=0, variance=1

Step 4: Scale and shift (learnable)
  yᵢ = γ · x̂ᵢ + β                  ← γ and β are learned parameters
```

Why the learnable $\gamma$ and $\beta$? If the optimal representation is NOT zero-mean unit-variance, the network can learn to undo the normalization. If $\gamma = \sigma$ and $\beta = \mu$, we recover the original input.

### Training vs. Inference

```
TRAINING:                          INFERENCE:
┌─────────────────┐               ┌─────────────────┐
│ Use BATCH stats  │               │ Use RUNNING stats│
│ μ_B, σ²_B       │               │ μ_run, σ²_run   │
│ from current     │               │ accumulated over │
│ mini-batch       │               │ all training     │
└─────────────────┘               └─────────────────┘

During training, update running stats:
  μ_run  ← (1 - α)·μ_run  + α·μ_B     (default α = 0.1)
  σ²_run ← (1 - α)·σ²_run + α·σ²_B
```

**Critical for USAAIO**: You MUST call `model.eval()` before inference. Otherwise, batch norm uses batch statistics (which are meaningless for a single test sample).

### Dropout Visualization

```
Training (p=0.5):                    Inference:
┌───┬───┬───┬───┬───┐              ┌───┬───┬───┬───┬───┐
│ h₁│ 0 │ h₃│ 0 │ h₅│              │ h₁│ h₂│ h₃│ h₄│ h₅│
└───┴───┴───┴───┴───┘              └───┴───┴───┴───┴───┘
  ×2   ×2   ×2   ×2   ×2           ×1   ×1   ×1   ×1   ×1
  (scale by 1/(1-p))               (no scaling needed)

With inverted dropout (PyTorch default), scaling happens during training.
At inference, outputs pass through unchanged.
```

---

## Math

### Batch Normalization Forward Pass

Given a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ of pre-activations:

$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

$$\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma, \beta \in \mathbb{R}^C$ are learnable, $\epsilon \approx 10^{-5}$ prevents division by zero.

### Batch Normalization Backward Pass

The backward pass through batch norm is non-trivial because each sample's gradient depends on ALL other samples in the batch (through $\mu$ and $\sigma^2$).

Let $\frac{\partial L}{\partial y_i}$ be the incoming gradient.

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

$$\frac{\partial L}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \frac{-1}{2} (\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}$$

$$\frac{\partial L}{\partial \mu_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_\mathcal{B}^2} \cdot \frac{-2}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})$$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial L}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

### BatchNorm for 2D (Images)

For `BatchNorm2d` with input shape $(B, C, H, W)$:
- Statistics are computed **per channel** across $(B, H, W)$
- $\mu, \sigma^2 \in \mathbb{R}^C$ (one mean/variance per channel)
- $\gamma, \beta \in \mathbb{R}^C$ (one scale/shift per channel)

### Dropout Mathematics

During training with drop probability $p$:

$$\tilde{h}_i = \frac{m_i}{1-p} \cdot h_i, \quad m_i \sim \text{Bernoulli}(1-p)$$

**Expected value**: $E[\tilde{h}_i] = \frac{(1-p)}{1-p} \cdot h_i = h_i$

The $\frac{1}{1-p}$ scaling ensures the expected output matches the no-dropout output, so no adjustment is needed at inference.

---

## Code

### Batch Norm from Scratch

```python
import torch

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.
    x: (B, D) input
    gamma: (D,) scale parameter
    beta: (D,) shift parameter
    Returns: (B, D) normalized output, cache for backward
    """
    mu = x.mean(dim=0)                           # (D,)
    var = x.var(dim=0, unbiased=False)            # (D,) — population variance
    x_hat = (x - mu) / torch.sqrt(var + eps)      # (B, D)
    y = gamma * x_hat + beta                       # (B, D)
    cache = (x, x_hat, mu, var, gamma, eps)
    return y, cache

def batch_norm_backward(dy, cache):
    """
    Batch normalization backward pass.
    dy: (B, D) gradient of loss w.r.t. output
    cache: saved values from forward pass
    Returns: dx, dgamma, dbeta
    """
    x, x_hat, mu, var, gamma, eps = cache
    B = x.shape[0]
    std_inv = 1.0 / torch.sqrt(var + eps)          # (D,)

    dgamma = (dy * x_hat).sum(dim=0)               # (D,)
    dbeta = dy.sum(dim=0)                            # (D,)

    dx_hat = dy * gamma                              # (B, D)
    dvar = (dx_hat * (x - mu) * -0.5 * std_inv**3).sum(dim=0)
    dmu = (dx_hat * -std_inv).sum(dim=0) + dvar * (-2.0 / B) * (x - mu).sum(dim=0)
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mu) / B + dmu / B

    return dx, dgamma, dbeta

# Test
B, D = 32, 64
x = torch.randn(B, D, requires_grad=True)
gamma = torch.ones(D, requires_grad=True)
beta = torch.zeros(D, requires_grad=True)

y, cache = batch_norm_forward(x, gamma, beta)
# Verify: output should have ~zero mean and ~unit variance per feature
print(f"Mean: {y.mean(dim=0).abs().max():.6f}")   # close to 0
print(f"Var:  {y.var(dim=0).mean():.6f}")           # close to 1
```

### Dropout from Scratch

```python
import torch

def dropout_forward(x, p=0.5, training=True):
    """
    Inverted dropout.
    x: input tensor
    p: drop probability
    training: if False, pass through unchanged
    """
    if not training or p == 0:
        return x, None
    mask = (torch.rand_like(x) > p).float()        # Bernoulli(1-p) mask
    out = x * mask / (1 - p)                         # Scale by 1/(1-p)
    return out, mask

def dropout_backward(dy, mask, p=0.5):
    """Backward pass for inverted dropout."""
    if mask is None:
        return dy
    return dy * mask / (1 - p)

# Test
x = torch.randn(4, 8)
out_train, mask = dropout_forward(x, p=0.5, training=True)
out_eval, _ = dropout_forward(x, p=0.5, training=False)

print(f"Training: ~{(mask == 0).float().mean():.1%} of neurons dropped")
print(f"Eval: all neurons active, output unchanged = {torch.allclose(out_eval, x)}")
```

### Using PyTorch's Built-in Modules

```python
import torch
import torch.nn as nn

class MLPWithBNDropout(nn.Module):
    def __init__(self, input_dim=784, hidden=256, output_dim=10, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)       # (B, 784) → (B, 256)
        self.bn1 = nn.BatchNorm1d(hidden)              # Normalize (B, 256)
        self.dropout1 = nn.Dropout(p)                   # Drop p fraction
        self.fc2 = nn.Linear(hidden, output_dim)        # (B, 256) → (B, 10)

    def forward(self, x):
        x = self.fc1(x)                                # (B, 784) → (B, 256)
        x = self.bn1(x)                                # (B, 256) — normalized
        x = torch.relu(x)                              # (B, 256) — activated
        x = self.dropout1(x)                            # (B, 256) — dropped (training only)
        x = self.fc2(x)                                # (B, 256) → (B, 10)
        return x

model = MLPWithBNDropout()

# CRITICAL: switch modes for training vs evaluation
model.train()   # dropout active, batch norm uses batch stats
model.eval()    # dropout off, batch norm uses running stats
```

### Common Ordering Question

**Where does batch norm go — before or after activation?**

Original paper: BN before activation (`Linear → BN → ReLU`). Some papers argue BN after activation works better. In practice, both work. The original ordering is more common:

```python
# Standard ordering (original paper)
x = torch.relu(self.bn(self.linear(x)))

# Alternative ordering
x = self.bn(torch.relu(self.linear(x)))
```

For USAAIO, use `Linear → BN → ReLU → Dropout` unless told otherwise.
