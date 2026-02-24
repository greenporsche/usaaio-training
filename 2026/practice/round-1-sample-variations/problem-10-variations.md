# Problem 10 Variations: Custom PyTorch Modules (EXHAUSTIVE)

> **Original:** Build a ReLU module that subclasses `torch.nn.Module` and is named `My_ReLU`.

> **Core Skills:** `torch.nn.Module` subclassing, `__init__` and `forward` methods, activation function implementations, custom backward passes with `torch.autograd.Function`, parameter registration, module composition

---

## Background: The Philosophy of PyTorch Modules

PyTorch's `nn.Module` is the foundation of all neural network components. Understanding it deeply means understanding:

1. **Initialization (`__init__`)**: Register submodules and parameters
2. **Forward pass (`forward`)**: Define computation graph
3. **Automatic differentiation**: PyTorch tracks operations for backprop
4. **State management**: Parameters vs buffers vs plain tensors

Every layer, activation, and even entire networks are `nn.Module` subclasses. Mastering this pattern unlocks the ability to build anything.

---

## CATEGORY A: Different Activation Functions (Same Pattern)

### Variation A1: Sigmoid Module

Build a Sigmoid module that subclasses `torch.nn.Module` and is named `My_Sigmoid`.

**Part 10.1:** Implement the module.

**Part 10.2:** Verify your module produces the same output as `torch.nn.Sigmoid()` on random input.

**Part 10.3:** What happens if you apply sigmoid to very large positive values? How does PyTorch handle numerical stability?

<details>
<summary>Solution A1</summary>

**10.1:**

```python
import torch
import torch.nn as nn

class My_Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

# Alternative using torch.sigmoid:
class My_Sigmoid_v2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)
```

**10.2:**

```python
# Verification
x = torch.randn(3, 4, requires_grad=True)

my_sigmoid = My_Sigmoid()
torch_sigmoid = nn.Sigmoid()

out_mine = my_sigmoid(x)
out_torch = torch_sigmoid(x)

print("Max difference:", (out_mine - out_torch).abs().max().item())
# Should be essentially 0 (< 1e-7)

# Check gradients too
out_mine.sum().backward()
grad_mine = x.grad.clone()
x.grad.zero_()

out_torch.sum().backward()
grad_torch = x.grad

print("Max gradient difference:", (grad_mine - grad_torch).abs().max().item())
```

**10.3:** For very large positive x:
- `exp(-x)` underflows to 0, so sigmoid → 1 (correct)
- For very large negative x: `exp(-x)` overflows to inf, so 1/(1+inf) → 0 (correct)

PyTorch's built-in `torch.sigmoid` uses a numerically stable implementation that handles both cases:

```python
# Stable implementation idea:
# For x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
# For x < 0: sigmoid(x) = exp(x) / (1 + exp(x))

class My_Sigmoid_Stable(nn.Module):
    def forward(self, x):
        pos_mask = x >= 0
        neg_mask = ~pos_mask

        result = torch.zeros_like(x)
        result[pos_mask] = 1 / (1 + torch.exp(-x[pos_mask]))
        result[neg_mask] = torch.exp(x[neg_mask]) / (1 + torch.exp(x[neg_mask]))

        return result
```

</details>

### Variation A2: Tanh Module

Build a Tanh module that subclasses `torch.nn.Module` and is named `My_Tanh`.

**Part 10.1:** Implement using the definition: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Part 10.2:** Implement using the relationship: tanh(x) = 2·sigmoid(2x) - 1

**Part 10.3:** Verify both implementations match `torch.tanh()`.

<details>
<summary>Solution A2</summary>

**10.1:** Using definition:

```python
class My_Tanh_v1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        exp_neg_x = torch.exp(-x)
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
```

**10.2:** Using sigmoid relationship:

```python
class My_Tanh_v2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 2 * torch.sigmoid(2 * x) - 1
```

**10.3:**

```python
x = torch.randn(3, 4)

v1 = My_Tanh_v1()
v2 = My_Tanh_v2()

out1 = v1(x)
out2 = v2(x)
out_torch = torch.tanh(x)

print("v1 vs torch:", (out1 - out_torch).abs().max().item())  # ~1e-7
print("v2 vs torch:", (out2 - out_torch).abs().max().item())  # ~1e-7
print("v1 vs v2:", (out1 - out2).abs().max().item())          # ~1e-7
```

*Key Insight*: The sigmoid-based implementation is often more numerically stable because `torch.sigmoid` uses internal optimizations.

</details>

### Variation A3: Leaky ReLU Module

Build a Leaky ReLU module that subclasses `torch.nn.Module` with a configurable negative slope.

**Part 10.1:** Implement `My_LeakyReLU(negative_slope=0.01)`.

**Part 10.2:** Show that when `negative_slope=0`, this is equivalent to ReLU.

**Part 10.3:** What is Parametric ReLU (PReLU)? How does it differ from Leaky ReLU?

<details>
<summary>Solution A3</summary>

**10.1:**

```python
class My_LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x >= 0, x, self.negative_slope * x)
        # Alternative: return torch.maximum(x, self.negative_slope * x)
```

**10.2:**

```python
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

leaky_zero = My_LeakyReLU(negative_slope=0.0)
my_relu = My_ReLU()  # From original problem

print("Leaky(0):", leaky_zero(x))  # tensor([0., 0., 0., 1., 2.])
print("ReLU:    ", my_relu(x))     # tensor([0., 0., 0., 1., 2.])
# Identical!
```

**10.3:** PReLU (Parametric ReLU):

- In Leaky ReLU, `negative_slope` is a **fixed hyperparameter**
- In PReLU, `negative_slope` is a **learnable parameter**

```python
class My_PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        # Learnable parameter!
        self.weight = nn.Parameter(torch.full((num_parameters,), init))

    def forward(self, x):
        return torch.where(x >= 0, x, self.weight * x)

# Usage:
prelu = My_PReLU()
print(list(prelu.parameters()))  # Shows the learnable weight
```

*Key Insight*: PReLU learns the optimal negative slope during training. Research (He et al., 2015) showed this can improve accuracy on ImageNet.

</details>

### Variation A4: ELU (Exponential Linear Unit)

Build an ELU module: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0

**Part 10.1:** Implement `My_ELU(alpha=1.0)`.

**Part 10.2:** Plot ELU vs ReLU vs Leaky ReLU for x in [-3, 3].

**Part 10.3:** What advantage does ELU have over ReLU regarding the "dying ReLU" problem?

<details>
<summary>Solution A4</summary>

**10.1:**

```python
class My_ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
```

**10.2:**

```python
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-3, 3, 100)

relu = My_ReLU()
leaky = My_LeakyReLU(0.1)
elu = My_ELU(1.0)

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), relu(x).numpy(), label='ReLU')
plt.plot(x.numpy(), leaky(x).numpy(), label='Leaky ReLU (α=0.1)')
plt.plot(x.numpy(), elu(x).numpy(), label='ELU (α=1.0)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.legend()
plt.title("Activation Function Comparison")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.show()
```

**10.3:** ELU advantages:

1. **Non-zero gradient for negative inputs**: Unlike ReLU (gradient = 0 for x < 0), ELU has gradient α·e^x
2. **Negative outputs**: ELU can output negative values (approaching -α), which helps push the mean activation closer to zero
3. **Smoother**: ELU is smooth everywhere (continuous derivative), unlike ReLU's kink at x=0

The "dying ReLU" problem: If a neuron's weights push all inputs to x < 0, the gradient is always 0, and the neuron never updates. ELU avoids this by always having non-zero gradients for x < 0.

</details>

### Variation A5: GELU (Gaussian Error Linear Unit)

Build a GELU module: GELU(x) = x · Φ(x) where Φ is the standard Gaussian CDF.

**Part 10.1:** Implement using the approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))

**Part 10.2:** Implement using `torch.erf` (the error function).

**Part 10.3:** Why is GELU used in Transformers (BERT, GPT)?

<details>
<summary>Solution A5</summary>

**10.1:** Approximation version (faster):

```python
import math

class My_GELU_Approx(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2_over_pi = math.sqrt(2 / math.pi)

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self.sqrt_2_over_pi * (x + 0.044715 * x**3)))
```

**10.2:** Exact version using erf:

```python
class My_GELU_Exact(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqrt_2 = math.sqrt(2)

    def forward(self, x):
        # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        return x * 0.5 * (1 + torch.erf(x / self.sqrt_2))
```

**Verification:**

```python
x = torch.randn(100)
approx = My_GELU_Approx()
exact = My_GELU_Exact()

print("Max difference:", (approx(x) - exact(x)).abs().max().item())
# Typically < 0.001

print("vs torch.nn.GELU:", (exact(x) - nn.GELU()(x)).abs().max().item())
# Should be ~1e-7
```

**10.3:** Why GELU in Transformers:

1. **Smooth everywhere**: Unlike ReLU, GELU is infinitely differentiable
2. **Non-monotonic**: GELU has a slight "dip" near x≈-0.5, which is argued to help with regularization
3. **Probabilistic interpretation**: GELU(x) = x · P(X ≤ x) where X ~ N(0,1). This can be seen as "gating" the input by its percentile in a Gaussian
4. **Empirical performance**: Experiments showed GELU outperformed ReLU on many NLP tasks

*Key Insight*: GELU was proposed by Hendrycks & Gimpel (2016) and became standard in BERT (2018) and GPT-2 (2019).

</details>

---

## CATEGORY B: Modules with Learnable Parameters

### Variation B1: Parametric Activation

Build a module with learnable scale and shift: f(x) = α · ReLU(x) + β

**Part 10.1:** Implement with `α` and `β` as learnable parameters.

**Part 10.2:** Initialize α=1 and β=0, then train on simple data to see parameters update.

<details>
<summary>Solution B1</summary>

**10.1:**

```python
class ScaledShiftedReLU(nn.Module):
    def __init__(self, init_alpha=1.0, init_beta=0.0):
        super().__init__()
        # Register as Parameters so they're included in model.parameters()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        return self.alpha * torch.relu(x) + self.beta

# Check parameters are registered:
model = ScaledShiftedReLU()
print(list(model.parameters()))  # [alpha, beta]
print(model.state_dict())  # OrderedDict([('alpha', 1.0), ('beta', 0.0)])
```

**10.2:**

```python
# Simple training example
import torch.optim as optim

model = ScaledShiftedReLU(init_alpha=1.0, init_beta=0.0)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Target: f(x) = 2 * relu(x) + 0.5
x = torch.linspace(-2, 2, 100)
y_target = 2 * torch.relu(x) + 0.5

for epoch in range(100):
    y_pred = model(x)
    loss = ((y_pred - y_target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Learned alpha: {model.alpha.item():.4f}")  # Should be ~2.0
print(f"Learned beta: {model.beta.item():.4f}")   # Should be ~0.5
```

</details>

### Variation B2: Channel-wise Scale (like BatchNorm's gamma)

Build a module that applies learnable per-channel scaling.

**Part 10.1:** Implement for input shape (B, C, H, W).

**Part 10.2:** How does this relate to BatchNorm's learnable parameters?

<details>
<summary>Solution B2</summary>

**10.1:**

```python
class ChannelScale(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # One scale parameter per channel
        self.scale = nn.Parameter(torch.ones(num_channels))

    def forward(self, x):
        # x shape: (B, C, H, W)
        # scale shape: (C,) -> need to reshape for broadcasting
        return x * self.scale.view(1, -1, 1, 1)

# Test:
layer = ChannelScale(num_channels=64)
x = torch.randn(8, 64, 32, 32)  # Batch=8, Channels=64, H=W=32
out = layer(x)
print(out.shape)  # torch.Size([8, 64, 32, 32])
```

**10.2:** BatchNorm has two learnable parameters per channel:
- `gamma` (scale): Multiplies the normalized output
- `beta` (shift): Added after scaling

```python
# Simplified BatchNorm (without running mean/var tracking):
class SimpleBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x_norm + beta
```

*Key Insight*: The learnable gamma and beta allow the network to "undo" normalization if beneficial. If gamma learns to equal sqrt(var) and beta learns to equal mean, the output equals the input!

</details>

### Variation B3: Softplus with Learnable Sharpness

Softplus is a smooth approximation to ReLU: softplus(x) = (1/β) · log(1 + e^(βx))

**Part 10.1:** Implement with learnable β parameter.

**Part 10.2:** Show that as β → ∞, softplus → ReLU.

<details>
<summary>Solution B3</summary>

**10.1:**

```python
class LearnableSoftplus(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        # Use log(beta) to ensure beta > 0
        self.log_beta = nn.Parameter(torch.tensor(math.log(init_beta)))

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        beta = self.beta
        # Numerically stable version
        return torch.where(
            x * beta > 20,  # For large values, use linear approx
            x,
            (1 / beta) * torch.log(1 + torch.exp(beta * x))
        )
```

**10.2:**

```python
x = torch.linspace(-2, 2, 100)

plt.figure(figsize=(10, 6))
for beta in [0.5, 1, 2, 5, 10, 50]:
    softplus_beta = (1/beta) * torch.log(1 + torch.exp(beta * x))
    plt.plot(x.numpy(), softplus_beta.numpy(), label=f'β={beta}')

plt.plot(x.numpy(), torch.relu(x).numpy(), 'k--', label='ReLU', linewidth=2)
plt.legend()
plt.title("Softplus approaches ReLU as β → ∞")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.show()
```

As β → ∞:
- For x > 0: e^(βx) dominates → log(1 + e^(βx)) ≈ βx → softplus ≈ x
- For x < 0: e^(βx) → 0 → log(1) = 0 → softplus → 0
- This is exactly ReLU!

</details>

---

## CATEGORY C: Custom Backward with autograd.Function

### Variation C1: Straight-Through Estimator (STE)

Build a module where forward does hard thresholding (0 or 1), but backward passes gradients straight through.

**Part 10.1:** Implement using `torch.autograd.Function`.

**Part 10.2:** Why is STE useful for binary neural networks?

<details>
<summary>Solution C1</summary>

**10.1:**

```python
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Hard threshold: 1 if x >= 0, else 0
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through (identity)
        return grad_output

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return STEFunction.apply(x)

# Test:
x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)
ste = StraightThroughEstimator()

out = ste(x)
print("Forward:", out)  # tensor([0., 0., 1., 1., 1.])

loss = out.sum()
loss.backward()
print("Gradient:", x.grad)  # tensor([1., 1., 1., 1., 1.]) - passed straight through!
```

**10.2:** STE is crucial for binary neural networks because:

1. **Problem**: Binary weights/activations (0/1 or -1/+1) are discrete—no gradient exists
2. **Solution**: During forward pass, use binary values; during backward pass, pretend we used the continuous input
3. **Effect**: The network can learn despite discrete operations

```python
# Example: Binary activation with STE
class BinaryActivation(nn.Module):
    def forward(self, x):
        # Forward: sign(x) ∈ {-1, +1}
        # Backward: gradient of sign is 0, so use STE
        return STESign.apply(x)

class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Clip gradient: only pass through where |x| <= 1
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1] = 0
        return grad_input
```

*Key Insight*: STE was popularized by Bengio et al. (2013) and is the foundation of quantization-aware training.

</details>

### Variation C2: Custom ReLU with Explicit Backward

Implement ReLU using `torch.autograd.Function` to define both forward and backward explicitly.

**Part 10.1:** Implement the Function class.

**Part 10.2:** Verify gradients match PyTorch's built-in ReLU using `torch.autograd.gradcheck`.

<details>
<summary>Solution C2</summary>

**10.1:**

```python
class MyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save what we need for backward
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Gradient is 1 where x > 0, else 0
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input

class MyReLU_Explicit(nn.Module):
    def forward(self, x):
        return MyReLUFunction.apply(x)
```

**10.2:**

```python
from torch.autograd import gradcheck

# Create double precision input for numerical stability
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

# gradcheck compares analytical gradients to numerical gradients
test_passed = gradcheck(MyReLUFunction.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {test_passed}")

# Compare to built-in
x2 = x.clone().detach().requires_grad_(True)
x3 = x.clone().detach().requires_grad_(True)

y_custom = MyReLU_Explicit()(x2)
y_builtin = torch.relu(x3)

y_custom.sum().backward()
y_builtin.sum().backward()

print("Max gradient difference:", (x2.grad - x3.grad).abs().max().item())
```

*Key Insight*: `gradcheck` is invaluable for debugging custom autograd functions. It numerically approximates gradients and compares to your analytical implementation.

</details>

### Variation C3: Clipped Gradient ReLU

Implement ReLU where the backward pass clips gradients to [-1, 1].

**Part 10.1:** Implement using `torch.autograd.Function`.

**Part 10.2:** When might gradient clipping in the activation be useful?

<details>
<summary>Solution C3</summary>

**10.1:**

```python
class ClippedGradReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_value):
        ctx.save_for_backward(x)
        ctx.clip_value = clip_value
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        clip_value = ctx.clip_value

        grad_input = grad_output.clone()
        grad_input[x < 0] = 0  # Standard ReLU backward
        grad_input = torch.clamp(grad_input, -clip_value, clip_value)  # Clip

        return grad_input, None  # None for clip_value (no gradient needed)

class MyClippedReLU(nn.Module):
    def __init__(self, clip_value=1.0):
        super().__init__()
        self.clip_value = clip_value

    def forward(self, x):
        return ClippedGradReLU.apply(x, self.clip_value)

# Test:
x = torch.tensor([5.0], requires_grad=True)
layer = MyClippedReLU(clip_value=1.0)

y = layer(x)
y.backward(torch.tensor([10.0]))  # Large upstream gradient

print("x.grad:", x.grad)  # tensor([1.]) - clipped from 10!
```

**10.2:** Gradient clipping in activations can help:

1. **Prevent exploding gradients**: Especially in deep networks or RNNs
2. **Stabilize training**: Large gradients can cause divergence
3. **Regularization effect**: Limits how much any single sample can influence weights

However, per-activation clipping is less common than global gradient clipping (applied to all parameters at once). Per-activation can distort the gradient direction.

</details>

---

## CATEGORY D: Composite Modules

### Variation D1: Swish / SiLU Module

Build a Swish module: Swish(x) = x · sigmoid(x)

**Part 10.1:** Implement as a single module.

**Part 10.2:** Implement by composing existing modules.

**Part 10.3:** Show the derivative of Swish.

<details>
<summary>Solution D1</summary>

**10.1:** Single module:

```python
class My_Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
```

**10.2:** Composing modules (illustrative, not more efficient):

```python
class My_Swish_Composed(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
```

**10.3:** Derivative of Swish:

Let σ(x) = sigmoid(x). Swish(x) = x · σ(x)

Using product rule:
d/dx[x · σ(x)] = σ(x) + x · σ'(x)
               = σ(x) + x · σ(x)(1 - σ(x))
               = σ(x)[1 + x(1 - σ(x))]
               = σ(x) + x·σ(x) - x·σ(x)²
               = Swish(x) + σ(x)(1 - Swish(x))

```python
# Verify with autograd:
x = torch.tensor([0.5], requires_grad=True)
swish = My_Swish()
y = swish(x)
y.backward()

# Manual calculation:
sigma = torch.sigmoid(x.detach())
manual_grad = sigma * (1 + x.detach() * (1 - sigma))

print(f"Autograd: {x.grad.item():.6f}")
print(f"Manual:   {manual_grad.item():.6f}")
```

*Key Insight*: Swish (also called SiLU) was discovered by neural architecture search (Ramachandran et al., 2017) and outperformed ReLU on many tasks. It's now default in EfficientNet and other architectures.

</details>

### Variation D2: Mish Activation

Build Mish: Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))

**Part 10.1:** Implement the module.

**Part 10.2:** Compare Mish vs Swish vs ReLU on the range [-5, 5].

<details>
<summary>Solution D2</summary>

**10.1:**

```python
class My_Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
        # Or equivalently:
        # return x * torch.tanh(torch.log(1 + torch.exp(x)))
```

**10.2:**

```python
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)

relu = torch.relu(x)
swish = x * torch.sigmoid(x)
mish = x * torch.tanh(torch.nn.functional.softplus(x))

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), relu.numpy(), label='ReLU')
plt.plot(x.numpy(), swish.numpy(), label='Swish')
plt.plot(x.numpy(), mish.numpy(), label='Mish')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.legend()
plt.title("ReLU vs Swish vs Mish")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.show()

# Key differences:
# - All three are similar for large positive x (≈ x)
# - ReLU is exactly 0 for x < 0
# - Swish and Mish have a small negative region
# - Mish has a slightly stronger negative dip than Swish
```

*Key Insight*: Mish was proposed by Diganta Misra (2019) and showed improvements on ImageNet. It's smoother than ReLU and allows small negative values, which may help with gradient flow.

</details>

### Variation D3: Hard Swish (for Mobile)

Hard Swish is a piecewise linear approximation: HardSwish(x) = x · HardSigmoid(x)

where HardSigmoid(x) = clamp(x/6 + 0.5, 0, 1)

**Part 10.1:** Implement HardSigmoid.

**Part 10.2:** Implement HardSwish using HardSigmoid.

**Part 10.3:** Why is Hard Swish preferred for mobile/edge deployment?

<details>
<summary>Solution D3</summary>

**10.1:**

```python
class My_HardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x / 6 + 0.5, 0, 1)
```

**10.2:**

```python
class My_HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.hard_sigmoid = My_HardSigmoid()

    def forward(self, x):
        return x * self.hard_sigmoid(x)

# Or more explicitly:
class My_HardSwish_v2(nn.Module):
    def forward(self, x):
        return x * torch.clamp(x / 6 + 0.5, 0, 1)
```

**10.3:** Hard Swish is preferred for mobile because:

1. **No exponentials**: Regular sigmoid requires `exp()`, which is expensive on CPUs
2. **Piecewise linear**: Only uses multiply, add, and clamp operations
3. **Similar accuracy**: Empirically, Hard Swish matches Swish accuracy on most tasks
4. **Quantization friendly**: Linear operations are easier to quantize to int8

```python
# Speed comparison (conceptual):
import time

x = torch.randn(1000, 1000)

# Warm up
_ = torch.sigmoid(x)
_ = torch.clamp(x / 6 + 0.5, 0, 1)

# Time sigmoid
start = time.time()
for _ in range(100):
    _ = x * torch.sigmoid(x)
swish_time = time.time() - start

# Time hard sigmoid
start = time.time()
for _ in range(100):
    _ = x * torch.clamp(x / 6 + 0.5, 0, 1)
hard_swish_time = time.time() - start

print(f"Swish: {swish_time:.4f}s")
print(f"Hard Swish: {hard_swish_time:.4f}s")
print(f"Speedup: {swish_time / hard_swish_time:.2f}x")
```

*Key Insight*: Hard Swish is used in MobileNetV3 (Howard et al., 2019) as a drop-in replacement for Swish to enable efficient mobile inference.

</details>

---

## CATEGORY E: Modules with Internal State

### Variation E1: Activation with Running Statistics

Build a module that tracks the running mean of activations (like BatchNorm but simpler).

**Part 10.1:** Use `register_buffer` for the running mean.

**Part 10.2:** Explain the difference between `nn.Parameter` and `register_buffer`.

<details>
<summary>Solution E1</summary>

**10.1:**

```python
class ReLUWithStats(nn.Module):
    def __init__(self, momentum=0.1):
        super().__init__()
        self.momentum = momentum
        # Buffer: saved in state_dict but NOT a learnable parameter
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # Update running statistics
            with torch.no_grad():
                batch_mean = x.mean()
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.num_batches_tracked += 1

        return torch.relu(x)

# Usage:
layer = ReLUWithStats()

# Training mode
layer.train()
for _ in range(10):
    x = torch.randn(32, 64)
    _ = layer(x)

print(f"Running mean: {layer.running_mean.item():.4f}")
print(f"Batches tracked: {layer.num_batches_tracked.item()}")

# Buffers are in state_dict but not parameters()
print("State dict:", layer.state_dict())
print("Parameters:", list(layer.parameters()))  # Empty!
```

**10.2:** Difference between `nn.Parameter` and `register_buffer`:

| Aspect | nn.Parameter | register_buffer |
|--------|--------------|-----------------|
| Returned by `parameters()` | Yes | No |
| Updated by optimizer | Yes | No (manual updates) |
| Saved in `state_dict` | Yes | Yes |
| Moved by `.to(device)` | Yes | Yes |
| Requires grad | True (default) | False (default) |

Use **Parameter** for: Weights, biases, learnable scales
Use **Buffer** for: Running statistics, cached values, non-learned state

</details>

### Variation E2: Activation with Dropout-like Noise

Build a module that adds noise during training but not during evaluation.

**Part 10.1:** Implement an activation that adds Gaussian noise (scaled by a factor) during training.

**Part 10.2:** Why must you check `self.training`?

<details>
<summary>Solution E2</summary>

**10.1:**

```python
class NoisyReLU(nn.Module):
    def __init__(self, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        out = torch.relu(x)

        if self.training:
            noise = torch.randn_like(out) * self.noise_std
            out = out + noise

        return out

# Usage:
layer = NoisyReLU(noise_std=0.1)

x = torch.ones(5)

layer.train()
print("Training outputs (varies each time):")
for _ in range(3):
    print(layer(x))

layer.eval()
print("\nEval outputs (consistent):")
for _ in range(3):
    print(layer(x))
```

**10.2:** Why check `self.training`:

1. **Deterministic evaluation**: During inference, we want consistent predictions. Random noise would make outputs non-reproducible.

2. **Expected value**: Many noise-based regularizations (Dropout, Gaussian noise) are designed so training expectation ≈ eval output.

3. **Deployment**: Production systems need deterministic behavior for debugging and reproducibility.

```python
# The training flag is set by model.train() and model.eval():
model = NoisyReLU()
print(model.training)  # True (default)

model.eval()
print(model.training)  # False

model.train()
print(model.training)  # True
```

*Key Insight*: Always use `if self.training:` for any stochastic behavior. This is how Dropout, BatchNorm, and all standard layers handle train/eval differences.

</details>

---

## CATEGORY F: Module Composition Patterns

### Variation F1: Sequential Activation Block

Build a module that combines ReLU + BatchNorm (but you implement the components).

**Part 10.1:** Implement a simple BatchNorm1d from scratch (without running stats for simplicity).

**Part 10.2:** Combine into a single module.

<details>
<summary>Solution F1</summary>

**10.1:**

```python
class SimpleBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x shape: (batch, features)
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

**10.2:**

```python
class ReLUBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.relu = My_ReLU()  # Our custom ReLU
        self.bn = SimpleBatchNorm1d(num_features)

    def forward(self, x):
        x = self.relu(x)
        x = self.bn(x)
        return x

# Or using nn.Sequential:
def make_relu_bn_block(num_features):
    return nn.Sequential(
        My_ReLU(),
        SimpleBatchNorm1d(num_features)
    )

# Test:
block = ReLUBatchNorm(64)
x = torch.randn(32, 64)
out = block(x)
print(out.shape)  # torch.Size([32, 64])
print(list(block.parameters()))  # gamma and beta from BatchNorm
```

</details>

### Variation F2: Activation with Skip Connection

Build a module that applies activation with a residual connection: f(x) = ReLU(x) + x

**Part 10.1:** Implement the module.

**Part 10.2:** Discuss why this might NOT be a good idea (unlike residual connections in ResNets).

<details>
<summary>Solution F2</summary>

**10.1:**

```python
class ResidualReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x) + x

# Or with a learnable blend factor:
class BlendedResidualReLU(nn.Module):
    def __init__(self, init_blend=0.5):
        super().__init__()
        self.blend = nn.Parameter(torch.tensor(init_blend))

    def forward(self, x):
        return self.blend * torch.relu(x) + (1 - self.blend) * x
```

**10.2:** Why this might not be ideal:

1. **Changes the output distribution**: For x > 0: output = 2x (doubles positive values). For x < 0: output = x (unchanged). This is not what we typically want.

2. **No non-linearity for negative inputs**: The whole point of ReLU is to introduce non-linearity by zeroing negatives. Adding back x removes this.

3. **Not the same as ResNet residuals**: In ResNets, the skip connection bypasses a BLOCK of layers (Conv-BN-ReLU-Conv-BN), not just an activation. The residual learning helps with:
   - Gradient flow through many layers
   - Learning the identity function if needed
   - Not meant to modify a single activation's behavior

```python
# ResNet-style residual (correct usage):
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # Save for skip

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity  # Skip connection
        out = torch.relu(out)

        return out
```

*Key Insight*: Residual connections are powerful when they skip multiple operations, not single activations. They help with the vanishing gradient problem in very deep networks.

</details>

---

## CATEGORY G: PyTorch Module Best Practices

### Variation G1: Proper Initialization Patterns

Show the correct way to initialize a custom module with proper weight initialization.

**Part 10.1:** Build a linear layer with custom initialization inside a module.

**Part 10.2:** What is `reset_parameters()` and when should you use it?

<details>
<summary>Solution G1</summary>

**10.1:**

```python
class MyLinearWithInit(nn.Module):
    def __init__(self, in_features, out_features, init_method='xavier'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.init_method = init_method
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.init_method == 'kaiming':
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif self.init_method == 'orthogonal':
            nn.init.orthogonal_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

# Usage:
layer_xavier = MyLinearWithInit(64, 32, init_method='xavier')
layer_kaiming = MyLinearWithInit(64, 32, init_method='kaiming')
```

**10.2:** `reset_parameters()` convention:

- **What**: A method that (re)initializes all learnable parameters in a module
- **When called**: Automatically during `__init__`, or manually to reset
- **Standard practice**: PyTorch's built-in modules all have `reset_parameters()`

```python
# PyTorch's Linear.reset_parameters() does this:
def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

# You can reinitialize anytime:
layer = nn.Linear(10, 5)
print("Before:", layer.weight[0, :3])
layer.reset_parameters()
print("After:", layer.weight[0, :3])  # Different values!
```

*Key Insight*: Always initialize weights in `reset_parameters()`, not directly in `__init__`. This follows PyTorch conventions and allows users to reinitialize if needed.

</details>

### Variation G2: Module with Multiple Forward Modes

Build a module that behaves differently based on an argument to forward().

**Part 10.1:** Implement a module with both "full" and "fast" forward modes.

**Part 10.2:** How do you handle extra forward arguments correctly?

<details>
<summary>Solution G2</summary>

**10.1:**

```python
class FlexibleActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mode='full'):
        if mode == 'full':
            # Full GELU computation
            return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        elif mode == 'fast':
            # Fast approximation
            return x * torch.sigmoid(1.702 * x)
        elif mode == 'relu':
            # Fallback to simple ReLU
            return torch.relu(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

# Usage:
act = FlexibleActivation()
x = torch.randn(10)

print("Full GELU:", act(x, mode='full'))
print("Fast approx:", act(x, mode='fast'))
print("ReLU:", act(x, mode='relu'))
```

**10.2:** Handling extra forward arguments:

```python
# Problem: nn.Sequential can't pass extra arguments to forward()
seq = nn.Sequential(
    nn.Linear(10, 10),
    FlexibleActivation()  # Can't pass mode='fast' here!
)

# Solution 1: Use a wrapper module
class FastActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = FlexibleActivation()

    def forward(self, x):
        return self.act(x, mode='fast')

# Solution 2: Set mode as an attribute
class ConfigurableActivation(nn.Module):
    def __init__(self, default_mode='full'):
        super().__init__()
        self.mode = default_mode

    def forward(self, x):
        if self.mode == 'full':
            return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        elif self.mode == 'fast':
            return x * torch.sigmoid(1.702 * x)
        else:
            return torch.relu(x)

# Usage:
act = ConfigurableActivation(default_mode='full')
act.mode = 'fast'  # Change mode dynamically
```

*Key Insight*: If using `nn.Sequential`, all layers must have `forward(self, x)` signature. For flexibility, either use wrapper modules or set modes as attributes.

</details>

### Variation G3: Module String Representation

Build a module with a custom `extra_repr()` for informative printing.

**Part 10.1:** Implement `extra_repr()` for a configurable activation.

**Part 10.2:** What information should `extra_repr()` include?

<details>
<summary>Solution G3</summary>

**10.1:**

```python
class ConfigurableLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.leaky_relu(x, self.negative_slope, self.inplace)

    def extra_repr(self):
        return f'negative_slope={self.negative_slope}, inplace={self.inplace}'

# Now when printed:
layer = ConfigurableLeakyReLU(negative_slope=0.2, inplace=True)
print(layer)
# Output: ConfigurableLeakyReLU(negative_slope=0.2, inplace=True)
```

**10.2:** What to include in `extra_repr()`:

- **Configuration parameters**: negative_slope, alpha values, etc.
- **Shape information**: in_features, out_features, kernel_size
- **Mode flags**: inplace, training-specific settings
- **NOT**: Actual tensor values, computed statistics

```python
# Examples from PyTorch source:

# nn.Linear:
def extra_repr(self):
    return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

# nn.Conv2d:
def extra_repr(self):
    s = f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}'
    if self.padding != (0,) * len(self.padding):
        s += f', padding={self.padding}'
    if self.dilation != (1,) * len(self.dilation):
        s += f', dilation={self.dilation}'
    if self.groups != 1:
        s += f', groups={self.groups}'
    if self.bias is None:
        s += ', bias=False'
    return s

# nn.Dropout:
def extra_repr(self):
    return f'p={self.p}, inplace={self.inplace}'
```

*Key Insight*: Good `extra_repr()` makes debugging much easier. When you print a model, you can see all the configuration at a glance.

</details>

---

## CATEGORY H: Edge Cases and Special Behaviors

### Variation H1: Handling Non-Standard Input Shapes

Build a ReLU that works correctly regardless of input dimensions.

**Part 10.1:** Verify ReLU works on 1D, 2D, 3D, and 4D tensors.

**Part 10.2:** Build an activation that requires a minimum dimension count.

<details>
<summary>Solution H1</summary>

**10.1:**

```python
class My_ReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)  # Works for any shape!

relu = My_ReLU()

# Works on any dimension
x1d = torch.randn(10)
x2d = torch.randn(5, 10)
x3d = torch.randn(2, 5, 10)
x4d = torch.randn(2, 3, 5, 10)
x5d = torch.randn(2, 3, 4, 5, 10)

print("1D:", relu(x1d).shape)  # torch.Size([10])
print("2D:", relu(x2d).shape)  # torch.Size([5, 10])
print("3D:", relu(x3d).shape)  # torch.Size([2, 5, 10])
print("4D:", relu(x4d).shape)  # torch.Size([2, 3, 5, 10])
print("5D:", relu(x5d).shape)  # torch.Size([2, 3, 4, 5, 10])
```

**10.2:**

```python
class ChannelwiseReLU(nn.Module):
    """ReLU with per-channel threshold (requires at least 2D input)."""

    def __init__(self, num_channels):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        if x.dim() < 2:
            raise ValueError(f"Expected at least 2D input, got {x.dim()}D")

        # Reshape threshold for broadcasting
        # Input could be (B, C), (B, C, L), (B, C, H, W), etc.
        shape = [1, -1] + [1] * (x.dim() - 2)  # e.g., [1, C, 1, 1] for 4D
        threshold = self.threshold.view(*shape)

        return torch.relu(x - threshold)

# Usage:
layer = ChannelwiseReLU(64)

x2d = torch.randn(8, 64)
x4d = torch.randn(8, 64, 32, 32)

print(layer(x2d).shape)  # torch.Size([8, 64])
print(layer(x4d).shape)  # torch.Size([8, 64, 32, 32])

# This would raise an error:
# x1d = torch.randn(64)
# layer(x1d)  # ValueError!
```

</details>

### Variation H2: Inplace Operations

Build an activation with an optional inplace mode.

**Part 10.1:** Implement inplace ReLU.

**Part 10.2:** When should you use inplace operations? When should you avoid them?

<details>
<summary>Solution H2</summary>

**10.1:**

```python
class My_ReLU_Inplace(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.relu_()  # Inplace operation (note the underscore!)
        else:
            return torch.relu(x)  # Creates new tensor

# Compare:
x = torch.tensor([-1.0, 0.0, 1.0])

# Without inplace:
x_copy = x.clone()
relu_normal = My_ReLU_Inplace(inplace=False)
y = relu_normal(x_copy)
print("Original after normal ReLU:", x_copy)  # tensor([-1.,  0.,  1.]) - unchanged

# With inplace:
x_copy = x.clone()
relu_inplace = My_ReLU_Inplace(inplace=True)
y = relu_inplace(x_copy)
print("Original after inplace ReLU:", x_copy)  # tensor([0., 0., 1.]) - modified!
```

**10.2:** When to use inplace:

**Use inplace when:**
- Memory is tight (e.g., on GPU with large batch)
- The input tensor won't be used again
- You're certain it won't break gradient computation

**Avoid inplace when:**
- The input is needed for backward pass (skip connections!)
- You're debugging (harder to trace)
- The input has `requires_grad=True` and is part of the computation graph

```python
# Danger: Inplace operations can break autograd!

x = torch.randn(3, requires_grad=True)
y = x * 2  # y depends on x for gradient

# This is FINE:
z = torch.relu(y)

# This BREAKS the gradient:
# y.relu_()  # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

# The error happens because y is needed to compute gradient w.r.t. x,
# but inplace operation modified it before backward() was called.
```

*Key Insight*: Use inplace operations cautiously. The memory savings are usually small, and the bugs can be hard to track down.

</details>

### Variation H3: Device and Dtype Handling

Build an activation that properly handles different devices and data types.

**Part 10.1:** Show how to build a module that works on both CPU and CUDA.

**Part 10.2:** Build a module with a constant that must be on the same device as input.

<details>
<summary>Solution H3</summary>

**10.1:** Standard modules work automatically:

```python
class My_ReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)

relu = My_ReLU()

# Works on CPU
x_cpu = torch.randn(10)
print(relu(x_cpu).device)  # cpu

# Works on CUDA (if available)
if torch.cuda.is_available():
    x_cuda = torch.randn(10, device='cuda')
    print(relu(x_cuda).device)  # cuda:0
```

**10.2:** Module with constant:

```python
# WRONG: Constant on wrong device
class BadSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 20.0  # Just a Python float - no device info
        self.one = torch.tensor(1.0)  # Tensor, but not registered!

    def forward(self, x):
        # self.one might be on CPU while x is on CUDA!
        return torch.where(x > self.threshold, x, torch.log(self.one + torch.exp(x)))

# CORRECT: Use register_buffer for device tracking
class GoodSoftplus(nn.Module):
    def __init__(self, threshold=20.0):
        super().__init__()
        self.threshold = threshold  # Python float is fine (no device needed)
        self.register_buffer('one', torch.tensor(1.0))  # Will move with module

    def forward(self, x):
        return torch.where(x > self.threshold, x, torch.log(self.one + torch.exp(x)))

# Test device transfer:
layer = GoodSoftplus()
print("one device before:", layer.one.device)  # cpu

if torch.cuda.is_available():
    layer = layer.cuda()
    print("one device after:", layer.one.device)  # cuda:0

    x = torch.randn(10, device='cuda')
    y = layer(x)  # Works!
```

*Key Insight*: Use `register_buffer` for any tensor constants that need to move with the model. Plain tensor attributes won't be moved by `.to()`, `.cuda()`, or `.cpu()`.

</details>

---

## KEY CONCEPTS SUMMARY

### nn.Module Structure

```python
class MyModule(nn.Module):
    def __init__(self, ...):
        super().__init__()  # Required!
        self.param = nn.Parameter(...)        # Learnable
        self.register_buffer('buf', ...)      # Saved, not learned
        self.submodule = SomeOtherModule()    # Nested modules

    def forward(self, x):
        # Define computation
        return result

    def extra_repr(self):
        return 'info=...'  # For nice printing
```

### Parameter vs Buffer vs Attribute

| Type | Included in `parameters()` | Saved in `state_dict` | Moved by `.to()` |
|------|----------------------------|----------------------|------------------|
| `nn.Parameter` | ✓ | ✓ | ✓ |
| `register_buffer` | ✗ | ✓ | ✓ |
| Plain tensor | ✗ | ✗ | ✗ |
| Python float/int | N/A | ✗ | N/A |

### Common Activation Derivatives

| Activation | Formula | Derivative |
|------------|---------|------------|
| ReLU | max(0, x) | 1 if x > 0, else 0 |
| Sigmoid | 1/(1+e^(-x)) | σ(x)(1-σ(x)) |
| Tanh | (e^x - e^(-x))/(e^x + e^(-x)) | 1 - tanh²(x) |
| Leaky ReLU | max(αx, x) | 1 if x > 0, else α |
| ELU | x if x > 0, α(e^x - 1) else | 1 if x > 0, αe^x else |
| Swish | x · σ(x) | σ(x)(1 + x(1 - σ(x))) |

---

## CATEGORY I: Speed Round (Rapid-Fire Conceptual Questions)

*These questions test quick recall and conceptual understanding—ideal for USAAIO multiple-choice and short-answer sections.*

### Variation I1: Parameter vs Buffer

**Question:** In a PyTorch module, you have a tensor that stores the running mean of activations (updated during training, but not learned via gradient descent). Should this be:

A) `self.running_mean = torch.zeros(10)`
B) `self.running_mean = nn.Parameter(torch.zeros(10))`
C) `self.register_buffer('running_mean', torch.zeros(10))`
D) `self.register_parameter('running_mean', torch.zeros(10))`

<details>
<summary>Solution I1</summary>

**Answer: C**

- **A is wrong**: Plain tensor attributes are NOT saved in `state_dict` and NOT moved by `.to(device)`
- **B is wrong**: `nn.Parameter` makes it learnable (included in `parameters()`, updated by optimizer)
- **C is correct**: `register_buffer` saves to `state_dict`, moves with device, but is NOT a learnable parameter
- **D is wrong**: `register_parameter` is used for Parameters, not buffers

*Key distinction*: Parameters are optimized; buffers are saved but not optimized.

</details>

---

### Variation I2: Gradient Flow

**Question:** What will `x.grad` be after running this code?

```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.relu(x)
z = y ** 2
z.backward()
```

A) `[0.0]`
B) `[2.0]`
C) `[4.0]`
D) `[8.0]`

<details>
<summary>Solution I2</summary>

**Answer: C) `[4.0]`**

Chain rule:
- z = y² → ∂z/∂y = 2y = 2(2) = 4
- y = relu(x) = x (since x=2 > 0) → ∂y/∂x = 1
- ∂z/∂x = ∂z/∂y · ∂y/∂x = 4 · 1 = **4.0**

</details>

---

### Variation I3: Training vs Eval Mode

**Question:** Which of the following behaviors change between `model.train()` and `model.eval()`?

A) `torch.relu(x)` outputs different values
B) `nn.Dropout(0.5)` applies dropout
C) `nn.Linear(10, 5)` uses different weights
D) Gradients are not computed in eval mode

<details>
<summary>Solution I3</summary>

**Answer: B**

- **A is wrong**: ReLU is deterministic—same output in train/eval
- **B is correct**: Dropout only applies during `self.training=True`; in eval mode it's identity
- **C is wrong**: Linear uses same weights always; only the mode flag changes
- **D is wrong**: `model.eval()` doesn't disable gradients! Use `torch.no_grad()` for that

*Common misconception*: Many confuse `model.eval()` with `torch.no_grad()`. They're independent!

</details>

---

### Variation I4: Module Call vs Forward

**Question:** Why should you call `output = model(x)` instead of `output = model.forward(x)`?

A) `model(x)` is faster
B) `model(x)` runs hooks (pre-forward, post-forward)
C) `model.forward(x)` doesn't compute gradients
D) There's no difference; they're exactly equivalent

<details>
<summary>Solution I4</summary>

**Answer: B**

When you call `model(x)`, PyTorch internally calls `__call__`, which:
1. Runs all registered **forward pre-hooks**
2. Calls `forward(x)`
3. Runs all registered **forward hooks**

Calling `model.forward(x)` bypasses hooks entirely.

```python
# Example: Hooks won't fire!
model = nn.Linear(10, 5)
model.register_forward_hook(lambda m, i, o: print("Hook fired!"))
model.forward(torch.randn(10))  # No output!
model(torch.randn(10))          # Prints "Hook fired!"
```

</details>

---

### Variation I5: Super Init

**Question:** What happens if you forget `super().__init__()` in a custom module?

```python
class BadModule(nn.Module):
    def __init__(self):
        # Missing: super().__init__()
        self.weight = nn.Parameter(torch.randn(10))

    def forward(self, x):
        return x @ self.weight
```

A) `RuntimeError` when creating the module
B) Parameters won't be registered; `list(model.parameters())` returns empty list
C) The forward pass will fail with dimension mismatch
D) Everything works fine

<details>
<summary>Solution I5</summary>

**Answer: B**

Without `super().__init__()`:
- The internal parameter/buffer tracking dicts aren't created
- `self.weight = nn.Parameter(...)` becomes a plain attribute
- `list(model.parameters())` returns `[]`
- `model.state_dict()` is empty
- Optimizer receives no parameters to update!

The code *runs* but doesn't *learn*. This is a silent failure—one of the most dangerous bugs.

</details>

---

### Variation I6: Inplace Operations

**Question:** Which of these will raise a `RuntimeError` during backward pass?

```python
# Option A:
x = torch.randn(5, requires_grad=True)
y = x.relu()
y.sum().backward()

# Option B:
x = torch.randn(5, requires_grad=True)
y = x * 2
y.relu_()  # inplace!
y.sum().backward()

# Option C:
x = torch.randn(5, requires_grad=True)
y = x.relu_()
y.sum().backward()
```

A) Option A only
B) Option B only
C) Option C only
D) Options B and C

<details>
<summary>Solution I6</summary>

**Answer: D) Options B and C**

- **Option A**: No inplace operation—works fine
- **Option B**: `y = x * 2` creates a view, then `y.relu_()` modifies it inplace. Autograd needs the original `y` values for the backward of `x * 2`. Fails!
- **Option C**: `x.relu_()` modifies `x` inplace. `x` is a leaf variable with `requires_grad=True`. Modifying leaves inplace is forbidden!

*Rule*: Never do inplace operations on tensors that are part of the computation graph.

</details>

---

### Variation I7: Device Movement

**Question:** After running this code, what device is `layer.scale` on?

```python
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.tensor(2.0)  # Plain tensor, NOT registered!

layer = MyLayer()
layer = layer.cuda()
```

A) CUDA
B) CPU
C) `RuntimeError`—can't move module with unregistered tensors
D) Depends on default device

<details>
<summary>Solution I7</summary>

**Answer: B) CPU**

Plain tensor attributes are NOT moved by `.cuda()`, `.to()`, or `.cpu()`. Only:
- `nn.Parameter` (learnable)
- Registered buffers (`register_buffer`)

...are tracked and moved with the module.

This is a common source of "device mismatch" errors:
```python
x = torch.randn(5, device='cuda')
layer(x)  # RuntimeError: expected all tensors on same device!
```

*Fix*: Use `self.register_buffer('scale', torch.tensor(2.0))`

</details>

---

### Variation I8: Autograd Function

**Question:** In a custom `torch.autograd.Function`, the `backward` method receives:

A) The input tensor from forward
B) The output tensor from forward
C) The gradient of the loss with respect to the output
D) The gradient of the loss with respect to the input

<details>
<summary>Solution I8</summary>

**Answer: C**

The `backward` method receives `grad_output`—the gradient of the loss with respect to the **output** of `forward`.

Your job is to compute and return the gradient with respect to the **input**.

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2  # output = x²

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = ∂L/∂output
        x, = ctx.saved_tensors
        # Return ∂L/∂x = ∂L/∂output · ∂output/∂x = grad_output · 2x
        return grad_output * 2 * x
```

*Chain rule*: You're handed ∂L/∂y, you return ∂L/∂x = ∂L/∂y · ∂y/∂x

</details>

---

### Variation I9: Output Shape

**Question:** What is `model(x).shape`?

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc(x)

model = MyModule()
x = torch.randn(8, 16, 64)  # (batch, seq, features)
```

A) `torch.Size([8, 32])`
B) `torch.Size([8, 16, 32])`
C) `torch.Size([8, 512])`
D) `RuntimeError`—dimension mismatch

<details>
<summary>Solution I9</summary>

**Answer: B) `torch.Size([8, 16, 32])`**

`nn.Linear` operates on the **last dimension** only. For input shape `(8, 16, 64)`:
- 64 → 32 transformation applied to last dim
- Other dims (8, 16) are preserved
- Output: `(8, 16, 32)`

This is why `nn.Linear` works seamlessly in Transformers: it processes `(batch, seq_len, d_model)` by transforming `d_model` at each position independently.

</details>

---

### Variation I10: State Dict Keys

**Question:** What are the keys in `model.state_dict()`?

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

model = MyModel()
print(list(model.state_dict().keys()))
```

A) `['layer1', 'layer2']`
B) `['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias']`
C) `['weight', 'bias', 'weight', 'bias']`
D) `['MyModel.layer1.weight', ...]`

<details>
<summary>Solution I10</summary>

**Answer: B)**

```python
['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias']
```

State dict keys use **dot notation** to show the hierarchy:
- `layer1` is the attribute name in `MyModel`
- `.weight` and `.bias` are from `nn.Linear`

This format allows:
- Saving/loading weights correctly
- Partial loading (freeze some layers)
- Inspecting model structure from checkpoint

*No* class name prefix—keys are relative to the module being saved.

</details>

---

## ATOMIC SKILLS CHECKLIST

- [ ] Subclass `torch.nn.Module` correctly (call `super().__init__()`)
- [ ] Implement `forward()` method
- [ ] Use `nn.Parameter` for learnable weights
- [ ] Use `register_buffer` for non-learned but saved tensors
- [ ] Implement `torch.autograd.Function` for custom backward
- [ ] Use `gradcheck` to verify gradients
- [ ] Handle `self.training` flag for train/eval differences
- [ ] Implement `extra_repr()` for informative printing
- [ ] Handle inplace operations safely
- [ ] Ensure device compatibility with buffers

---

## COMMON MISCONCEPTIONS

1. **Forgetting `super().__init__()`**: This breaks parameter registration and causes silent failures.

2. **Using plain tensors instead of Parameters**: They won't be optimized or saved.

3. **Not registering buffers**: Tensors not moved with `.to(device)` cause device mismatch errors.

4. **Inplace operations breaking gradients**: Modifying tensors needed for backward causes errors.

5. **Assuming `forward()` is called directly**: Always call the module as `module(x)`, not `module.forward(x)`, to ensure hooks work.

6. **Putting initialization in `forward()`**: Parameters should be created in `__init__`, not `forward()`.

7. **Confusing `self.training` with `torch.no_grad()`**: `self.training` is about module mode; `torch.no_grad()` is about gradient tracking.
