# Activation Functions

**Prerequisites**: Derivatives (calculus), forward/backward propagation (Study Guides 02–03)
**USAAIO Relevance**: Round 1 tests derivative computation for each activation. Understanding vanishing/dying gradients is essential for architecture analysis. Round 2 may require custom activation implementations.

---

## Discovery

### Why Nonlinearity?

Without activation functions, a neural network collapses to a single linear transformation no matter how many layers it has:

$$W_3(W_2(W_1 x + b_1) + b_2) + b_3 = W'x + b'$$

A composition of linear functions is still linear. To learn nonlinear decision boundaries, complex feature interactions, and curved regression surfaces, we need nonlinear activation functions between layers.

> **Socratic question**: If we use $\sigma(x) = 2x$ (a linear "activation"), what happens to a 100-layer network?
>
> *Answer: Each layer multiplies by 2, so the output is $2^{100}$ times the original linear transformation — an astronomically scaled linear function. No matter the depth, the function class remains linear. Nonlinearity is what makes depth useful.*

### The Landscape of Activations

Different activations have different trade-offs:
- **Sigmoid** (1990s): Smooth, bounded, differentiable everywhere — but kills gradients for extreme inputs
- **ReLU** (2010s): Simple, fast, no vanishing gradient for positive inputs — but neurons can "die"
- **GELU** (2020s): Smooth approximation of ReLU used in transformers — probabilistic gating

---

## Intuition

### Visual Comparison

```
Sigmoid: σ(x) = 1/(1+e^-x)        ReLU: max(0, x)
    1 ─────────────╱──────              ╱
                 ╱                    ╱
   0.5 ───────╱──────────           ╱
            ╱                     ╱
    0 ────╱───────────────   ────╱──────────────
       -5    0    5              0

Tanh: (e^x - e^-x)/(e^x + e^-x)   GELU: x·Φ(x)
    1 ──────────────╱──────              ╱
                  ╱                    ╱
    0 ──────────╱────────────   ──────╱─────────
              ╱                    ╱
   -1 ──────╱────────────────   ──╱─────────────
          -5    0    5           -3   0    3
```

### The Vanishing Gradient Problem

Sigmoid's derivative has a maximum of 0.25 (at $x=0$). In a 10-layer network, the gradient at the first layer is multiplied by sigmoid derivatives at every layer:

$$\left(\frac{1}{4}\right)^{10} = \frac{1}{1{,}048{,}576} \approx 10^{-6}$$

The gradient effectively disappears — early layers cannot learn. This is why sigmoid fell out of favor for hidden layers.

### The Dying ReLU Problem

ReLU's derivative is 0 for $x \leq 0$. If a neuron's pre-activation becomes permanently negative (e.g., due to a large negative bias or a bad gradient update), its gradient is forever 0. The neuron is "dead" — it never activates and never learns again.

Solutions:
- **Leaky ReLU**: Small positive slope for $x < 0$ ($\alpha = 0.01$)
- **Parametric ReLU (PReLU)**: Learnable slope for $x < 0$
- **Careful initialization**: Xavier/He initialization keeps pre-activations centered

---

## Math

### Activation Functions and Their Derivatives

**Sigmoid**:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

Max derivative: $\sigma'(0) = 0.25$. Output range: $(0, 1)$.

**Tanh**:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

Max derivative: $\tanh'(0) = 1$. Output range: $(-1, 1)$. Better than sigmoid (zero-centered), but still vanishes for large $|x|$.

**ReLU**:

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

Convention: $\text{ReLU}'(0) = 0$ (or sometimes 1). No vanishing gradient for $x > 0$.

**Leaky ReLU**:

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$

Typically $\alpha = 0.01$. Never zero gradient, so no dying neurons.

**GELU** (Gaussian Error Linear Unit):

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the CDF of the standard normal distribution.

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ is the PDF. Used in GPT, BERT, and most modern transformers.

**Swish** (SiLU):

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

$$\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \sigma(x)(1 + x(1 - \sigma(x)))$$

Smooth, non-monotonic (slightly negative for $x \approx -1.28$). Used in EfficientNet, many vision models.

**Softmax** (for output layer only):

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

$$\frac{\partial \text{softmax}(x)_i}{\partial x_j} = \text{softmax}(x)_i \cdot (\delta_{ij} - \text{softmax}(x)_j)$$

where $\delta_{ij}$ is the Kronecker delta. Outputs a probability distribution (sums to 1).

### When to Use Which

| Situation | Recommended Activation |
|---|---|
| Hidden layers (default) | ReLU |
| Hidden layers (if dying ReLU is a problem) | Leaky ReLU or GELU |
| Binary classification output | Sigmoid |
| Multi-class classification output | Softmax (but often handled by loss function) |
| Transformer hidden layers | GELU |
| Regression output | None (identity) |
| Bounded output needed | Sigmoid (0,1) or Tanh (-1,1) |

---

## Code

### Implementing Activations from Scratch

```python
import torch
import torch.nn.functional as F

x = torch.linspace(-5, 5, 100)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Tanh
def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def tanh_deriv(x):
    return 1 - tanh(x) ** 2

# ReLU
def relu(x):
    return torch.clamp(x, min=0)

def relu_deriv(x):
    return (x > 0).float()

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, alpha))

# GELU (exact)
def gelu(x):
    return x * 0.5 * (1 + torch.erf(x / (2 ** 0.5)))

# GELU (approximate, as used in practice)
def gelu_approx(x):
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

# Swish / SiLU
def swish(x):
    return x * sigmoid(x)
```

### Verifying Derivatives with Autograd

```python
import torch

x = torch.tensor([−2.0, −1.0, 0.0, 1.0, 2.0], requires_grad=True)

# Sigmoid derivative verification
y = torch.sigmoid(x)
y.sum().backward()
manual_deriv = torch.sigmoid(x.detach()) * (1 - torch.sigmoid(x.detach()))
assert torch.allclose(x.grad, manual_deriv, atol=1e-6)
print("Sigmoid derivative verified!")

# ReLU derivative verification
x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y2 = torch.relu(x2)
y2.sum().backward()
manual_relu = (x2.detach() > 0).float()
assert torch.allclose(x2.grad, manual_relu, atol=1e-6)
print("ReLU derivative verified!")
```

### Demonstrating the Vanishing Gradient Problem

```python
import torch
import torch.nn as nn

# Deep network with sigmoid activations
class DeepSigmoid(nn.Module):
    def __init__(self, depth=10, width=64):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = DeepSigmoid(depth=10, width=64)
x = torch.randn(1, 64)
y = model(x).sum()
y.backward()

# Check gradient magnitudes at different layers
for i, (name, param) in enumerate(model.named_parameters()):
    if 'weight' in name:
        print(f"Layer {i//2}: gradient norm = {param.grad.norm():.6f}")
# You will see gradient norms decrease dramatically for earlier layers
```
