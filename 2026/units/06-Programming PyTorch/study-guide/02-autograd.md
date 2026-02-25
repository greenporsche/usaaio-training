# Autograd

**Prerequisites**: Calculus (derivatives, chain rule), tensors (Unit 06.01)
**USAAIO Relevance**: **CRITICAL**. USAAIO 2025 Round 2 Problem 1 (PINNs) directly tested `torch.autograd.grad` with `create_graph=True` for computing higher-order partial derivatives. You must be able to use autograd fluently under time pressure.

---

## Discovery

### How Machines Compute Derivatives Automatically

When you train a neural network, you need the gradient of the loss with respect to every parameter. For a model with millions of parameters, computing these derivatives by hand is impossible. So how does PyTorch do it?

The answer is **automatic differentiation** (autodiff), specifically **reverse-mode autodiff** (also called backpropagation). The key insight:

1. Every PyTorch operation on tensors with `requires_grad=True` is **recorded** in a directed acyclic graph (DAG) — the computation graph
2. When you call `.backward()`, PyTorch walks the graph **in reverse**, applying the chain rule at each node
3. The gradients accumulate in the `.grad` attribute of each leaf tensor

> **Socratic question**: Why reverse mode and not forward mode? Consider a function $f: \mathbb{R}^n \to \mathbb{R}$ (n parameters, 1 scalar loss). Forward mode computes one column of the Jacobian per pass (n passes needed). Reverse mode computes one ROW — and since the output is scalar, one pass gives ALL gradients. For neural networks (many inputs, one loss), reverse mode is optimal.

### Three Flavors of Differentiation

| Method | How | Pros | Cons |
|---|---|---|---|
| **Symbolic** | Manipulate math expressions (like Wolfram Alpha) | Exact | Expression swell, can't handle loops |
| **Numerical** | Finite differences: $\frac{f(x+\epsilon) - f(x)}{\epsilon}$ | Simple | Slow ($O(n)$ evaluations), numerical errors |
| **Automatic** | Record operations, apply chain rule mechanically | Exact, efficient | Requires framework support |

PyTorch uses automatic differentiation. It is NOT symbolic (it does not simplify expressions) and NOT numerical (no finite differences).

---

## Intuition

### The Computation Graph

Every operation creates a node in the graph. Consider $y = x^2 + 3x$:

```
   x (leaf, requires_grad=True)
   ├──→ [pow] ──→ x²
   │                ├──→ [add] ──→ y
   └──→ [mul 3] ──→ 3x ─┘

Backward pass (chain rule, right to left):
  dy/dy = 1
  dy/d(x²) = 1,  dy/d(3x) = 1
  d(x²)/dx = 2x, d(3x)/dx = 3
  dy/dx = 2x + 3
```

For x = 2: dy/dx = 2(2) + 3 = 7.

### Gradient Accumulation

Gradients **accumulate** by default. If you call `.backward()` twice without zeroing:

```
First backward:  x.grad = 7
Second backward: x.grad = 14  (7 + 7, accumulated!)
```

This is why `optimizer.zero_grad()` is essential before each training step.

### Detaching from the Graph

Sometimes you want to use a tensor's value without tracking gradients:

```
┌────────────────────────────┐
│  x ──→ y = f(x)           │  y has grad_fn (tracked)
│         ↓                  │
│  z = y.detach()            │  z has same values, NO grad_fn
│  z.requires_grad = False   │  Operations on z are NOT tracked
└────────────────────────────┘
```

Use cases:
- Feature extraction (use pretrained model's output as fixed input)
- Target networks in reinforcement learning
- Computing metrics without wasting memory on graphs

### Higher-Order Derivatives: create_graph=True

Standard `.backward()` destroys the computation graph after use. To compute second (or higher) derivatives, you need to keep the graph alive:

```
           x
           ↓
     y = f(x)
           ↓    (first derivative, create_graph=True → graph preserved)
    dy/dx = g(x)
           ↓    (second derivative)
  d²y/dx² = g'(x)
```

This is exactly what PINNs need: if $u(x,t)$ is a neural network, computing $\frac{\partial^2 u}{\partial x^2}$ requires two levels of differentiation through the computation graph.

---

## Math

### Reverse-Mode Autodiff (Backpropagation)

Given a computation $y = f(g(h(x)))$, the chain rule gives:

$$\frac{dy}{dx} = \frac{dy}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

Reverse mode computes this right-to-left, accumulating the product:

$$\bar{x} = \bar{y} \cdot \frac{\partial y}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

where $\bar{v} = \frac{\partial L}{\partial v}$ is the "adjoint" (gradient of the loss with respect to variable $v$).

### torch.autograd.grad

For more control than `.backward()`, use the functional interface:

$$\texttt{torch.autograd.grad(outputs, inputs, grad\_outputs, create\_graph)}$$

Returns a tuple of gradients $\left(\frac{\partial \text{outputs}}{\partial \text{inputs}_i}\right)$ for each input.

**Key parameters:**
- `outputs`: The tensor(s) to differentiate
- `inputs`: The tensor(s) to differentiate with respect to
- `grad_outputs`: Seed gradient (usually `torch.ones_like(outputs)`)
- `create_graph`: If `True`, the gradient computation itself is differentiable (needed for higher-order derivatives)

### Jacobian and Hessian

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

PyTorch's `backward()` computes vector-Jacobian products: $v^T J$ (one backward pass). For the full Jacobian, use `torch.autograd.functional.jacobian()`.

The Hessian (second derivative matrix) requires `create_graph=True`:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

---

## Code

### Basic Autograd

```python
import torch

# Create tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward pass: y = x^2 + 3x
y = x ** 2 + 3 * x                         # y has grad_fn=<AddBackward0>
print(y)                                    # tensor([10., 18.])

# Backward pass: dy/dx = 2x + 3
# .backward() only works on scalars, so sum first
loss = y.sum()                              # scalar
loss.backward()

print(x.grad)                               # tensor([7., 9.]) — dy/dx at x=[2,3]
```

### Gradient Accumulation and Zeroing

```python
x = torch.tensor([1.0], requires_grad=True)

# First backward
y = x ** 2
y.backward()
print(x.grad)        # tensor([2.])

# Second backward WITHOUT zeroing — gradients accumulate!
y = x ** 2
y.backward()
print(x.grad)        # tensor([4.])  ← 2 + 2, NOT 2

# Correct pattern: zero before each backward
x.grad.zero_()       # In-place zero
y = x ** 2
y.backward()
print(x.grad)        # tensor([2.])  ← correct
```

### torch.no_grad() Context

```python
x = torch.tensor([1.0], requires_grad=True)

# Inside no_grad, operations are NOT tracked
with torch.no_grad():
    y = x * 2
    print(y.requires_grad)    # False
    # y.backward() would fail — no graph

# Common use: evaluation
model.eval()
with torch.no_grad():
    predictions = model(test_data)      # No gradient tracking — saves memory
```

### detach()

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

# Detach: same values, no gradient connection
z = y.detach()
print(z.requires_grad)    # False
print(z)                   # tensor([1.]) — same value as y

# Useful for: using intermediate values as constants
# Example: target network in RL
target = model_target(state).detach()    # Stop gradients here
loss = (prediction - target) ** 2        # Gradients flow to prediction only
```

### torch.autograd.grad — Functional Interface

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3                                 # y = x^3

# Compute dy/dx using functional interface
dy_dx = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=torch.ones_like(y),       # seed: dy/dy = 1
    create_graph=True                       # keep graph for second derivative
)[0]
print(dy_dx)                                # tensor([12.]) — 3x^2 = 3(4) = 12

# Compute d²y/dx² (second derivative)
d2y_dx2 = torch.autograd.grad(
    outputs=dy_dx,
    inputs=x,
    grad_outputs=torch.ones_like(dy_dx),
    create_graph=True                       # could go even higher
)[0]
print(d2y_dx2)                              # tensor([12.]) — 6x = 6(2) = 12
```

### PINNs Pattern (USAAIO 2025 Round 2 Style)

```python
# Physics-Informed Neural Network: solve u_t = alpha * u_xx
# The network u(x, t) takes coordinates and outputs the solution

def compute_pde_residual(model, x, t, alpha=0.01):
    """Compute PDE residual: u_t - alpha * u_xx should be zero."""
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(torch.cat([x, t], dim=-1))    # u(x, t), shape: (N, 1)

    # First derivatives
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True                    # MUST keep graph for u_xx
    )[0]                                     # shape: (N, 1)

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True                    # MUST keep graph for u_xx
    )[0]                                     # shape: (N, 1)

    # Second derivative
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True                    # Need graph for backward through loss
    )[0]                                     # shape: (N, 1)

    residual = u_t - alpha * u_xx            # Should be zero if PDE is satisfied
    return residual                          # shape: (N, 1)
```

### Custom Gradient with autograd.Function

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)         # Save for backward pass
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0            # Gradient is 0 where input < 0
        return grad_input

# Usage
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)                         # Use .apply(), not direct call
y.sum().backward()
print(x.grad)                                # Zero where x < 0, 1 where x > 0
```

### Useful Autograd Utilities

```python
# Check if gradient tracking is on
x = torch.randn(3)
print(x.requires_grad)                      # False (default)

# Enable gradient tracking
x.requires_grad_(True)                      # In-place enable
# or
x = torch.randn(3, requires_grad=True)      # At creation

# Check if we're in no_grad mode
print(torch.is_grad_enabled())               # True normally, False inside no_grad

# Gradient of non-scalar: must provide grad_outputs
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2                                   # shape: (3,)
y.backward(gradient=torch.tensor([1.0, 1.0, 1.0]))  # dy/dx with uniform weights
print(x.grad)                                # tensor([2., 4., 6.])
```
