# Autograd Exercises

**Topic**: Computation graphs, backward(), grad, higher-order derivatives, detach, no_grad
**Difficulty**: Intermediate → Advanced (autograd is USAAIO-critical)

---

## Exercise 1: Trace the Computation Graph

Predict the value of `x.grad` after each code block. Do NOT run the code first.

### Part A

```python
x = torch.tensor([3.0], requires_grad=True)
y = 2 * x ** 2 + 5 * x - 3
y.backward()
# What is x.grad?
```

### Part B

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
y.backward()
# x.grad = ?

z = x ** 2
z.backward()
# x.grad = ?  (without zeroing!)
```

### Part C

```python
x = torch.tensor([1.0], requires_grad=True)
y = torch.sin(x)
y.backward()
# x.grad = ?
```

<details>
<summary>Solution</summary>

**Part A**: $y = 2x^2 + 5x - 3$, so $\frac{dy}{dx} = 4x + 5$. At $x = 3$: $4(3) + 5 = 17$. **x.grad = tensor([17.])**

**Part B**: First backward: $y = x^3$, $\frac{dy}{dx} = 3x^2 = 3(4) = 12$. So x.grad = tensor([12.]).

Second backward WITHOUT zeroing: $z = x^2$, $\frac{dz}{dx} = 2x = 2(2) = 4$. Gradients accumulate: x.grad = tensor([12. + 4.]) = **tensor([16.])**

**Part C**: $y = \sin(x)$, $\frac{dy}{dx} = \cos(x)$. At $x = 1$: $\cos(1) \approx 0.5403$. **x.grad = tensor([0.5403])**

**Key insight**: Gradients accumulate by default. Always call `x.grad.zero_()` or `optimizer.zero_grad()` between backward passes.
</details>

---

## Exercise 2: Higher-Order Derivatives

Compute the first, second, and third derivatives of $f(x) = x^4$ at $x = 2$ using `torch.autograd.grad`.

```python
x = torch.tensor([2.0], requires_grad=True)

# f(x) = x^4
f = x ** 4

# Compute f'(x), f''(x), f'''(x)
# YOUR CODE HERE
# f_prime = ...
# f_double_prime = ...
# f_triple_prime = ...
```

Verify:
- $f'(x) = 4x^3 = 4(8) = 32$
- $f''(x) = 12x^2 = 12(4) = 48$
- $f'''(x) = 24x = 24(2) = 48$

<details>
<summary>Solution</summary>

```python
x = torch.tensor([2.0], requires_grad=True)
f = x ** 4

f_prime = torch.autograd.grad(f, x, create_graph=True)[0]
print(f"f'(x)   = {f_prime.item()}")        # 32.0

f_double_prime = torch.autograd.grad(f_prime, x, create_graph=True)[0]
print(f"f''(x)  = {f_double_prime.item()}")  # 48.0

f_triple_prime = torch.autograd.grad(f_double_prime, x, create_graph=True)[0]
print(f"f'''(x) = {f_triple_prime.item()}")  # 48.0
```

**Key insight**: `create_graph=True` is essential for higher-order derivatives. Without it, the computation graph is destroyed after the first `autograd.grad` call, and the second call would fail. This pattern is exactly what PINNs require for computing $u_{xx}$.
</details>

---

## Exercise 3: Detach Bug

This code tries to implement a "stop gradient" for the target in a contrastive loss. It has a bug. Find and fix it.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute two "views" of x
view_1 = x * 2
view_2 = x * 3

# We want gradients to flow through view_1 but NOT through view_2
loss = ((view_1 - view_2) ** 2).sum()
loss.backward()

print(x.grad)
# Expected: gradient only from view_1 (view_2 treated as constant)
# Actual: gradient from BOTH view_1 and view_2
```

<details>
<summary>Solution</summary>

The bug is that `view_2` is not detached. Gradients flow through both branches.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

view_1 = x * 2
view_2 = (x * 3).detach()              # Stop gradient here

loss = ((view_1 - view_2) ** 2).sum()
loss.backward()

print(x.grad)
# Now gradients only flow through view_1
# loss = (2x - c)^2 where c = 3x (treated as constant)
# d(loss)/dx = 2 * (2x - c) * 2 = 4(2x - 3x) = 4(-x) = -4x
# At x = [1, 2, 3]: grad = [-4, -8, -12]
```

Without detach: $\frac{\partial}{\partial x}(2x - 3x)^2 = \frac{\partial}{\partial x}(-x)^2 = 2(-x)(-1) = 2x \cdot (-1 + ?) $... let's compute properly:

$\text{loss} = (2x - 3x)^2 = x^2$, $\frac{d}{dx}x^2 = 2x = [2, 4, 6]$ (without detach).

With detach (c = 3x is constant): $\text{loss} = (2x - c)^2$, $\frac{d}{dx} = 2(2x - c) \cdot 2 = 4(2x - 3x) = -4x = [-4, -8, -12]$.

**Key insight**: `detach()` is critical for implementing techniques like target networks (DQN), stop-gradient in contrastive learning (SimCLR, BYOL), and anywhere you need asymmetric gradient flow.
</details>

---

## Exercise 4: Gradient of a Vector-Valued Function

Compute the Jacobian of $f(x) = [x_1^2 + x_2, x_1 x_2]$ at $x = [1, 2]$ using `torch.autograd.grad`.

The Jacobian should be:

$$J = \begin{bmatrix} 2x_1 & 1 \\ x_2 & x_1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}$$

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

f1 = x[0] ** 2 + x[1]
f2 = x[0] * x[1]

# Compute the full 2x2 Jacobian
# YOUR CODE HERE
```

<details>
<summary>Solution</summary>

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

f1 = x[0] ** 2 + x[1]
f2 = x[0] * x[1]

# Row 1: df1/dx
grad_f1 = torch.autograd.grad(f1, x, retain_graph=True)[0]
print(f"df1/dx = {grad_f1}")       # tensor([2., 1.])

# Row 2: df2/dx
grad_f2 = torch.autograd.grad(f2, x)[0]
print(f"df2/dx = {grad_f2}")       # tensor([2., 1.])

# Full Jacobian
J = torch.stack([grad_f1, grad_f2])
print(f"Jacobian:\n{J}")
# tensor([[2., 1.],
#         [2., 1.]])
```

Note: we need `retain_graph=True` for the first call because we need the graph again for the second call. The last call can consume the graph.

**Alternative using `torch.autograd.functional.jacobian`:**

```python
def f(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]])

J = torch.autograd.functional.jacobian(f, x)
print(J)   # Same result
```

**Key insight**: `backward()` computes one row of the Jacobian at a time (vector-Jacobian product). For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, computing the full Jacobian requires $m$ backward passes.
</details>

---

## Exercise 5: PINN Derivative Computation

Given a simple 1D function approximated by a neural network $u(x) = \text{net}(x)$, compute:
- $u'(x) = \frac{du}{dx}$
- $u''(x) = \frac{d^2u}{dx^2}$
- The PDE residual: $u''(x) + u(x) = 0$ (simple harmonic oscillator ODE)

```python
import torch
import torch.nn as nn

# Simple network
net = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)

# Collocation points
x = torch.linspace(0, 2 * 3.14159, 100).reshape(-1, 1)
x.requires_grad_(True)

# YOUR CODE: compute u, u_x, u_xx, and the PDE residual
# u = ...
# u_x = ...
# u_xx = ...
# residual = ...  (should be close to zero after training)
```

<details>
<summary>Solution</summary>

```python
# Forward pass
u = net(x)                                     # (100, 1)

# First derivative
u_x = torch.autograd.grad(
    u, x,
    grad_outputs=torch.ones_like(u),           # du/dx for each sample
    create_graph=True                           # Keep graph for second derivative
)[0]                                           # (100, 1)

# Second derivative
u_xx = torch.autograd.grad(
    u_x, x,
    grad_outputs=torch.ones_like(u_x),
    create_graph=True                           # Keep graph for backward through loss
)[0]                                           # (100, 1)

# PDE residual: u'' + u = 0
residual = u_xx + u                            # (100, 1)

# The loss for training would be:
pde_loss = (residual ** 2).mean()              # scalar

# To train:
# optimizer.zero_grad()
# pde_loss.backward()
# optimizer.step()
```

**Key insight**: Each `autograd.grad` call with `create_graph=True` builds the derivative computation into the graph itself. This means `pde_loss.backward()` will compute gradients of the loss (which involves second derivatives of $u$) with respect to the network parameters — effectively computing third-order derivative information. This is the core mechanism behind PINNs.
</details>
