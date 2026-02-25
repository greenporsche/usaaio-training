# 07 — Multivariable Calculus

**Prerequisites**: Basic single-variable calculus (derivatives, chain rule). Helpful: `02-matrix-operations.md` (matrix properties)
**USAAIO Relevance**: **High priority.** Gradients, Jacobians, and the chain rule for vectors are how neural networks learn. USAAIO tests gradient computation, matrix calculus identities, and the ability to derive gradients of loss functions.

---

## Discovery

It's 1788, and you're Joseph-Louis Lagrange. You've just published *Mecanique Analytique*, reformulating all of mechanics using calculus. But you're working in one dimension. Now imagine you're dropped into a mountain landscape with fog so thick you can't see more than a step ahead. You want to descend to the valley floor. The ground slopes differently in the north-south direction versus the east-west direction. How do you figure out which way is steepest downhill?

**Motivating challenge**: You're standing at position $(x, y) = (1, 2)$ on a surface defined by:

$$f(x, y) = x^2 + 2y^2 + xy$$

Which direction should you step to decrease $f$ the fastest?

**Socratic questions**:
1. The slope in the $x$-direction (holding $y$ fixed) is $\frac{\partial f}{\partial x}$. What is it at $(1, 2)$?
   - $\frac{\partial f}{\partial x} = 2x + y = 2(1) + 2 = 4$
2. The slope in the $y$-direction is $\frac{\partial f}{\partial y}$. What is it at $(1, 2)$?
   - $\frac{\partial f}{\partial y} = 4y + x = 4(2) + 1 = 9$
3. The steepest ascent direction is $\nabla f = \begin{bmatrix} 4 \\ 9 \end{bmatrix}$. So steepest descent is $-\nabla f$. Does this make sense geometrically?

**Misconception trap**: The gradient $\nabla f$ at a point is a *vector*, not a scalar. Students sometimes confuse the gradient (a direction in input space) with the directional derivative (a scalar measuring slope in a specific direction). The gradient is the direction of maximum directional derivative.

---

## Intuition

What you discovered is the **gradient** — the multi-dimensional generalization of the derivative. It points in the direction of steepest ascent, and its magnitude tells you how steep that ascent is.

### Geometric Picture

```
        Contour plot of f(x,y):     Gradient vectors:

        y                           y
        |   (  (  (                 |   ↗  ↗  ↗
        |  (  (  (                  |  ↗  ↗  ↗
        | (  (  (                   | ↗  ↗  ↗
        |(  (  (                    |↗  ↗  ↗
        +----------> x              +----------> x

  Contour lines = constant f        Gradients are perpendicular
                                    to contour lines!
```

The gradient is always perpendicular to the contour lines of $f$. This is the key geometric insight that makes gradient descent work.

### The Calculus Hierarchy

| Object | Input → Output | Shape | What it measures |
|--------|---------------|-------|-----------------|
| **Gradient** $\nabla f$ | $\mathbb{R}^n \to \mathbb{R}$ | $(n,)$ | Steepest ascent direction of scalar function |
| **Jacobian** $\mathbf{J}$ | $\mathbb{R}^n \to \mathbb{R}^m$ | $(m, n)$ | How each output changes w.r.t. each input |
| **Hessian** $\mathbf{H}$ | $\mathbb{R}^n \to \mathbb{R}$ | $(n, n)$ | Curvature — how the gradient itself changes |

### What Goes Wrong Without Multivariable Calculus?

- Can't compute gradients for backpropagation (the chain rule for vectors)
- Can't understand why gradient descent converges (or doesn't)
- Can't derive loss function gradients by hand
- Can't reason about the curvature of the loss landscape (important for optimizer design)

---

## Math

### Partial Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$, the partial derivative with respect to $x_i$:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Just differentiate treating all other variables as constants.

### Gradient

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n$$

**Key property**: The directional derivative of $f$ in direction $\mathbf{d}$ ($\|\mathbf{d}\| = 1$) is:

$$D_\mathbf{d} f = \nabla f \cdot \mathbf{d} = \|\nabla f\| \cos\theta$$

This is maximized when $\mathbf{d}$ is parallel to $\nabla f$ ($\theta = 0$), confirming: **the gradient points in the direction of steepest ascent**.

### Jacobian Matrix

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ (vector-valued function), the Jacobian:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

Row $i$ of $\mathbf{J}$ = gradient of $f_i$ (transposed). The Jacobian is the "matrix version" of the derivative.

**Note**: For a scalar function ($m = 1$), the Jacobian is the gradient transposed: $\mathbf{J} = \nabla f^\top$.

### Hessian Matrix

For $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian:

$$\mathbf{H} = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix} \in \mathbb{R}^{n \times n}$$

$\mathbf{H}$ is symmetric (assuming continuous second derivatives). Its eigenvalues determine curvature:
- All eigenvalues $> 0$: local minimum (bowl shape)
- All eigenvalues $< 0$: local maximum (dome shape)
- Mixed signs: saddle point

### Chain Rule for Vectors

This is the most important rule for understanding backpropagation.

**Single composition**: If $L = g(\mathbf{z})$ and $\mathbf{z} = \mathbf{f}(\mathbf{x})$:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{x}}^\top \frac{\partial L}{\partial \mathbf{z}} = \mathbf{J}_\mathbf{f}^\top \nabla_\mathbf{z} L$$

**Shape check**: If $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{z} \in \mathbb{R}^m$, $L \in \mathbb{R}$:
- $\mathbf{J}_\mathbf{f} \in \mathbb{R}^{m \times n}$, so $\mathbf{J}_\mathbf{f}^\top \in \mathbb{R}^{n \times m}$
- $\nabla_\mathbf{z} L \in \mathbb{R}^m$
- Result: $\mathbb{R}^{n \times m} \cdot \mathbb{R}^m = \mathbb{R}^n$ $\checkmark$

**Multi-layer composition** (backpropagation):
$$L = g(\mathbf{z}_3), \quad \mathbf{z}_3 = \mathbf{f}_3(\mathbf{z}_2), \quad \mathbf{z}_2 = \mathbf{f}_2(\mathbf{z}_1), \quad \mathbf{z}_1 = \mathbf{f}_1(\mathbf{x})$$

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}_{\mathbf{f}_1}^\top \mathbf{J}_{\mathbf{f}_2}^\top \mathbf{J}_{\mathbf{f}_3}^\top \nabla_{\mathbf{z}_3} g$$

This is exactly how backpropagation works — multiplying Jacobians from output to input.

*Reasoning required*: USAAIO expects you to apply the chain rule to compute gradients of composite functions.

### Matrix Calculus Identities

These are the essential identities for ML. Learn them by heart.

| # | Function | $\nabla_\mathbf{x}$ (gradient w.r.t. $\mathbf{x}$) |
|---|----------|-----------------------------------------------------|
| 1 | $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| 2 | $\mathbf{x}^\top \mathbf{x} = \|\mathbf{x}\|^2$ | $2\mathbf{x}$ |
| 3 | $\mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$; if $A = A^\top$: $2A\mathbf{x}$ |
| 4 | $\|A\mathbf{x} - \mathbf{b}\|^2$ | $2A^\top(A\mathbf{x} - \mathbf{b})$ |

**Derivation of #3** ($f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$):

$$f(\mathbf{x}) = \sum_{i,j} x_i A_{ij} x_j$$

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i x_i A_{ik} = (A\mathbf{x})_k + (A^\top\mathbf{x})_k$$

$$\therefore \nabla_\mathbf{x} f = (A + A^\top)\mathbf{x} \qquad \blacksquare$$

**Derivation of #4** ($f(\mathbf{x}) = \|A\mathbf{x} - \mathbf{b}\|^2 = (A\mathbf{x} - \mathbf{b})^\top(A\mathbf{x} - \mathbf{b})$):

Let $\mathbf{r} = A\mathbf{x} - \mathbf{b}$. Then $f = \mathbf{r}^\top \mathbf{r}$.

$$f = \mathbf{x}^\top A^\top A \mathbf{x} - 2\mathbf{b}^\top A \mathbf{x} + \mathbf{b}^\top\mathbf{b}$$

Using identities #1 and #3 (note $A^\top A$ is symmetric):

$$\nabla_\mathbf{x} f = 2A^\top A \mathbf{x} - 2A^\top\mathbf{b} = 2A^\top(A\mathbf{x} - \mathbf{b}) \qquad \blacksquare$$

Setting this to zero gives the **normal equations**: $A^\top A \mathbf{x} = A^\top \mathbf{b}$, which is the solution to least-squares regression.

### Gradient of Common ML Losses

**Mean Squared Error**: $L = \frac{1}{N}\|X\mathbf{w} - \mathbf{y}\|^2$

$$\nabla_\mathbf{w} L = \frac{2}{N} X^\top(X\mathbf{w} - \mathbf{y})$$

Shape: $\nabla_\mathbf{w} L \in \mathbb{R}^d$ where $\mathbf{w} \in \mathbb{R}^d$, $X \in \mathbb{R}^{N \times d}$, $\mathbf{y} \in \mathbb{R}^N$.

**Cross-Entropy Loss** (softmax + NLL): For logits $\mathbf{z}$ and target class $k$:

$$\nabla_{\mathbf{z}} L = \text{softmax}(\mathbf{z}) - \mathbf{e}_k$$

where $\mathbf{e}_k$ is the one-hot vector for class $k$.

*Reasoning required*: Deriving the gradient of MSE loss.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute gradient numerically using central differences.

    Args:
        f: scalar-valued function taking np.ndarray
        x: shape (n,) — point to evaluate gradient
        eps: perturbation size
    Returns:
        grad: shape (n,) — numerical gradient
    """
    n = len(x)
    grad = np.zeros(n)  # (n,)

    for i in range(n):
        x_plus = x.copy()  # (n,)
        x_minus = x.copy()  # (n,)
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)  # scalar

    return grad  # (n,)

def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute Jacobian numerically.

    Args:
        f: vector-valued function R^n -> R^m
        x: shape (n,)
        eps: perturbation size
    Returns:
        J: shape (m, n)
    """
    n = len(x)
    f0 = f(x)  # (m,)
    m = len(f0)
    J = np.zeros((m, n))  # (m, n)

    for i in range(n):
        x_plus = x.copy()  # (n,)
        x_minus = x.copy()  # (n,)
        x_plus[i] += eps
        x_minus[i] -= eps
        J[:, i] = (f(x_plus) - f(x_minus)) / (2 * eps)  # (m,)

    return J  # (m, n)

def numerical_hessian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute Hessian numerically.

    Args:
        f: scalar-valued function
        x: shape (n,)
    Returns:
        H: shape (n, n)
    """
    n = len(x)
    H = np.zeros((n, n))  # (n, n)

    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)

    return H  # (n, n)

# --- Demo: Verify matrix calculus identities ---

# Identity #3: grad of x^T A x
A = np.array([[3, 1], [2, 4]], dtype=float)  # (2, 2)
x = np.array([1.0, 2.0])  # (2,)

f_quadratic = lambda x: x @ A @ x  # scalar
grad_numerical = numerical_gradient(f_quadratic, x)  # (2,)
grad_analytical = (A + A.T) @ x  # (2,) — identity #3
print(f"x^T A x gradient:")
print(f"  Numerical:  {grad_numerical}")
print(f"  Analytical: {grad_analytical}")
print(f"  Match: {np.allclose(grad_numerical, grad_analytical)}")

# Identity #4: grad of ||Ax - b||^2
A_rect = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # (3, 2)
b = np.array([1.0, 2.0, 3.0])  # (3,)
x = np.array([0.5, 1.5])  # (2,)

f_lsq = lambda x: np.sum((A_rect @ x - b)**2)  # scalar
grad_numerical = numerical_gradient(f_lsq, x)  # (2,)
grad_analytical = 2 * A_rect.T @ (A_rect @ x - b)  # (2,) — identity #4
print(f"\n||Ax - b||^2 gradient:")
print(f"  Numerical:  {grad_numerical}")
print(f"  Analytical: {grad_analytical}")
print(f"  Match: {np.allclose(grad_numerical, grad_analytical)}")

# Hessian of quadratic form
H = numerical_hessian(f_quadratic, np.array([1.0, 2.0]))  # (2, 2)
print(f"\nHessian of x^T A x:")
print(f"  Numerical:\n{H}")
print(f"  Analytical (A + A^T):\n{A + A.T}")

# Chain rule example: f(x) = ||sigma(Wx)||^2
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # same shape as z

W = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)  # (2, 3)
x = np.array([0.5, -1.0, 2.0])  # (3,)

def composite_f(x):
    z = W @ x  # (2,) = (2, 3) @ (3,)
    a = sigmoid(z)  # (2,)
    return np.sum(a**2)  # scalar

grad_chain = numerical_gradient(composite_f, x)  # (3,)
print(f"\nChain rule composite gradient: {grad_chain}")

# Analytical chain rule: dL/dx = W^T * diag(sigmoid'(z)) * 2*sigmoid(z)
z = W @ x  # (2,)
a = sigmoid(z)  # (2,)
da_dz = a * (1 - a)  # (2,) — sigmoid derivative
dL_da = 2 * a  # (2,)
dL_dz = dL_da * da_dz  # (2,) — element-wise
dL_dx = W.T @ dL_dz  # (3,) = (3, 2) @ (2,)
print(f"Analytical (chain rule):       {dL_dx}")
print(f"Match: {np.allclose(grad_chain, dL_dx)}")
```

### PyTorch Equivalent (Autograd)

```python
import torch

# PyTorch computes gradients automatically!
x = torch.tensor([1.0, 2.0], requires_grad=True)  # (2,)
A = torch.tensor([[3., 1.], [2., 4.]])  # (2, 2)

# Forward pass
f = x @ A @ x  # scalar

# Backward pass — computes all gradients automatically
f.backward()

# Gradient
print(f"Autograd gradient: {x.grad}")  # (2,) — matches (A + A^T) @ x

# Multi-layer example (like a neural network)
W1 = torch.randn(3, 4, requires_grad=True)  # (3, 4)
W2 = torch.randn(2, 3, requires_grad=True)  # (2, 3)
x = torch.randn(4)  # (4,)

z1 = torch.relu(W1 @ x)  # (3,)
z2 = W2 @ z1  # (2,)
loss = z2.sum()  # scalar

loss.backward()
print(f"dL/dW1 shape: {W1.grad.shape}")  # (3, 4)
print(f"dL/dW2 shape: {W2.grad.shape}")  # (2, 3)
```

---

## Resources

- [Matrix Calculus for Deep Learning](https://explained.ai/matrix-calculus/) — Parr & Howard, best practical reference
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — comprehensive identity reference
- MML Book, Chapter 5: Vector Calculus
- [Stanford CS231n: Backpropagation](https://cs231n.stanford.edu/handouts/derivatives.pdf)
