# Linear Regression

**Prerequisites**: Linear algebra (matrix inverse, projections), calculus (gradients), NumPy
**USAAIO Relevance**: Foundation for all regression tasks; normal equation derivation is a classic exam question; gradient descent implementation tested in coding rounds

---

## Discovery

### The Origin Story

In 1809, Carl Friedrich Gauss published his method of least squares to predict the orbit of the asteroid Ceres. The idea was deceptively simple: given noisy observations, find the line (or hyperplane) that minimizes the sum of squared errors. Over 200 years later, this remains the starting point for virtually all of machine learning.

### The Core Question

> Given $n$ data points $(x_1, y_1), \ldots, (x_n, y_n)$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, find a linear function $f(x) = w^Tx + b$ that best predicts $y$ from $x$.

But what does "best" mean? And why squared errors rather than absolute errors?

### Socratic Warm-Up

1. If you have more features than data points ($d > n$), can you still find a unique solution? Why or why not?
2. Why might minimizing squared error give different results than minimizing absolute error?
3. If you add a constant feature $x_0 = 1$ to every data point, how does this absorb the bias term $b$?

### Misconception Traps

- **"Linear regression assumes a linear relationship."** — Actually, we can model nonlinear relationships by adding polynomial features. The "linear" refers to linearity in the *parameters* $w$, not in $x$.
- **"The normal equation always works."** — It fails when $X^TX$ is singular (collinear features or $d > n$).
- **"Gradient descent is always better than the normal equation."** — For small $d$, the normal equation is faster. For large $d$, gradient descent wins.

---

## Intuition

### Geometric View: Projection onto Column Space

The key insight is that linear regression is a *projection*. The columns of $X$ span a subspace of $\mathbb{R}^n$. The predicted values $\hat{y} = Xw$ must live in this subspace. The best prediction is the point in $\text{col}(X)$ closest to the true $y$.

```
y (true labels, lives in R^n)
│  ╲
│   ╲  residual = y - X̂w
│    ╲
│     ╲
│      ● ŷ = Xw (projection onto col(X))
│     ╱
│    ╱
│   ╱ col(X) subspace
│  ╱
└──────────────────────
```

The residual vector $(y - Xw)$ is *perpendicular* to the column space:

$$X^T(y - Xw) = 0 \implies X^TXw = X^Ty$$

This is the **normal equation** — "normal" as in "perpendicular."

### Why Squared Error?

Consider fitting $y = wx$ to three points:

```
  y
  │     /
  │   *  / ← line
  │  / *
  │ /
  │/*
  └──────── x
```

- **Squared error** $(y_i - \hat{y}_i)^2$: Penalizes large errors heavily, differentiable everywhere, leads to a unique closed-form solution.
- **Absolute error** $|y_i - \hat{y}_i|$: More robust to outliers, but not differentiable at zero, no closed-form solution.

### The Bias Trick

Instead of tracking $w$ and $b$ separately, prepend a $1$ to every feature vector:

$$\tilde{x} = \begin{bmatrix} 1 \\ x \end{bmatrix} \in \mathbb{R}^{d+1}, \quad \tilde{w} = \begin{bmatrix} b \\ w \end{bmatrix}$$

Then $f(x) = \tilde{w}^T\tilde{x} = b + w^Tx$. From now on, we absorb $b$ into $w$ and assume $X$ has a column of ones.

---

## Math

### Setup

**Notation**:
- $X \in \mathbb{R}^{n \times d}$: design matrix (rows are data points)
- $y \in \mathbb{R}^n$: target vector
- $w \in \mathbb{R}^d$: weight vector

**Loss function** (MSE):

$$\mathcal{L}(w) = \frac{1}{n}\|Xw - y\|_2^2 = \frac{1}{n}(Xw - y)^T(Xw - y)$$

### Derivation of the Normal Equation

*[Reasoning required for USAAIO]*

**Step 1**: Expand the loss.

$$\mathcal{L}(w) = \frac{1}{n}\left(w^TX^TXw - 2w^TX^Ty + y^Ty\right)$$

**Step 2**: Compute the gradient.

Using matrix calculus identities $\nabla_w (w^TAw) = 2Aw$ (for symmetric $A$) and $\nabla_w (w^Tb) = b$:

$$\nabla_w \mathcal{L} = \frac{1}{n}\left(2X^TXw - 2X^Ty\right) = \frac{2}{n}X^T(Xw - y)$$

**Step 3**: Set gradient to zero.

$$\nabla_w \mathcal{L} = 0 \implies X^TXw = X^Ty$$

**Step 4**: Solve for $w$.

If $X^TX$ is invertible:

$$\boxed{\hat{w} = (X^TX)^{-1}X^Ty}$$

### When Does $X^TX$ Fail to be Invertible?

$X^TX$ is singular when:
1. **Collinear features**: Two columns of $X$ are linearly dependent
2. **More features than data**: $d > n$ means $\text{rank}(X) \leq n < d$

Solution: Use **regularization** (Ridge adds $\lambda I$, making it always invertible).

### Gradient Descent Alternative

When $d$ is large, computing $(X^TX)^{-1}$ is $O(d^3)$. Gradient descent is $O(nd)$ per step:

$$w^{(t+1)} = w^{(t)} - \eta \cdot \frac{2}{n}X^T(Xw^{(t)} - y)$$

**Convergence**: If $\mathcal{L}$ is convex (it is — $X^TX$ is PSD) and $\eta < \frac{1}{\lambda_{\max}(X^TX/n)}$, gradient descent converges to the global minimum.

### Complexity Comparison

| Method | Time | Space |
|---|---|---|
| Normal equation | $O(nd^2 + d^3)$ | $O(d^2)$ |
| Gradient descent | $O(ndT)$ where $T$ = iterations | $O(d)$ |
| Stochastic GD | $O(dT)$ per epoch | $O(d)$ |

---

## Code

### NumPy From-Scratch

```python
import numpy as np

def linear_regression_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve linear regression via normal equation.

    X: (n, d) design matrix (include column of ones for bias)
    y: (n,) target vector
    Returns: (d,) weight vector
    """
    # w = (X^T X)^{-1} X^T y
    return np.linalg.solve(X.T @ X, X.T @ y)  # (d, d) @ (d,) → (d,)


def linear_regression_gd(
    X: np.ndarray,    # (n, d)
    y: np.ndarray,    # (n,)
    lr: float = 0.01,
    n_steps: int = 1000,
) -> np.ndarray:
    """
    Solve linear regression via gradient descent.
    Returns: (d,) weight vector
    """
    n, d = X.shape
    w = np.zeros(d)                           # (d,)

    for _ in range(n_steps):
        grad = (2 / n) * X.T @ (X @ w - y)   # (d, n) @ (n,) → (d,)
        w = w - lr * grad                     # (d,)

    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """X: (n, d), w: (d,) → (n,)"""
    return X @ w


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    n, d = 100, 3
    X_raw = np.random.randn(n, d)                       # (100, 3)
    X = np.hstack([np.ones((n, 1)), X_raw])              # (100, 4) — bias trick
    w_true = np.array([2.0, -1.0, 0.5, 3.0])            # (4,)
    y = X @ w_true + 0.1 * np.random.randn(n)           # (100,)

    w_normal = linear_regression_normal(X, y)
    w_gd = linear_regression_gd(X, y, lr=0.01, n_steps=5000)

    print(f"True weights:    {w_true}")
    print(f"Normal equation: {w_normal.round(4)}")
    print(f"Gradient descent:{w_gd.round(4)}")
    print(f"MSE (normal):    {mse_loss(y, predict(X, w_normal)):.6f}")
    print(f"MSE (GD):        {mse_loss(y, predict(X, w_gd)):.6f}")
```

### PyTorch Equivalent

```python
import torch
import torch.nn as nn

# Data
X_tensor = torch.randn(100, 3)
w_true = torch.tensor([2.0, -1.0, 0.5])
b_true = 3.0
y_tensor = X_tensor @ w_true + b_true + 0.1 * torch.randn(100)

# Model
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(1000):
    pred = model(X_tensor).squeeze()       # (100,)
    loss = loss_fn(pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Learned weights: {model.weight.data}")
print(f"Learned bias:    {model.bias.data}")
```

### Shape Annotation Summary

```
X:       (n, d)     — design matrix
y:       (n,)       — targets
w:       (d,)       — weights
X @ w:   (n,)       — predictions
X.T @ r: (d,)       — gradient (r = residuals)
X.T @ X: (d, d)     — Gram matrix
```

---

## Resources

- CS229 Lecture Notes 1: Linear Regression — [cs229.stanford.edu/main_notes.pdf](https://cs229.stanford.edu/main_notes.pdf)
- 3Blue1Brown: Least Squares as Projection — visual geometry
- Gauss, C.F. (1809) *Theoria Motus Corporum Coelestium* — the original
