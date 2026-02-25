# 08 — Convex Optimization

**Prerequisites**: `07-multivariable-calculus.md` (gradient, Hessian), `02-matrix-operations.md` (positive definiteness)
**USAAIO Relevance**: **Medium-High priority.** Gradient descent is the workhorse of all ML training. USAAIO tests convexity identification, gradient descent convergence, and conceptual understanding of KKT conditions and Lagrangian duality.

---

## Discovery

It's 1847, and you're Augustin-Louis Cauchy. You've just proposed a radical idea: instead of solving equations analytically (which is often impossible for complex functions), why not iteratively take small steps downhill until you reach the bottom?

**Motivating challenge**: Find the minimum of $f(x) = x^4 - 3x^2 + x + 1$.

Analytically, you'd set $f'(x) = 4x^3 - 6x + 1 = 0$ — a cubic equation. Messy! Now imagine this in 1 million dimensions — analytical solutions are hopeless.

Instead: start at $x_0 = 2$, then repeatedly:
1. Compute $f'(x_t) = 4x_t^3 - 6x_t + 1$
2. Step: $x_{t+1} = x_t - 0.01 \cdot f'(x_t)$

After enough steps, you converge to a minimum. But will this ALWAYS work? When does it fail?

**Socratic questions**:
1. If $f$ has multiple valleys (local minima), will gradient descent always find the deepest one? (No! It finds a *local* minimum, which might not be global)
2. What special property would $f$ need to guarantee that any local minimum is the global minimum? (Convexity!)
3. What happens if your step size is too large? (You overshoot and oscillate or diverge)

**Misconception trap**: "Gradient descent finds the global minimum." This is only true for **convex** functions. For non-convex functions (like neural network losses), gradient descent finds local minima. The surprising fact that deep learning works despite non-convexity is one of the great mysteries of modern ML.

---

## Intuition

What you just discovered is **gradient descent** — the simplest and most important optimization algorithm in ML. And the question "when is gradient descent guaranteed to work?" leads to **convexity theory**.

### Convex vs Non-Convex

```
  Convex function:              Non-convex function:

       \_____/                      /\    /\
      \_______/                   _/  \__/  \_
     \_________/                 /    saddle   \
     single minimum             multiple local minima
     = global minimum           GD may get stuck!
```

A convex function has ONE valley — no matter where you start, gradient descent will reach the bottom.

### The Lagrangian: Turning Constraints into Penalties

Imagine you want to minimize $f(x, y) = x + y$ but you're constrained to stay on the circle $x^2 + y^2 = 1$. The Lagrangian trick: instead of enforcing the constraint exactly, add a penalty term:

$$\mathcal{L}(x, y, \lambda) = (x + y) + \lambda(x^2 + y^2 - 1)$$

The optimal $\lambda$ automatically enforces the constraint. This is the foundation of constrained optimization in ML (e.g., SVMs, regularization).

### What Goes Wrong Without Optimization Theory?

- Can't set learning rates properly → training diverges or is painfully slow
- Don't understand why Adam works better than SGD in some cases
- Can't diagnose training failures (saddle points, ill-conditioning)
- Can't design new loss functions with desirable optimization properties

---

## Math

### Convex Sets

A set $S \subseteq \mathbb{R}^n$ is **convex** if for all $\mathbf{x}, \mathbf{y} \in S$ and $\theta \in [0, 1]$:

$$\theta\mathbf{x} + (1-\theta)\mathbf{y} \in S$$

Geometrically: the line segment between any two points in $S$ lies entirely in $S$.

Examples: hyperplanes, half-spaces, balls, polyhedra, the PSD cone.

### Convex Functions

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\theta \in [0, 1]$:

$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

The function lies below (or on) the line segment connecting any two points.

**Equivalent conditions** (when $f$ is differentiable):

**First-order condition**: $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$

The function lies above its tangent plane at every point.

**Second-order condition**: $\nabla^2 f(\mathbf{x}) \succeq 0$ (Hessian is positive semi-definite) for all $\mathbf{x}$.

*Reasoning required*: You should be able to check convexity using the second-order condition.

**Strictly convex**: $\nabla^2 f \succ 0$ (PD) everywhere. Has at most one minimum.

**Key examples**:

| Function | Convex? | Hessian |
|----------|---------|---------|
| $f(x) = x^2$ | Yes (strictly) | $f'' = 2 > 0$ |
| $f(\mathbf{x}) = \|\mathbf{x}\|^2$ | Yes (strictly) | $H = 2I \succ 0$ |
| $f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x}$, $A \succeq 0$ | Yes | $H = A + A^\top \succeq 0$ |
| $f(x) = e^x$ | Yes (strictly) | $f'' = e^x > 0$ |
| $f(x) = \|x\|$ | Yes (not strictly) | — (not differentiable at 0) |
| $f(x) = x^3$ | No | $f'' = 6x$ (changes sign) |

### Gradient Descent

**Algorithm**: Starting from $\mathbf{x}_0$, iterate:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

where $\eta > 0$ is the **learning rate** (step size).

**Convergence for convex functions**:

If $f$ is convex with $L$-Lipschitz gradient ($\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$), then with $\eta = 1/L$:

$$f(\mathbf{x}_T) - f(\mathbf{x}^*) \leq \frac{L\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2T}$$

Rate: $O(1/T)$ — halving the error requires doubling the iterations.

**For strongly convex functions** (eigenvalues of Hessian bounded below by $\mu > 0$): linear convergence rate $O((1 - \mu/L)^T)$.

### Learning Rate Selection

- $\eta$ too large: overshooting → oscillation or divergence
- $\eta$ too small: convergence too slow
- Optimal: $\eta = 1/L$ (if $L$ is known)
- Practice: use adaptive methods (Adam) or learning rate schedules

```
η too large:          η too small:          η just right:
  *                     *                     *
   \  *                  \                     \
    \/                    \                     \
    /\                     *                     *
   /  *                     \                     \
  *    \                     *                     *
  (diverges)                  \                    (converges to x*)
                               *
                              (very slow)
```

### Momentum

Standard gradient descent can oscillate in "narrow valleys." Momentum smooths the trajectory:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \mathbf{v}_{t+1}$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Intuition**: $\mathbf{v}$ is a "velocity" that accumulates gradient history. Like a ball rolling downhill, it builds speed in consistent directions and dampens oscillations.

### Constrained Optimization: Lagrangian Duality

**Problem**: minimize $f(\mathbf{x})$ subject to $g_i(\mathbf{x}) \leq 0$ and $h_j(\mathbf{x}) = 0$.

**Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i} \lambda_i g_i(\mathbf{x}) + \sum_{j} \nu_j h_j(\mathbf{x})$$

- $\lambda_i \geq 0$: Lagrange multipliers for inequality constraints
- $\nu_j$: Lagrange multipliers for equality constraints

### KKT Conditions

The **Karush-Kuhn-Tucker conditions** are necessary conditions for optimality of constrained optimization. At the optimal point $(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$:

1. **Stationarity**: $\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) = \mathbf{0}$
   - The gradient of the Lagrangian w.r.t. $\mathbf{x}$ is zero

2. **Primal feasibility**: $g_i(\mathbf{x}^*) \leq 0$ for all $i$; $h_j(\mathbf{x}^*) = 0$ for all $j$
   - The solution satisfies all constraints

3. **Dual feasibility**: $\lambda_i^* \geq 0$ for all $i$
   - Multipliers for inequality constraints are non-negative

4. **Complementary slackness**: $\lambda_i^* g_i(\mathbf{x}^*) = 0$ for all $i$
   - Either the constraint is active ($g_i = 0$) or the multiplier is zero

**For convex problems**: KKT conditions are both necessary AND sufficient.

*Reasoning required*: Conceptual understanding of KKT conditions. USAAIO may ask you to identify which conditions are violated or to apply them to simple problems.

### Example: Constrained Quadratic

Minimize $f(x, y) = x^2 + y^2$ subject to $x + y = 1$.

**Lagrangian**: $\mathcal{L}(x, y, \nu) = x^2 + y^2 + \nu(x + y - 1)$

**Stationarity**:
$$\frac{\partial \mathcal{L}}{\partial x} = 2x + \nu = 0 \implies x = -\nu/2$$
$$\frac{\partial \mathcal{L}}{\partial y} = 2y + \nu = 0 \implies y = -\nu/2$$

**Primal feasibility**: $x + y = 1 \implies -\nu = 1 \implies \nu = -1$

**Solution**: $x^* = y^* = 1/2$, $f^* = 1/2$.

---

## Code

### From-Scratch NumPy Implementation

```python
import numpy as np

def gradient_descent(f, grad_f, x0: np.ndarray, lr: float = 0.01,
                     max_iter: int = 1000, tol: float = 1e-8) -> dict:
    """Gradient descent optimization.

    Args:
        f: objective function R^n -> R
        grad_f: gradient function R^n -> R^n
        x0: shape (n,) — starting point
        lr: learning rate
        max_iter: maximum iterations
        tol: convergence tolerance on gradient norm
    Returns:
        dict with x_opt, f_opt, trajectory, n_iter
    """
    x = x0.copy()  # (n,)
    trajectory = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)  # (n,) — gradient

        if np.linalg.norm(g) < tol:
            break

        x = x - lr * g  # (n,) — gradient step
        trajectory.append(x.copy())

    return {
        'x_opt': x,  # (n,)
        'f_opt': f(x),  # scalar
        'trajectory': np.array(trajectory),  # (T, n)
        'n_iter': i + 1
    }

def gradient_descent_momentum(f, grad_f, x0: np.ndarray, lr: float = 0.01,
                               beta: float = 0.9, max_iter: int = 1000,
                               tol: float = 1e-8) -> dict:
    """Gradient descent with momentum.

    Args:
        f, grad_f, x0, lr, max_iter, tol: same as above
        beta: momentum coefficient
    Returns:
        dict with x_opt, f_opt, trajectory, n_iter
    """
    x = x0.copy()  # (n,)
    v = np.zeros_like(x)  # (n,) — velocity
    trajectory = [x.copy()]

    for i in range(max_iter):
        g = grad_f(x)  # (n,)

        if np.linalg.norm(g) < tol:
            break

        v = beta * v + g  # (n,) — update velocity
        x = x - lr * v  # (n,) — step with momentum
        trajectory.append(x.copy())

    return {
        'x_opt': x,
        'f_opt': f(x),
        'trajectory': np.array(trajectory),
        'n_iter': i + 1
    }

def is_convex(hessian_fn, test_points: np.ndarray) -> bool:
    """Check convexity by verifying Hessian is PSD at test points.

    Args:
        hessian_fn: function returning (n, n) Hessian at a point
        test_points: shape (K, n) — points to check
    Returns:
        True if Hessian is PSD at all test points
    """
    for x in test_points:
        H = hessian_fn(x)  # (n, n)
        eigenvalues = np.linalg.eigvalsh(H)  # (n,)
        if np.any(eigenvalues < -1e-10):
            return False
    return True

# --- Demo ---
np.random.seed(42)

# Quadratic: f(x) = x^T A x + b^T x  (convex if A is PSD)
A = np.array([[4, 1], [1, 3]], dtype=float)  # (2, 2) — PSD
b = np.array([-2, -1], dtype=float)  # (2,)

f = lambda x: 0.5 * x @ A @ x + b @ x  # scalar
grad_f = lambda x: A @ x + b  # (2,) — gradient of quadratic

# Verify convexity
print(f"A eigenvalues: {np.linalg.eigvalsh(A)}")  # both positive → convex
print(f"Is convex: {np.all(np.linalg.eigvalsh(A) >= 0)}")

# Run gradient descent
result = gradient_descent(f, grad_f, x0=np.array([5.0, 5.0]), lr=0.1)
print(f"\nGD converged in {result['n_iter']} iterations")
print(f"Optimal x: {result['x_opt']}")
print(f"Optimal f: {result['f_opt']:.6f}")

# Analytical solution: x* = -A^{-1} b
x_star = -np.linalg.solve(A, b)  # (2,)
print(f"Analytical x*: {x_star}")
print(f"Match: {np.allclose(result['x_opt'], x_star, atol=1e-5)}")

# Compare with momentum
result_mom = gradient_descent_momentum(f, grad_f,
                                        x0=np.array([5.0, 5.0]),
                                        lr=0.05, beta=0.9)
print(f"\nMomentum converged in {result_mom['n_iter']} iterations")
print(f"Optimal x: {result_mom['x_opt']}")

# Learning rate effect
for lr in [0.01, 0.1, 0.5, 1.0]:
    try:
        result_lr = gradient_descent(f, grad_f, x0=np.array([5.0, 5.0]), lr=lr)
        print(f"lr={lr:.2f}: converged in {result_lr['n_iter']} iter, f={result_lr['f_opt']:.4f}")
    except (OverflowError, FloatingPointError):
        print(f"lr={lr:.2f}: DIVERGED")
```

### PyTorch Equivalent

```python
import torch
import torch.optim as optim

# Define parameters
x = torch.tensor([5.0, 5.0], requires_grad=True)  # (2,)
A = torch.tensor([[4., 1.], [1., 3.]])  # (2, 2)
b = torch.tensor([-2., -1.])  # (2,)

# Optimizer
optimizer = optim.SGD([x], lr=0.1)
# Alternative: optimizer = optim.Adam([x], lr=0.1)
# Alternative: optimizer = optim.SGD([x], lr=0.05, momentum=0.9)

for i in range(100):
    optimizer.zero_grad()
    loss = 0.5 * x @ A @ x + b @ x  # scalar
    loss.backward()
    optimizer.step()

    if x.grad.norm() < 1e-6:
        print(f"Converged at iteration {i}")
        break

print(f"Optimal x: {x.data}")
print(f"Optimal f: {(0.5 * x @ A @ x + b @ x).item():.6f}")
```

---

## Resources

- [Boyd & Vandenberghe, *Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) — the definitive reference (free PDF)
- [Sebastian Ruder: An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/) — SGD, Adam, etc.
- MML Book, Chapter 7: Continuous Optimization
- [3Blue1Brown: Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent) — visual intuition
