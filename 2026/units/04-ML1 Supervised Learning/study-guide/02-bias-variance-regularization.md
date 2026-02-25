# Bias-Variance Tradeoff & Regularization

**Prerequisites**: Linear regression (Topic 01), probability (expectation, variance), linear algebra
**USAAIO Relevance**: Bias-variance decomposition derivation is a frequent exam question; regularization choice (Ridge vs Lasso) tested in both theory and coding problems

---

## Discovery

### The Paradox of Complexity

You build a degree-20 polynomial to fit 15 data points. It passes through every point perfectly — zero training error. You test it on new data and the error is catastrophic. Meanwhile, a simple line with nonzero training error generalizes far better.

This is the **bias-variance tradeoff**: the fundamental tension in machine learning between models that are too simple (underfit) and too complex (overfit).

### Socratic Warm-Up

1. If you double your training data, which decreases more — bias or variance?
2. Can a model have zero bias and zero variance simultaneously?
3. You have two models: one with training error 0.01 and test error 0.50, another with training error 0.15 and test error 0.18. Which is better?

### Misconception Traps

- **"More features always help."** — Extra features can increase variance faster than they decrease bias.
- **"Regularization always hurts training performance."** — True, but that's the point. We sacrifice training accuracy for generalization.
- **"L1 and L2 do the same thing."** — They have fundamentally different geometry and effects on the solution.

---

## Intuition

### The Dartboard Analogy

Imagine throwing darts at a target:

```
HIGH BIAS, LOW VARIANCE      LOW BIAS, HIGH VARIANCE
    (Underfit)                     (Overfit)
  ┌─────────────┐              ┌─────────────┐
  │      ◎      │              │      ◎      │
  │             │              │  •       •  │
  │    • • •   │              │             │
  │    • • •   │              │  •       •  │
  │             │              │             │
  └─────────────┘              └─────────────┘
  Clustered but off-center     Scattered around center

LOW BIAS, LOW VARIANCE        HIGH BIAS, HIGH VARIANCE
    (Ideal)                        (Worst)
  ┌─────────────┐              ┌─────────────┐
  │      ◎      │              │      ◎      │
  │     •••    │              │  •          │
  │     •••    │              │          •  │
  │             │              │    •        │
  │             │              │         •   │
  └─────────────┘              └─────────────┘
  Clustered at center          Scattered and off-center
```

- **Bias** = how far the average dart is from the bullseye (systematic error)
- **Variance** = how spread out the darts are (sensitivity to training data)

### Regularization Geometry

Why does L1 produce sparsity but L2 doesn't? Look at the constraint regions:

```
L2 (Ridge): Circle              L1 (Lasso): Diamond

    w2                               w2
    │   ╱╲                           │   /\
    │  /  \  ← contours             │  / \  ← contours
    │ / ●  \   of loss              │ / ●  \   of loss
    │/  ╱╲  \                       │/  /\  \
────●──────●──── w1             ────●──────●──── w1
    │\  ╲╱  /                       │\  \/  /
    │ \  ● /                        │ \  ● /
    │  \  /                         │  \ /
    │   ╲╱                          │   \/

  ← touches at                    ← touches at corner
    arbitrary point                  (some w_j = 0)
```

The L1 diamond has corners on the axes. The loss contours (ellipses) are more likely to touch the constraint at a corner, setting some $w_j$ exactly to zero.

---

## Math

### Bias-Variance Decomposition

*[Reasoning required for USAAIO]*

**Setup**: Let $y = f(x) + \epsilon$ where $\epsilon \sim (0, \sigma^2)$. Let $\hat{f}(x)$ be our estimate trained on a random dataset $D$.

**Goal**: Decompose $E_D\left[(y - \hat{f}(x))^2\right]$.

**Step 1**: Expand the squared error.

$$E\left[(y - \hat{f})^2\right] = E\left[(f + \epsilon - \hat{f})^2\right]$$

$$= E\left[(f - \hat{f})^2 + 2\epsilon(f - \hat{f}) + \epsilon^2\right]$$

Since $\epsilon$ is independent of $\hat{f}$ and $E[\epsilon] = 0$:

$$= E\left[(f - \hat{f})^2\right] + \sigma^2$$

**Step 2**: Decompose $(f - \hat{f})^2$ using the add-and-subtract trick.

Let $\bar{f} = E_D[\hat{f}]$ (the expected prediction over all possible training sets).

$$E[(f - \hat{f})^2] = E[(f - \bar{f} + \bar{f} - \hat{f})^2]$$

$$= (f - \bar{f})^2 + 2(f - \bar{f})E[\bar{f} - \hat{f}] + E[(\bar{f} - \hat{f})^2]$$

Since $E[\bar{f} - \hat{f}] = \bar{f} - E[\hat{f}] = 0$:

$$= \underbrace{(f - \bar{f})^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f} - \bar{f})^2]}_{\text{Variance}}$$

**Final result**:

$$\boxed{E\left[(y - \hat{f})^2\right] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2}$$

### Ridge Regression (L2 Regularization)

**Objective**:

$$\mathcal{L}_{\text{Ridge}}(w) = \frac{1}{n}\|Xw - y\|^2 + \lambda\|w\|_2^2$$

**Derivation** (set gradient to zero):

$$\nabla_w \mathcal{L} = \frac{2}{n}X^T(Xw - y) + 2\lambda w = 0$$

$$X^TXw + n\lambda w = X^Ty$$

$$(X^TX + n\lambda I)w = X^Ty$$

$$\boxed{\hat{w}_{\text{Ridge}} = (X^TX + n\lambda I)^{-1}X^Ty}$$

Note: Many references absorb the $n$ into $\lambda$, writing $(X^TX + \lambda I)^{-1}X^Ty$.

**Key property**: $(X^TX + \lambda I)$ is always invertible for $\lambda > 0$ (eigenvalues are $\sigma_i^2 + \lambda > 0$).

**Effect on solution**: If $X = U\Sigma V^T$ (SVD), then:

$$\hat{w}_{\text{Ridge}} = \sum_{j=1}^{d} \frac{\sigma_j^2}{\sigma_j^2 + \lambda} \frac{u_j^T y}{\sigma_j} v_j$$

The factor $\frac{\sigma_j^2}{\sigma_j^2 + \lambda} < 1$ *shrinks* each component, especially for small singular values.

### Lasso Regression (L1 Regularization)

**Objective**:

$$\mathcal{L}_{\text{Lasso}}(w) = \frac{1}{n}\|Xw - y\|^2 + \lambda\|w\|_1$$

**No closed-form solution** because $|w_j|$ is not differentiable at $w_j = 0$.

**Subgradient**: The subdifferential of $|w_j|$ is:

$$\partial |w_j| = \begin{cases} \{-1\} & w_j < 0 \\ [-1, 1] & w_j = 0 \\ \{+1\} & w_j > 0 \end{cases}$$

**Coordinate descent** (for orthonormal $X$):

$$\hat{w}_j = \text{sign}(\hat{w}_j^{\text{OLS}}) \max(|\hat{w}_j^{\text{OLS}}| - \lambda, 0)$$

This is the **soft-thresholding operator** — it sets small coefficients exactly to zero.

### Elastic Net

Combines L1 and L2:

$$\mathcal{L}_{\text{EN}}(w) = \frac{1}{n}\|Xw - y\|^2 + \lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$$

Gets sparsity from L1 and grouped selection from L2.

### Ridge: Bias-Variance Tradeoff

For Ridge regression:
- **Bias increases** with $\lambda$ (we constrain $w$ away from OLS solution)
- **Variance decreases** with $\lambda$ (solution is more stable)
- **Optimal $\lambda$** minimizes test error = bias$^2$ + variance

---

## Code

### NumPy From-Scratch

```python
import numpy as np

def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge regression closed-form solution.

    X: (n, d), y: (n,), lam: regularization strength
    Returns: (d,) weight vector
    """
    d = X.shape[1]
    # w = (X^T X + λI)^{-1} X^T y
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)  # (d,)


def lasso_coordinate_descent(
    X: np.ndarray,    # (n, d)
    y: np.ndarray,    # (n,)
    lam: float,
    n_steps: int = 1000,
) -> np.ndarray:
    """
    Lasso via coordinate descent.
    Returns: (d,) weight vector
    """
    n, d = X.shape
    w = np.zeros(d)

    for _ in range(n_steps):
        for j in range(d):
            # Compute residual without feature j
            r_j = y - X @ w + X[:, j] * w[j]          # (n,)
            # OLS solution for feature j alone
            rho_j = X[:, j] @ r_j / n                  # scalar
            norm_j = np.sum(X[:, j] ** 2) / n           # scalar
            # Soft thresholding
            w[j] = np.sign(rho_j) * max(abs(rho_j) - lam, 0) / norm_j

    return w


def soft_threshold(x: float, lam: float) -> float:
    """Soft thresholding operator."""
    return np.sign(x) * max(abs(x) - lam, 0)


def regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    method: str = "ridge",
) -> np.ndarray:
    """
    Compute weights for a range of lambda values.

    Returns: (len(lambdas), d) array of weight vectors
    """
    d = X.shape[1]
    weights = np.zeros((len(lambdas), d))

    for i, lam in enumerate(lambdas):
        if method == "ridge":
            weights[i] = ridge_regression(X, y, lam)
        elif method == "lasso":
            weights[i] = lasso_coordinate_descent(X, y, lam, n_steps=500)

    return weights  # (len(lambdas), d)


# --- Bias-Variance Simulation ---
def bias_variance_simulation(
    n_train: int = 50,
    n_test: int = 200,
    n_trials: int = 100,
    degrees: list = [1, 3, 5, 10, 15],
    noise_std: float = 0.5,
) -> dict:
    """
    Empirical bias-variance decomposition for polynomial regression.
    """
    # True function
    f_true = lambda x: np.sin(2 * np.pi * x)

    x_test = np.linspace(0, 1, n_test)              # (n_test,)
    y_true = f_true(x_test)                          # (n_test,)

    results = {}

    for deg in degrees:
        predictions = np.zeros((n_trials, n_test))   # (n_trials, n_test)

        for t in range(n_trials):
            # Generate training data
            x_train = np.random.rand(n_train)
            y_train = f_true(x_train) + noise_std * np.random.randn(n_train)

            # Fit polynomial
            X_train = np.vander(x_train, deg + 1, increasing=True)  # (n_train, deg+1)
            X_test = np.vander(x_test, deg + 1, increasing=True)    # (n_test, deg+1)

            w = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            predictions[t] = X_test @ w              # (n_test,)

        mean_pred = predictions.mean(axis=0)          # (n_test,)
        bias_sq = np.mean((y_true - mean_pred) ** 2)  # scalar
        variance = np.mean(predictions.var(axis=0))    # scalar

        results[deg] = {"bias_sq": bias_sq, "variance": variance}

    return results
```

### PyTorch Equivalent

```python
import torch
import torch.nn as nn

# Ridge regression via weight_decay (which implements L2 regularization)
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)  # weight_decay = λ
loss_fn = nn.MSELoss()

# Lasso requires manual L1 penalty
def train_lasso(model, X, y, lr=0.01, lam=0.1, n_epochs=1000):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        pred = model(X).squeeze()
        mse = nn.MSELoss()(pred, y)
        l1 = sum(p.abs().sum() for p in model.parameters())
        loss = mse + lam * l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Resources

- Hastie, Tibshirani, Friedman: *Elements of Statistical Learning*, Chapter 3 (free online)
- CS229 Notes: Regularization and Model Selection
- Bishop: *Pattern Recognition and Machine Learning*, Section 3.2
