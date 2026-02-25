# Loss Functions

**Prerequisites**: Calculus (derivatives, convexity), linear regression, logistic regression
**USAAIO Relevance**: Loss function properties (convexity, differentiability) are common theory questions; connecting loss functions to optimization landscapes; hinge loss and SVM geometry

---

## Discovery

### Choosing the Right Objective

Every machine learning model optimizes *something*. The choice of what to optimize — the **loss function** — determines what the model learns. Choose MSE and you'll be sensitive to outliers. Choose MAE and you won't, but you lose smoothness. Choose cross-entropy for classification and your model outputs calibrated probabilities. Choose hinge loss and you get maximum-margin separation.

The loss function is arguably the most important design decision in ML. The algorithm is secondary — it's just the tool we use to minimize the loss.

### Socratic Warm-Up

1. MSE penalizes a prediction that's off by 10 a hundred times more than one that's off by 1. Is this always desirable?
2. Why is cross-entropy preferred over MSE for classification, even though both "work"?
3. The hinge loss is zero when the prediction is correct with margin $\geq 1$. What does this mean geometrically?
4. Can a non-convex loss function still have a unique global minimum?

### Misconception Traps

- **"All loss functions have the same minimizer."** — Different losses can lead to *different* optimal models, even on the same data.
- **"Convex loss = easy optimization."** — Convexity guarantees no local minima, but the loss can still be ill-conditioned (slow convergence).
- **"Non-differentiable loss functions can't be optimized."** — Subgradient methods and proximal operators handle non-smooth losses.

---

## Intuition

### Loss Function Gallery

```
MSE: (y - ŷ)²                   MAE: |y - ŷ|
  Loss                             Loss
  │      ╱╲                        │     ╱╲
  │     ╱  ╲                       │    ╱  ╲
  │    ╱    ╲                      │   ╱    ╲
  │   ╱      ╲                     │  ╱      ╲
  │  ╱        ╲                    │ ╱        ╲
  │ ╱          ╲                   │╱          ╲
  └──────●──────── r               └──────●──────── r
       r = 0                            r = 0
  Smooth, sensitive               Not smooth at 0,
  to outliers                     robust to outliers


Huber Loss:                      Cross-Entropy: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  Loss                             Loss
  │     ╱╲                         │╲          ╱│
  │    ╱  ╲                        │ ╲        ╱ │
  │   ╱    ╲  ← linear            │  ╲      ╱  │
  │  ╱╲    ╱╲  for |r|>δ          │   ╲    ╱   │
  │ ╱  ╲  ╱  ╲                    │    ╲  ╱    │
  │╱    ╲╱    ╲ ← quadratic       │     ╲╱     │
  └──────●──────── r               └──────●──────── ŷ
       r = 0    for |r|≤δ              correct
  Best of both worlds              → ∞ as ŷ → wrong answer


Hinge Loss: max(0, 1 - y·f(x))
  Loss
  │╲
  │ ╲
  │  ╲
  │   ╲
  │    ╲
  │     ╲_______________
  └────────●──────────── y·f(x)
         1.0
  Zero loss when correctly
  classified with margin ≥ 1
```

### Convexity Matters

```
Convex loss:                     Non-convex loss:
  │     ╱╲                        │  ╱╲   ╱╲
  │    ╱  ╲                       │ ╱  ╲ ╱  ╲
  │   ╱    ╲                      │╱    ╲╱    ╲
  │  ╱      ╲                     │      local
  │ ╱        ╲                    │      minima
  │╱    ●     ╲                   │  ●?   ●?   ●?
  └──── global ──── w             └──────────────── w
       minimum                    Which is global?

Gradient descent → always finds    Gradient descent → may get
the global minimum                 stuck in local minima
```

---

## Math

### MSE (Mean Squared Error)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Gradient**: $\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$

**Properties**:
- Convex (Hessian is $\frac{2}{n}I \succ 0$)
- Differentiable everywhere
- Penalizes large errors quadratically (sensitive to outliers)
- Optimal predictor: $E[Y|X]$ (conditional mean)

### MAE (Mean Absolute Error)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Subgradient**: $\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{n}\text{sign}(\hat{y}_i - y_i)$

**Properties**:
- Convex but not differentiable at $\hat{y}_i = y_i$
- Robust to outliers (gradient is constant regardless of error magnitude)
- Optimal predictor: $\text{median}(Y|X)$ (conditional median)

### Huber Loss

$$\mathcal{L}_{\text{Huber}}(r) = \begin{cases} \frac{1}{2}r^2 & \text{if } |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & \text{if } |r| > \delta \end{cases}$$

where $r = y - \hat{y}$ is the residual.

**Properties**:
- Convex and differentiable everywhere (including at $|r| = \delta$)
- Behaves like MSE for small errors, MAE for large errors
- $\delta$ controls the transition point

**Proof that Huber is differentiable at $|r| = \delta$**:

From the left: $\frac{d}{dr}\left(\frac{1}{2}r^2\right)\bigg|_{r=\delta} = \delta$

From the right: $\frac{d}{dr}\left(\delta r - \frac{1}{2}\delta^2\right)\bigg|_{r=\delta} = \delta$ ✓

### Binary Cross-Entropy (Log Loss)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Gradient w.r.t. logit $z$** (where $\hat{y} = \sigma(z)$):

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{1}{n}(\hat{y}_i - y_i)$$

**Properties**:
- Convex in the logits $z$ (proven in Topic 05)
- $\to \infty$ as prediction approaches the wrong answer
- Derived from maximum likelihood estimation
- Optimal predictor: $P(Y=1|X)$ (true conditional probability)

### Why Cross-Entropy, Not MSE, for Classification?

*[Reasoning required for USAAIO]*

Consider $\hat{y} = \sigma(z)$ and the true label $y = 1$.

**MSE gradient w.r.t. $z$**:

$$\frac{\partial}{\partial z}(\hat{y} - 1)^2 = 2(\hat{y} - 1)\hat{y}(1-\hat{y})$$

When $\hat{y} \approx 0$ (very wrong): gradient $\approx 2(-1)(0)(1) = 0$ — **vanishing gradient!**

**Cross-entropy gradient w.r.t. $z$**:

$$\frac{\partial}{\partial z}\left[-\log\hat{y}\right] = \hat{y} - 1$$

When $\hat{y} \approx 0$: gradient $\approx -1$ — **strong corrective signal!**

Cross-entropy provides a strong gradient even when the model is very wrong, enabling faster and more reliable learning.

### Hinge Loss (SVM)

$$\mathcal{L}_{\text{hinge}} = \frac{1}{n}\sum_{i=1}^{n}\max(0, 1 - y_i \cdot f(x_i))$$

where $y_i \in \{-1, +1\}$ and $f(x_i) = w^Tx_i + b$.

**Subgradient**:

$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{1}{n}\sum_{i: y_i f(x_i) < 1} y_i x_i$$

**Properties**:
- Convex but not differentiable at $y_i f(x_i) = 1$
- Zero loss when classified correctly with margin $\geq 1$
- Only "support vectors" (points with $y_i f(x_i) \leq 1$) contribute to the gradient

### Comparison Table

| Loss | Formula | Gradient | Convex | Smooth | Optimal Predictor |
|---|---|---|---|---|---|
| MSE | $(y-\hat{y})^2$ | $2(\hat{y}-y)$ | Yes | Yes | $E[Y|X]$ |
| MAE | $|y-\hat{y}|$ | $\text{sign}(\hat{y}-y)$ | Yes | No | $\text{med}(Y|X)$ |
| Huber | hybrid | hybrid | Yes | Yes | between mean/median |
| BCE | $-y\log\hat{y} - \ldots$ | $\hat{y}-y$ | Yes* | Yes | $P(Y|X)$ |
| Hinge | $\max(0, 1-yf)$ | $-y \cdot \mathbb{1}[yf<1]$ | Yes | No | — |

*Convex in logits, not in probabilities.

### Convexity Proofs

*[Reasoning required for USAAIO]*

**Claim**: MSE is convex.

**Proof**: $\mathcal{L}(w) = \frac{1}{n}\|Xw - y\|^2$. The Hessian is $H = \frac{2}{n}X^TX$, which is PSD (for any $v$, $v^TX^TXv = \|Xv\|^2 \geq 0$). A function with PSD Hessian everywhere is convex. $\square$

**Claim**: Hinge loss is convex.

**Proof**: $\max(0, 1 - y_i w^Tx_i)$ is the maximum of two convex functions ($0$ and $1 - y_i w^Tx_i$, the latter being affine and therefore convex). The max of convex functions is convex. The sum of convex functions is convex. $\square$

---

## Code

### NumPy From-Scratch

```python
import numpy as np

# --- Regression Losses ---

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(n,), (n,) → scalar"""
    return float(np.mean((y_true - y_pred) ** 2))

def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(n,), (n,) → scalar"""
    return float(np.mean(np.abs(y_true - y_pred)))

def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """(n,), (n,) → scalar"""
    r = y_true - y_pred                            # (n,)
    return float(np.mean(
        np.where(np.abs(r) <= delta,
                 0.5 * r ** 2,
                 delta * np.abs(r) - 0.5 * delta ** 2)
    ))


# --- Classification Losses ---

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    y_true: (n,) binary labels {0, 1}
    y_pred: (n,) predicted probabilities
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

def hinge_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    y_true: (n,) labels in {-1, +1}
    scores: (n,) raw scores f(x)
    """
    return float(np.mean(np.maximum(0, 1 - y_true * scores)))

def multiclass_cross_entropy(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Y_true: (n, C) one-hot
    Y_pred: (n, C) predicted probabilities
    """
    Y_pred = np.clip(Y_pred, eps, 1.0)
    return -float(np.mean(np.sum(Y_true * np.log(Y_pred), axis=1)))


# --- Gradients ---

def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Gradient w.r.t. y_pred. (n,) → (n,)"""
    return (2 / len(y_true)) * (y_pred - y_true)

def bce_gradient_wrt_logit(y_true: np.ndarray, y_pred_proba: np.ndarray) -> np.ndarray:
    """Gradient of BCE w.r.t. logit z (where y_pred = sigmoid(z)). (n,) → (n,)"""
    return (1 / len(y_true)) * (y_pred_proba - y_true)

def hinge_subgradient(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Subgradient w.r.t. scores. y_true in {-1, +1}. (n,) → (n,)"""
    return (1 / len(y_true)) * np.where(y_true * scores < 1, -y_true, 0)


# --- Convexity Check ---

def check_convexity_numerical(loss_fn, w_range: np.ndarray, X: np.ndarray, y: np.ndarray) -> bool:
    """
    Numerical check: f(λw1 + (1-λ)w2) ≤ λf(w1) + (1-λ)f(w2)
    for random w1, w2 and λ ∈ (0, 1).
    """
    d = X.shape[1]
    n_tests = 100
    for _ in range(n_tests):
        w1, w2 = np.random.randn(d), np.random.randn(d)
        lam = np.random.rand()
        w_mid = lam * w1 + (1 - lam) * w2

        f_mid = loss_fn(y, X @ w_mid)
        f_convex = lam * loss_fn(y, X @ w1) + (1 - lam) * loss_fn(y, X @ w2)

        if f_mid > f_convex + 1e-10:
            return False
    return True


# --- Demo ---
if __name__ == "__main__":
    # Compare loss functions on regression
    y_true = np.array([1.0, 2.0, 3.0, 100.0])  # note outlier
    y_pred = np.array([1.1, 2.2, 2.8, 4.0])

    print("Regression losses (with outlier y=100, ŷ=4):")
    print(f"  MSE:   {mse_loss(y_true, y_pred):.2f}")    # dominated by outlier
    print(f"  MAE:   {mae_loss(y_true, y_pred):.2f}")    # more robust
    print(f"  Huber: {huber_loss(y_true, y_pred):.2f}")  # in between

    # Classification losses
    y_bin = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.7, 0.3, 0.1])
    y_svm = np.array([1, 1, -1, -1])
    scores = np.array([2.0, 0.5, -0.3, -1.5])

    print(f"\nBCE:   {binary_cross_entropy(y_bin, y_prob):.4f}")
    print(f"Hinge: {hinge_loss(y_svm, scores):.4f}")
```

### PyTorch Equivalent

```python
import torch
import torch.nn.functional as F

y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.1, 2.2, 2.8])

# Regression losses
mse = F.mse_loss(y_pred, y_true)
mae = F.l1_loss(y_pred, y_true)
huber = F.huber_loss(y_pred, y_true, delta=1.0)

# Classification losses (expect logits, not probabilities)
logits = torch.tensor([2.0, -1.0, 0.5])
labels = torch.tensor([1.0, 0.0, 1.0])

bce = F.binary_cross_entropy_with_logits(logits, labels)

# Hinge loss (SVM)
y_svm = torch.tensor([1.0, -1.0, 1.0])  # {-1, +1}
hinge = torch.mean(torch.clamp(1 - y_svm * logits, min=0))
```

---

## Resources

- Rosasco et al. (2004): "Are Loss Functions All the Same?"
- CS229 Notes: Generalized Linear Models
- Bishop: *Pattern Recognition and Machine Learning*, Chapter 4 (Loss functions for classification)
