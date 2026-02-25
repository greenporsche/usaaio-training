# Logistic Regression

**Prerequisites**: Linear regression, calculus (chain rule), probability (Bernoulli, MLE), NumPy
**USAAIO Relevance**: Core classification algorithm; sigmoid/softmax derivations are very common exam questions; gradient of cross-entropy loss is a must-know; coding from scratch is standard

---

## Discovery

### From Regression to Classification

Linear regression predicts continuous values in $(-\infty, +\infty)$. But what if your target is binary: spam or not spam, sick or healthy, 0 or 1?

You could threshold linear regression at 0.5, but this is fragile — the output isn't bounded between 0 and 1, so it can't represent a valid probability. We need a function that "squishes" any real number into $[0, 1]$.

Enter the **sigmoid function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$. This turns logistic regression into a probabilistic classifier.

### Socratic Warm-Up

1. If $P(y=1|x) = \sigma(w^Tx)$, what is $P(y=0|x)$?
2. Why do we use cross-entropy loss instead of MSE for classification?
3. The decision boundary of logistic regression is $w^Tx + b = 0$. What geometric shape is this?
4. How does softmax generalize sigmoid to $C > 2$ classes?

### Misconception Traps

- **"Logistic regression is for regression."** — Despite its name, it's a classification algorithm. The "regression" refers to the fact that we're fitting a linear model to the *log-odds*.
- **"The sigmoid output is a probability."** — Only if the model is well-calibrated. In practice, sigmoid outputs are often overconfident.
- **"Gradient descent for logistic regression can get stuck in local minima."** — No! The cross-entropy loss for logistic regression is *convex*, so gradient descent always finds the global minimum.

---

## Intuition

### The Sigmoid Function

```
σ(z) = 1 / (1 + e^{-z})

  1.0 ─────────────────────────●●●●●●
                             ●●
                           ●●
  0.5 ─ ─ ─ ─ ─ ─ ─ ─ ─●─ ─ ─ ─ ─ ─
                       ●●
                     ●●
  0.0 ●●●●●●─────────────────────────
     -6  -4  -2   0   2   4   6
                  z →

Properties:
  • σ(0) = 0.5
  • σ(-z) = 1 - σ(z)            ← symmetry
  • σ'(z) = σ(z)(1 - σ(z))      ← elegant derivative
  • As z → +∞, σ(z) → 1
  • As z → -∞, σ(z) → 0
```

### Log-Odds (Logit) Interpretation

Logistic regression models the **log-odds** as a linear function:

$$\log\frac{P(y=1|x)}{P(y=0|x)} = w^Tx + b$$

```
P(y=1)    Log-odds         Linear
  │         │                │
 0.99  →  +4.6   =  w^T x + b
 0.95  →  +2.9
 0.80  →  +1.4
 0.50  →   0.0   ← decision boundary
 0.20  →  -1.4
 0.05  →  -2.9
 0.01  →  -4.6
```

### Decision Boundary

The decision boundary is where $P(y=1|x) = 0.5$, i.e., $w^Tx + b = 0$.

```
  x2
   │        ○ ○
   │  ○ ○  ○○     w^Tx + b = 0
   │  ○ ○ ╱         (hyperplane)
   │ ○ ○ ╱ ● ●
   │ ○  ╱ ● ● ●
   │   ╱ ● ● ● ●
   │  ╱ ● ● ●
   │ ╱
   └──────────── x1

   ○ = class 0 (w^Tx + b < 0)
   ● = class 1 (w^Tx + b > 0)
```

---

## Math

### Maximum Likelihood Estimation

*[Reasoning required for USAAIO]*

**Setup**: Given data $(x_i, y_i)$ with $y_i \in \{0, 1\}$ and model $P(y=1|x) = \sigma(w^Tx)$.

**Step 1**: Write the likelihood for one data point.

$$P(y_i|x_i; w) = \hat{y}_i^{y_i}(1 - \hat{y}_i)^{1-y_i}$$

where $\hat{y}_i = \sigma(w^Tx_i)$.

**Step 2**: Write the log-likelihood for all data.

$$\ell(w) = \sum_{i=1}^{n}\left[y_i\log\hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

**Step 3**: The **negative log-likelihood** (NLL) is the **binary cross-entropy loss**:

$$\mathcal{L}(w) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log\hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i)\right]$$

### Gradient Derivation

*[Reasoning required for USAAIO]*

**Step 1**: Compute $\frac{\partial \mathcal{L}}{\partial \hat{y}_i}$.

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right)$$

**Step 2**: Use the chain rule with $\hat{y}_i = \sigma(z_i)$ where $z_i = w^Tx_i$.

$$\frac{\partial \hat{y}_i}{\partial z_i} = \sigma(z_i)(1 - \sigma(z_i)) = \hat{y}_i(1 - \hat{y}_i)$$

**Step 3**: Combine.

$$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{1}{n}\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right)\hat{y}_i(1-\hat{y}_i)$$

$$= -\frac{1}{n}\left(y_i(1-\hat{y}_i) - (1-y_i)\hat{y}_i\right)$$

$$= -\frac{1}{n}(y_i - \hat{y}_i) = \frac{1}{n}(\hat{y}_i - y_i)$$

**Step 4**: Since $z_i = w^Tx_i$, we have $\frac{\partial z_i}{\partial w} = x_i$.

$$\nabla_w \mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_i = \frac{1}{n}X^T(\hat{y} - y)$$

**Beautiful result**: The gradient has the *exact same form* as linear regression — just replace the linear prediction with the sigmoid prediction.

$$\boxed{\nabla_w \mathcal{L} = \frac{1}{n}X^T(\sigma(Xw) - y)}$$

### Convexity of Cross-Entropy

*[Reasoning not required but useful for USAAIO]*

The Hessian of the cross-entropy loss is:

$$H = \frac{1}{n}X^TSX$$

where $S = \text{diag}(\hat{y}_i(1-\hat{y}_i))$ is a diagonal matrix with positive entries (since $0 < \hat{y}_i < 1$).

For any vector $v$: $v^THv = \frac{1}{n}(Xv)^TS(Xv) \geq 0$

So $H$ is PSD, meaning $\mathcal{L}$ is convex. Gradient descent will find the global minimum.

### Softmax Extension (Multiclass)

For $C$ classes, replace sigmoid with softmax:

$$P(y = c | x) = \frac{e^{w_c^Tx}}{\sum_{j=1}^{C} e^{w_j^Tx}} = \text{softmax}(Wx)_c$$

where $W \in \mathbb{R}^{C \times d}$ (one weight vector per class).

**Cross-entropy loss** (multiclass):

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C} y_{ic}\log\hat{y}_{ic}$$

where $y_{ic}$ is the one-hot encoding.

**Gradient w.r.t. logits**: $\nabla_z \mathcal{L} = \hat{y} - y$ (same elegant form!)

**Numerical stability**: Use the log-sum-exp trick:

$$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \quad m = \max_j z_j$$

---

## Code

### NumPy From-Scratch

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid. z: any shape → same shape"""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    y_true: (n,) binary labels
    y_pred: (n,) predicted probabilities
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def logistic_regression_gd(
    X: np.ndarray,     # (n, d) — include bias column
    y: np.ndarray,     # (n,) binary labels
    lr: float = 0.1,
    n_steps: int = 1000,
) -> np.ndarray:
    """
    Logistic regression via gradient descent.
    Returns: (d,) weight vector
    """
    n, d = X.shape
    w = np.zeros(d)                              # (d,)

    for step in range(n_steps):
        z = X @ w                                 # (n,)
        y_hat = sigmoid(z)                        # (n,)
        grad = (1 / n) * X.T @ (y_hat - y)       # (d, n) @ (n,) → (d,)
        w = w - lr * grad                         # (d,)

    return w


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """X: (n, d), w: (d,) → (n,) probabilities"""
    return sigmoid(X @ w)


def predict(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """X: (n, d), w: (d,) → (n,) binary predictions"""
    return (predict_proba(X, w) >= threshold).astype(int)


# --- Softmax / Multiclass ---

def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Z: (n, C) logits → (n, C) probabilities
    """
    Z_shifted = Z - Z.max(axis=1, keepdims=True)  # (n, C) log-sum-exp trick
    exp_Z = np.exp(Z_shifted)                       # (n, C)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True) # (n, C)


def cross_entropy_loss(Y_true: np.ndarray, Y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Y_true: (n, C) one-hot
    Y_pred: (n, C) predicted probabilities
    """
    Y_pred = np.clip(Y_pred, eps, 1.0)
    return -float(np.mean(np.sum(Y_true * np.log(Y_pred), axis=1)))


def softmax_regression_gd(
    X: np.ndarray,      # (n, d)
    Y: np.ndarray,      # (n, C) one-hot
    lr: float = 0.1,
    n_steps: int = 1000,
) -> np.ndarray:
    """
    Softmax regression via gradient descent.
    Returns: (d, C) weight matrix
    """
    n, d = X.shape
    C = Y.shape[1]
    W = np.zeros((d, C))                          # (d, C)

    for step in range(n_steps):
        Z = X @ W                                  # (n, C)
        Y_hat = softmax(Z)                         # (n, C)
        grad = (1 / n) * X.T @ (Y_hat - Y)        # (d, n) @ (n, C) → (d, C)
        W = W - lr * grad                          # (d, C)

    return W


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 200

    # Generate 2D data
    X_pos = np.random.randn(n // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n // 2, 2) + np.array([-2, -2])
    X_raw = np.vstack([X_pos, X_neg])                # (200, 2)
    X = np.hstack([np.ones((n, 1)), X_raw])           # (200, 3) — bias trick
    y = np.array([1] * (n // 2) + [0] * (n // 2))    # (200,)

    w = logistic_regression_gd(X, y, lr=0.1, n_steps=1000)
    y_pred = predict(X, w)
    accuracy = np.mean(y_pred == y)
    loss = binary_cross_entropy(y, predict_proba(X, w))

    print(f"Weights: {w.round(4)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"BCE Loss: {loss:.4f}")
```

### PyTorch Equivalent

```python
import torch
import torch.nn as nn

# Binary logistic regression
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()  # combines sigmoid + BCE

X = torch.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).float()

for epoch in range(1000):
    logits = model(X).squeeze()          # (200,)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Multiclass with softmax
model_mc = nn.Linear(2, 5)  # 5 classes
loss_fn_mc = nn.CrossEntropyLoss()  # combines softmax + CE

# Note: CrossEntropyLoss expects raw logits, NOT softmax output
# It also expects integer labels, NOT one-hot
```

### Shape Annotation Summary

```
Binary:
  X:     (n, d)
  w:     (d,)
  z:     (n,)    = X @ w
  ŷ:     (n,)    = σ(z)
  grad:  (d,)    = X^T @ (ŷ - y) / n

Multiclass:
  X:     (n, d)
  W:     (d, C)
  Z:     (n, C)  = X @ W
  Ŷ:     (n, C)  = softmax(Z)
  Y:     (n, C)  one-hot
  grad:  (d, C)  = X^T @ (Ŷ - Y) / n
```

---

## Resources

- CS229 Lecture Notes: Classification and Logistic Regression
- Bishop: *Pattern Recognition and Machine Learning*, Section 4.3
- Andrew Ng's Machine Learning Specialization, Course 1: Logistic Regression
