# k-Nearest Neighbors & Cross-Validation

**Prerequisites**: Distance metrics (Euclidean, Manhattan), basic probability, NumPy
**USAAIO Relevance**: kNN is a common baseline; cross-validation appears in nearly every ML problem on the exam; curse of dimensionality is a popular theory question

---

## Discovery

### The Simplest Classifier That Works

Forget about learning weights, computing gradients, or optimizing objectives. What if you just looked at the nearest labeled examples and copied their answer?

This is **k-Nearest Neighbors** — a "lazy" algorithm that stores all training data and makes predictions by finding the $k$ closest points. Despite its simplicity, kNN can approximate *any* decision boundary given enough data (it's a universal approximator).

But how do you choose $k$? And how do you evaluate a model without cheating by testing on training data? Enter **cross-validation**.

### Socratic Warm-Up

1. What happens to the decision boundary as $k \to 1$? As $k \to n$?
2. If you double the number of features but keep $n$ fixed, what happens to kNN's performance? Why?
3. Why can't you just evaluate on the training set to pick the best $k$?

### Misconception Traps

- **"kNN is too simple to be useful."** — In high-data, low-dimension regimes, kNN can match or beat complex models. It's also the backbone of many recommendation systems.
- **"Just use Euclidean distance."** — In high dimensions, all distances converge to the same value (curse of dimensionality). Feature scaling and distance choice matter enormously.
- **"Leave-one-out CV is always best."** — LOOCV has low bias but high variance and is computationally expensive. 5-fold or 10-fold is usually better in practice.

---

## Intuition

### kNN Decision Boundary

```
k = 1 (very complex)         k = 5 (smoother)           k = n (trivial)
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│● ○     ○ ●  ○   │       │● ○     ○ ●  ○   │       │                  │
│ ╲╱╲   ╱╲╱╲╱    │       │  ╲   ╱    ╲     │       │                  │
│○  ● ● ○  ● ○   │       │○  ╲ ╱  ○   ╲    │       │  All predict     │
│ ╱╲╱╲╱╲╱╲╱╲╱╲   │       │   ─╱────────╲─  │       │  majority class  │
│●  ○ ●  ○ ●  ○  │       │●  ╱  ●  ○ ●  ╲  │       │                  │
│╱╲╱╲╱╲╱╲╱╲╱╲╱╲  │       │  ╱         ╲  │       │                  │
└──────────────────┘       └──────────────────┘       └──────────────────┘
 High variance               Balanced                   High bias
 Low bias                                               Low variance
```

### Curse of Dimensionality

In $d$ dimensions, the volume of a unit hypercube is 1, but the volume of the inscribed hypersphere $\to 0$ as $d \to \infty$.

To capture a fraction $f$ of the data with a local neighborhood, the edge length of that neighborhood must be:

$$\ell = f^{1/d}$$

```
d = 1:  f=0.1 → ℓ = 0.10  (10% of range)
d = 2:  f=0.1 → ℓ = 0.32  (32% of each axis)
d = 10: f=0.1 → ℓ = 0.79  (79% of each axis!)
d = 100: f=0.1 → ℓ = 0.977 (97.7% — virtually everything)
```

In high dimensions, "local" neighborhoods must span nearly the entire space, making kNN useless.

### Cross-Validation Diagram

```
5-Fold Cross-Validation:

Data: [████████████████████████████████████████]

Fold 1: [TEST ][TRAIN][TRAIN][TRAIN][TRAIN]  → error₁
Fold 2: [TRAIN][TEST ][TRAIN][TRAIN][TRAIN]  → error₂
Fold 3: [TRAIN][TRAIN][TEST ][TRAIN][TRAIN]  → error₃
Fold 4: [TRAIN][TRAIN][TRAIN][TEST ][TRAIN]  → error₄
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST ]  → error₅

CV error = (error₁ + error₂ + error₃ + error₄ + error₅) / 5
```

Every data point appears in exactly one test fold and in $k-1$ training folds.

---

## Math

### kNN Formalization

**Classification** (majority vote):

$$\hat{y}(x) = \text{mode}\{y_j : x_j \in N_k(x)\}$$

where $N_k(x)$ is the set of $k$ nearest neighbors of $x$ in the training set.

**Regression** (average):

$$\hat{y}(x) = \frac{1}{k}\sum_{x_j \in N_k(x)} y_j$$

**Weighted kNN** (inverse-distance weighting):

$$\hat{y}(x) = \frac{\sum_{j \in N_k(x)} w_j y_j}{\sum_{j \in N_k(x)} w_j}, \quad w_j = \frac{1}{d(x, x_j)}$$

### Distance Metrics

| Metric | Formula | Notes |
|---|---|---|
| Euclidean (L2) | $d(x, x') = \sqrt{\sum_i (x_i - x_i')^2}$ | Standard, rotation-invariant |
| Manhattan (L1) | $d(x, x') = \sum_i |x_i - x_i'|$ | Axis-aligned, robust |
| Minkowski (Lp) | $d(x, x') = \left(\sum_i |x_i - x_i'|^p\right)^{1/p}$ | Generalizes L1 and L2 |
| Cosine | $d(x, x') = 1 - \frac{x^Tx'}{\|x\|\|x'\|}$ | Direction only, ignores magnitude |

### Cross-Validation Theory

*[Reasoning required for USAAIO]*

**k-fold CV estimator**:

$$\hat{R}_{CV} = \frac{1}{k}\sum_{i=1}^{k} \mathcal{L}(f^{(-i)}, D_i)$$

where $f^{(-i)}$ is trained on all data except fold $i$, and $D_i$ is the $i$-th fold.

**Properties**:
- Each model is trained on $\frac{k-1}{k} \cdot n$ data points
- **Bias**: k-fold CV has pessimistic bias because we train on less data than available. LOOCV has least bias (trains on $n-1$ points).
- **Variance**: LOOCV has high variance because the $n$ training sets overlap heavily (differ by only 1 point), making the error estimates highly correlated.

**LOOCV for linear regression** has a shortcut:

$$\text{CV}_n = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$$

where $h_{ii}$ is the $i$-th diagonal of the hat matrix $H = X(X^TX)^{-1}X^T$. This computes LOOCV in $O(n d^2)$ instead of $O(n^2 d^2)$.

### Curse of Dimensionality — Formal

For data uniformly distributed in $[0, 1]^d$, the expected distance to the nearest neighbor is:

$$E[d_{NN}] \approx \frac{1}{n^{1/d}} \cdot \frac{\Gamma(1 + 1/d)}{d}$$

As $d \to \infty$, $E[d_{NN}] \to 1$ regardless of $n$. All points become equidistant.

The ratio of max to min distance also converges to 1:

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0 \text{ as } d \to \infty$$

---

## Code

### NumPy From-Scratch

```python
import numpy as np

def knn_classify(
    X_train: np.ndarray,   # (n, d)
    y_train: np.ndarray,   # (n,) integer labels
    X_test: np.ndarray,    # (m, d)
    k: int = 5,
) -> np.ndarray:
    """
    k-Nearest Neighbors classification (no loops over test points).
    Returns: (m,) predicted labels
    """
    # Compute pairwise distances: (m, n)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2a^Tb
    sq_test = np.sum(X_test ** 2, axis=1, keepdims=True)   # (m, 1)
    sq_train = np.sum(X_train ** 2, axis=1, keepdims=True) # (n, 1)
    dists = sq_test + sq_train.T - 2 * X_test @ X_train.T  # (m, n)

    # Find k nearest neighbors
    knn_indices = np.argpartition(dists, k, axis=1)[:, :k]  # (m, k)
    knn_labels = y_train[knn_indices]                        # (m, k)

    # Majority vote (vectorized for integer labels)
    n_classes = int(y_train.max()) + 1
    # One-hot encode and sum
    votes = np.zeros((X_test.shape[0], n_classes))           # (m, C)
    for c in range(n_classes):
        votes[:, c] = np.sum(knn_labels == c, axis=1)

    return np.argmax(votes, axis=1)                          # (m,)


def knn_regress(
    X_train: np.ndarray,   # (n, d)
    y_train: np.ndarray,   # (n,)
    X_test: np.ndarray,    # (m, d)
    k: int = 5,
) -> np.ndarray:
    """
    k-Nearest Neighbors regression.
    Returns: (m,) predicted values
    """
    sq_test = np.sum(X_test ** 2, axis=1, keepdims=True)
    sq_train = np.sum(X_train ** 2, axis=1, keepdims=True)
    dists = sq_test + sq_train.T - 2 * X_test @ X_train.T   # (m, n)

    knn_indices = np.argpartition(dists, k, axis=1)[:, :k]   # (m, k)
    knn_values = y_train[knn_indices]                         # (m, k)

    return np.mean(knn_values, axis=1)                        # (m,)


def k_fold_cv(
    X: np.ndarray,         # (n, d)
    y: np.ndarray,         # (n,)
    model_fn,              # callable(X_train, y_train) → model
    predict_fn,            # callable(model, X_test) → y_pred
    loss_fn,               # callable(y_true, y_pred) → scalar
    k: int = 5,
) -> float:
    """
    k-fold cross-validation.
    Returns: average loss across folds
    """
    n = len(y)
    indices = np.random.permutation(n)
    fold_size = n // k
    losses = []

    for i in range(k):
        # Split
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                     indices[(i + 1) * fold_size:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train and evaluate
        model = model_fn(X_train, y_train)
        y_pred = predict_fn(model, X_test)
        losses.append(loss_fn(y_test, y_pred))

    return float(np.mean(losses))


def loocv_linear_regression(X: np.ndarray, y: np.ndarray) -> float:
    """
    LOOCV for linear regression using the hat matrix shortcut.

    CV = (1/n) Σ (y_i - ŷ_i)^2 / (1 - h_ii)^2

    Returns: LOOCV error
    """
    H = X @ np.linalg.solve(X.T @ X, X.T)   # (n, n) hat matrix
    y_hat = H @ y                              # (n,)
    residuals = y - y_hat                      # (n,)
    h_diag = np.diag(H)                        # (n,)

    loocv = np.mean((residuals / (1 - h_diag)) ** 2)
    return float(loocv)


# --- Demo: Select k via CV ---
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)

    best_k, best_loss = 1, float("inf")
    for k_val in [1, 3, 5, 7, 11, 15, 21]:
        loss = k_fold_cv(
            X, y,
            model_fn=lambda Xtr, ytr, k=k_val: (Xtr, ytr, k),
            predict_fn=lambda m, Xte: knn_classify(m[0], m[1], Xte, m[2]),
            loss_fn=lambda yt, yp: np.mean(yt != yp),  # error rate
            k=5,
        )
        print(f"k={k_val:3d}: CV error = {loss:.4f}")
        if loss < best_loss:
            best_k, best_loss = k_val, loss

    print(f"\nBest k: {best_k} (CV error: {best_loss:.4f})")
```

### PyTorch Equivalent

```python
import torch

def knn_classify_torch(
    X_train: torch.Tensor,  # (n, d)
    y_train: torch.Tensor,  # (n,) long
    X_test: torch.Tensor,   # (m, d)
    k: int = 5,
) -> torch.Tensor:
    """Returns: (m,) predicted labels"""
    dists = torch.cdist(X_test, X_train, p=2)   # (m, n)
    _, knn_idx = dists.topk(k, largest=False)    # (m, k)
    knn_labels = y_train[knn_idx]                 # (m, k)

    # Majority vote
    return knn_labels.mode(dim=1).values          # (m,)
```

---

## Resources

- Hastie, Tibshirani, Friedman: *Elements of Statistical Learning*, Chapters 7 (CV) and 13 (kNN)
- CS229 Notes: Model Selection and Cross-Validation
- Beyer et al. (1999): "When Is Nearest Neighbor Meaningful?" — curse of dimensionality
