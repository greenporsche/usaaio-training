# Support Vector Machines (SVM)

**Prerequisites**: Linear algebra (dot products, norms), calculus (Lagrange multipliers), optimization (constrained)
**USAAIO Relevance**: **HIGH** — SVM concepts (margin, support vectors, kernel trick) are tested in Round 1 and implementation appears in Round 2. Know the primal and dual formulations.

---

## Discovery

It's 1963, and you're Vladimir Vapnik at the Institute of Control Sciences in Moscow. You're thinking about the simplest classification problem: given points in 2D that belong to two classes, find a line that separates them.

But wait — there are infinitely many separating lines:

```
Class ●: positive          Class ○: negative

     ● ●     All these lines separate the classes:
    ● ●
   ● ●        Line A /     Line B |     Line C \
              ────/────    ────|────    ────\────
     ○ ○          /             |             \
    ○ ○
   ○ ○

Which line is "best"?
```

**Your question**: Among all separating lines, which one will generalize best to unseen data?

Your intuition: the line that is **as far as possible from both classes** — the one with the **maximum margin**. A thin margin means you're barely separating the classes, and slight noise could flip predictions. A wide margin means you have a comfortable buffer.

```
Narrow margin (fragile):     Wide margin (robust):

  ● ● |                      ● ●    |
 ● ●  | ○ ○                 ● ●     |     ○ ○
  ● ● | ○ ○                  ● ●    |    ○ ○
      |                       ←margin width→
```

**Socratic questions**:
- How do you measure the "width" of the margin mathematically?
- What if the data is NOT linearly separable? Should you give up, or allow some violations?
- What if the boundary should be curved, not straight? Can you somehow work in a higher-dimensional space where a linear boundary suffices?

---

## Intuition

### The Maximum Margin Classifier

Given data $(x_i, y_i)$ with $y_i \in \{-1, +1\}$, we want to find a hyperplane $w^T x + b = 0$ that separates the classes with maximum margin.

```
      ← margin = 2/||w|| →

  ●    ‖  w^Tx+b = +1     (positive boundary)
 ●  ●  ‖
  ●    ‖  w^Tx+b = 0      (decision boundary)
       ‖
  ○  ○ ‖  w^Tx+b = -1     (negative boundary)
   ○   ‖
  ○    ‖

  ☆ = support vectors (points ON the margin boundaries)
```

The **margin** is the distance between the two boundary hyperplanes $w^T x + b = +1$ and $w^T x + b = -1$. This distance is $\frac{2}{\|w\|}$.

**Maximizing** $\frac{2}{\|w\|}$ is equivalent to **minimizing** $\frac{1}{2}\|w\|^2$ (easier to optimize — convex quadratic!).

### Support Vectors — The Key Players

**Support vectors** are the training points that lie exactly on the margin boundary ($w^T x_i + b = \pm 1$). They are the "hardest" points — the closest to the decision boundary.

**Critical insight**: The optimal hyperplane depends ONLY on the support vectors. You could remove all other points and get the same boundary. This makes SVMs memory-efficient and robust to outliers far from the boundary.

### Soft Margin — Handling Noise

Real data is rarely perfectly separable. The **soft margin** SVM allows some points to violate the margin, controlled by penalty parameter $C$:

```
C = large (hard margin):         C = small (soft margin):

  ● ● |                          ●  ●  |
 ● ●  |  ○ ○                    ●  ●   |   ○ ○
  ● ●✗|  ○ ○                    ● ✗● ✗ | ✗○ ○
      |                             ← wider margin, some errors →
(few violations, thin margin)    (more violations, wide margin)
```

Each violation incurs a slack variable $\xi_i$. The objective balances margin width vs. total violation: $\frac{1}{2}\|w\|^2 + C\sum_i \xi_i$.

### The Kernel Trick — Non-Linear Boundaries

What if the data is not linearly separable in the original space?

```
Original space (not separable):     After mapping to higher dim:

    ○  ○  ○                          (using φ(x) = [x₁², √2·x₁x₂, x₂²])
  ○  ●  ●  ○
  ○  ●  ●  ○                        Now linearly separable!
    ○  ○  ○
```

**Key insight**: You don't need to compute $\phi(x)$ explicitly. The dual formulation only needs dot products $\phi(x_i)^T \phi(x_j)$, which can be computed directly as a **kernel function** $K(x_i, x_j)$.

| Kernel | Formula | Effect |
|--------|---------|--------|
| Linear | $x^T z$ | Straight line/hyperplane |
| Polynomial | $(x^T z + c)^d$ | Polynomial boundary of degree $d$ |
| RBF (Gaussian) | $\exp(-\gamma\|x-z\|^2)$ | Flexible, smooth boundary |

The RBF kernel implicitly maps to an **infinite-dimensional** space! And yet we never compute coordinates in that space — only dot products through the kernel.

### Hinge Loss Perspective

The SVM objective can be rewritten as an **unconstrained** optimization with hinge loss:

$$\min_w \frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1 - y_i(w^T x_i + b))$$

This is just: **L2 regularization + hinge loss**. Compare to logistic regression: **L2 regularization + logistic loss**.

```
Loss
  |  ╲               Hinge loss: max(0, 1-yf)
  |   ╲  .....       Logistic: log(1+exp(-yf))
  |    ╲.
  |     ·╲.
  |     ╲  ·...___
  +──────╲────────── y·f(x)
  0      1
```

Hinge loss is **exactly zero** for points beyond the margin ($y_i f(x_i) > 1$). This sparsity gives rise to support vectors.

### Failure Cases

- **Slow on large datasets**: Solving the QP scales $O(n^2)$ to $O(n^3)$ with $n$ data points. For $n > 100k$, consider linear SVM or stochastic gradient descent.
- **Sensitive to feature scaling**: Margin depends on distance, so features must be on similar scales. Always standardize.
- **No probabilistic output**: SVMs give distances to the boundary, not probabilities. Use Platt scaling for probability estimates.
- **Kernel choice is crucial**: Wrong kernel = wrong bias. RBF is often a safe default.

---

## Math

### Hard Margin SVM (Primal)

*Reasoning required for USAAIO.*

For linearly separable data $(x_i, y_i)$ with $y_i \in \{-1, +1\}$:

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^T x_i + b) \geq 1, \quad i = 1, \ldots, n$$

The constraint $y_i(w^T x_i + b) \geq 1$ means every point is on the correct side of its margin boundary.

**Margin width**: The distance from $w^T x + b = 1$ to $w^T x + b = -1$ is $\frac{2}{\|w\|}$.

**Proof of margin width**: A point on the positive boundary satisfies $w^T x_+ + b = 1$. A point on the negative boundary satisfies $w^T x_- + b = -1$. The distance along the normal direction $w/\|w\|$ is:

$$\frac{w^T(x_+ - x_-)}{||w||} = \frac{(1-b) - (-1-b)}{\|w\|} = \frac{2}{\|w\|}$$

### Soft Margin SVM (Primal)

*Reasoning required for USAAIO.*

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i$$

$$\text{subject to} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

- $\xi_i = 0$: point is on or beyond the margin (correctly classified with margin)
- $0 < \xi_i < 1$: point is within the margin but correctly classified
- $\xi_i \geq 1$: point is misclassified
- $C$ controls the tradeoff: large $C$ = few violations (nearly hard margin), small $C$ = more violations allowed (wider margin)

### Dual Formulation

*Reasoning not required for full derivation, but know the result.*

Using Lagrange multipliers $\alpha_i \geq 0$ for each constraint:

**Lagrangian**:
$$\mathcal{L} = \frac{1}{2}\|w\|^2 + C\sum_i \xi_i - \sum_i \alpha_i[y_i(w^Tx_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i$$

Setting $\frac{\partial \mathcal{L}}{\partial w} = 0$ gives $w = \sum_i \alpha_i y_i x_i$.
Setting $\frac{\partial \mathcal{L}}{\partial b} = 0$ gives $\sum_i \alpha_i y_i = 0$.
Setting $\frac{\partial \mathcal{L}}{\partial \xi_i} = 0$ gives $\alpha_i + \mu_i = C$, so $0 \leq \alpha_i \leq C$.

**Dual problem**:
$$\max_\alpha \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i \alpha_j y_i y_j x_i^T x_j$$

$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_i y_i = 0$$

**KKT conditions** tell us:
- $\alpha_i = 0 \Rightarrow$ point is beyond margin (not a support vector)
- $0 < \alpha_i < C \Rightarrow$ point is exactly on margin (support vector, $\xi_i = 0$)
- $\alpha_i = C \Rightarrow$ point is inside margin or misclassified ($\xi_i > 0$)

### Prediction with Dual

$$f(x) = \sum_{i=1}^{n}\alpha_i y_i x_i^T x + b = \sum_{i \in SV}\alpha_i y_i x_i^T x + b$$

Only support vectors ($\alpha_i > 0$) contribute — the sum is sparse!

### Kernel Trick

Replace $x_i^T x_j$ with $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$:

**Dual**: $\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j K(x_i, x_j)$

**Prediction**: $f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$

**Mercer's condition**: $K$ is a valid kernel iff the kernel matrix $K_{ij} = K(x_i, x_j)$ is positive semi-definite for all possible datasets.

### RBF Kernel — Why It Works

$$K(x, z) = \exp(-\gamma\|x - z\|^2) = \exp\left(-\gamma\sum_d (x_d - z_d)^2\right)$$

Expanding the exponential using Taylor series gives an infinite sum of polynomial terms — effectively mapping to an infinite-dimensional feature space. Points that are close ($\|x-z\| \approx 0$) get $K \approx 1$; far points get $K \approx 0$.

$\gamma$ controls the "reach" of each support vector:
- Large $\gamma$: narrow influence, complex boundary (overfitting risk)
- Small $\gamma$: wide influence, smooth boundary (underfitting risk)

---

## Code

### From-Scratch Linear SVM (Gradient Descent on Hinge Loss)

```python
import numpy as np

class LinearSVM:
    def __init__(self, C=1.0, lr=0.01, n_iters=1000):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters
        self.w = None  # (D,)
        self.b = None  # scalar

    def fit(self, X, y):
        """Fit SVM using SGD on hinge loss + L2 regularization."""
        # X: (N, D), y: (N,) in {-1, +1}
        N, D = X.shape
        self.w = np.zeros(D)  # (D,)
        self.b = 0.0

        for _ in range(self.n_iters):
            # Compute margins: y_i * (w^T x_i + b)
            margins = y * (X @ self.w + self.b)  # (N,)

            # Hinge loss gradient
            # For violated points (margin < 1): gradient = -y_i * x_i
            violated = margins < 1  # (N,) boolean

            # Gradient of (1/2)||w||^2 + C * sum(max(0, 1 - y*f))
            dw = self.w - self.C * np.sum(
                (y[violated])[:, np.newaxis] * X[violated], axis=0
            )  # (D,)
            db = -self.C * np.sum(y[violated])  # scalar

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        """Predict class labels."""
        # X: (N, D) -> (N,) in {-1, +1}
        return np.sign(X @ self.w + self.b)

    def decision_function(self, X):
        """Return distance to decision boundary."""
        # X: (N, D) -> (N,)
        return X @ self.w + self.b

    def margin_width(self):
        """Return the margin width."""
        return 2.0 / np.linalg.norm(self.w)

# Example
np.random.seed(42)
# Generate linearly separable data
X_pos = np.random.randn(50, 2) + np.array([2, 2])
X_neg = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])     # (100, 2)
y = np.array([1]*50 + [-1]*50)    # (100,)

svm = LinearSVM(C=1.0, lr=0.001, n_iters=1000)
svm.fit(X, y)
print(f"Margin width: {svm.margin_width():.3f}")
print(f"Accuracy: {np.mean(svm.predict(X) == y):.3f}")
```

### Kernel SVM Using Dual (Simplified)

```python
from scipy.optimize import minimize

def kernel_svm_dual(X, y, C=1.0, kernel='rbf', gamma=1.0):
    """Solve kernel SVM via dual formulation."""
    # X: (N, D), y: (N,) in {-1, +1}
    N = X.shape[0]

    # Compute kernel matrix
    if kernel == 'linear':
        K = X @ X.T  # (N, N)
    elif kernel == 'rbf':
        sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2*X@X.T + np.sum(X**2, axis=1)
        K = np.exp(-gamma * sq_dists)  # (N, N)
    elif kernel == 'poly':
        K = (X @ X.T + 1) ** 3  # degree 3

    # Q matrix for QP
    Q = np.outer(y, y) * K  # (N, N)

    # Objective: minimize -sum(alpha) + 0.5 * alpha^T Q alpha
    def objective(alpha):
        return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

    def gradient(alpha):
        return Q @ alpha - np.ones(N)

    # Constraints: sum(alpha_i * y_i) = 0
    constraints = {'type': 'eq', 'fun': lambda a: a @ y}

    # Bounds: 0 <= alpha_i <= C
    bounds = [(0, C)] * N

    result = minimize(objective, np.zeros(N), jac=gradient,
                     bounds=bounds, constraints=constraints, method='SLSQP')
    alpha = result.x

    # Support vectors: alpha > threshold
    sv_mask = alpha > 1e-5
    sv_alpha = alpha[sv_mask]
    sv_X = X[sv_mask]
    sv_y = y[sv_mask]

    # Compute bias from support vectors on margin
    margin_mask = (alpha > 1e-5) & (alpha < C - 1e-5)
    if margin_mask.sum() > 0:
        idx = np.where(margin_mask)[0][0]
        b = y[idx] - np.sum(sv_alpha * sv_y * K[idx, sv_mask])
    else:
        b = 0.0

    return sv_alpha, sv_X, sv_y, b

```

### scikit-learn Equivalent

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Non-linear data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Scale features (critical for SVM!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# RBF kernel SVM
svm = SVC(kernel='rbf', C=10, gamma=1.0)
svm.fit(X_scaled, y)

print(f"Number of support vectors: {svm.n_support_}")
print(f"Support vector indices: {svm.support_}")
print(f"Accuracy: {svm.score(X_scaled, y):.3f}")

# Compare kernels
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, kernel in zip(axes, ['linear', 'poly', 'rbf']):
    svm = SVC(kernel=kernel, C=10, gamma=1.0, degree=3)
    svm.fit(X_scaled, y)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='RdBu', edgecolors='k', s=30)
    ax.set_title(f'{kernel} kernel (SV={sum(svm.n_support_)})')
plt.tight_layout()
plt.show()
```

---

## Resources

- Vapnik, V. (1963). "Pattern Recognition Using Generalized Portrait Method." *Automation and Remote Control*, 24, 774–780.
- Boser, B., Guyon, I., & Vapnik, V. (1992). "A Training Algorithm for Optimal Margin Classifiers." *COLT*.
- Cortes, C. & Vapnik, V. (1995). "Support-Vector Networks." *Machine Learning*, 20, 273–297.
- ISLR Chapter 9 — Support Vector Machines
- [scikit-learn: SVM](https://scikit-learn.org/stable/modules/svm.html)
