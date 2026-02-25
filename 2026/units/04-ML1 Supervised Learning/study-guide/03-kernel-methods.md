# Kernel Methods

**Prerequisites**: Linear regression, Ridge regression, linear algebra (inner products, PSD matrices)
**USAAIO Relevance**: Kernel trick concept tested frequently; kernel matrix computation and properties (PSD) appear in theory questions; kernel ridge regression in coding rounds

---

## Discovery

### Computing in Infinite Dimensions

Suppose you want to fit a polynomial of degree 1000 to your data. You'd need to construct feature vectors with millions of dimensions — computationally impossible. But what if you could get the *exact same result* without ever computing those features?

This is the **kernel trick**: one of the most elegant ideas in machine learning. It lets you implicitly work in arbitrarily high (even infinite) dimensional feature spaces by only computing pairwise similarities.

### Socratic Warm-Up

1. In linear regression, the solution depends on $X$ only through $X^TX$ and $X^Ty$. What are these in terms of inner products between data points?
2. If $\phi: \mathbb{R}^d \to \mathbb{R}^D$ with $D \gg d$, computing $\phi(x_i)^T\phi(x_j)$ costs $O(D)$. Can we do better?
3. The RBF kernel maps to an *infinite*-dimensional feature space. How can we compute with infinite dimensions?

### Misconception Traps

- **"Kernels are just similarity functions."** — Not any similarity function is a valid kernel. It must correspond to an inner product in *some* feature space (Mercer's condition: the kernel matrix must be PSD).
- **"The kernel trick makes things faster."** — It makes high-dimensional features tractable, but kernel methods scale as $O(n^2)$ or $O(n^3)$ in the number of data points, not features.
- **"You need to know the feature map."** — The whole point is that you don't. You only need the kernel function $K(x_i, x_j)$.

---

## Intuition

### Feature Maps and Inner Products

Consider mapping 2D data to a higher-dimensional space to make it linearly separable:

```
Original space (R^2):            Feature space (R^3):
                                      z = x^2 + y^2
    ○ ○ ○ ○                              │
  ○ ○ ● ● ○ ○                       ○ ○ │ ○ ○
  ○ ● ● ● ○ ○          φ           ○    │    ○
  ○ ○ ● ● ○ ○    ──────────→        ● ● │ ● ●
    ○ ○ ○ ○                          ● ● │ ● ●
                                  ───────┼────── x
  Not linearly                        │
  separable                    Linearly separable!
```

The feature map $\phi(x, y) = (x, y, x^2 + y^2)$ lifts the data into 3D where a hyperplane can separate the classes. The kernel trick says: we don't need to compute $\phi$ explicitly; we just need $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$.

### The Kernel Matrix

Given $n$ data points, the **kernel matrix** (Gram matrix) is:

$$K_{ij} = K(x_i, x_j)$$

```
K = ┌                         ┐
    │ K(x1,x1)  K(x1,x2)  …  │
    │ K(x2,x1)  K(x2,x2)  …  │
    │    ⋮         ⋮       ⋱  │
    │ K(xn,x1)  K(xn,x2)  …  │
    └                         ┘

Properties:
  • Symmetric: K_ij = K_ji
  • Positive semi-definite: v^T K v ≥ 0 for all v
  • Size: (n × n) — depends on data, not feature dim!
```

### Why PSD?

If $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$, then $K = \Phi\Phi^T$ where $\Phi$ has rows $\phi(x_i)^T$. For any vector $v$:

$$v^TKv = v^T\Phi\Phi^Tv = \|\Phi^Tv\|^2 \geq 0$$

So any valid kernel matrix is PSD. Conversely, Mercer's theorem says any PSD kernel function corresponds to some feature map.

---

## Math

### From Linear to Kernel Regression

*[Reasoning required for USAAIO]*

**Step 1**: Start with Ridge regression.

$$\hat{w} = (X^TX + \lambda I)^{-1}X^Ty$$

**Step 2**: Use the matrix identity (Woodbury). The prediction on training data is:

$$\hat{y} = X\hat{w} = X(X^TX + \lambda I)^{-1}X^Ty$$

By the push-through identity: $X(X^TX + \lambda I)^{-1}X^T = (XX^T + \lambda I)^{-1}XX^T$

Actually, the cleaner derivation uses the **representer theorem**: the solution can be written as $w = X^T\alpha$ for some $\alpha \in \mathbb{R}^n$.

**Step 3**: Substitute $w = X^T\alpha$ into the Ridge objective:

$$\mathcal{L}(\alpha) = \frac{1}{n}\|XX^T\alpha - y\|^2 + \lambda \alpha^T XX^T \alpha$$

Let $K = XX^T$ (the kernel matrix with linear kernel):

$$\mathcal{L}(\alpha) = \frac{1}{n}\|K\alpha - y\|^2 + \lambda \alpha^T K \alpha$$

**Step 4**: Take gradient and set to zero:

$$\nabla_\alpha \mathcal{L} = \frac{2}{n}K(K\alpha - y) + 2\lambda K\alpha = 0$$

$$K\alpha - y + n\lambda\alpha = 0$$

$$\boxed{\hat{\alpha} = (K + n\lambda I)^{-1}y}$$

(Again, many references absorb $n$ into $\lambda$: $\hat{\alpha} = (K + \lambda I)^{-1}y$.)

**Step 5**: Prediction on new point $x_*$:

$$\hat{y}(x_*) = w^Tx_* = \alpha^TXx_* = \sum_{i=1}^{n} \alpha_i \cdot x_i^Tx_*$$

Replace $x_i^Tx_*$ with $K(x_i, x_*)$ for nonlinear kernels:

$$\boxed{\hat{y}(x_*) = \sum_{i=1}^{n} \alpha_i K(x_i, x_*) = k_*^T\alpha}$$

where $k_* = [K(x_1, x_*), \ldots, K(x_n, x_*)]^T$.

### Common Kernels

**Linear**: $K(x, x') = x^Tx'$

**Polynomial**: $K(x, x') = (x^Tx' + c)^d$

For $x \in \mathbb{R}^2$, $d=2$, $c=0$: $K(x, x') = (x_1x_1' + x_2x_2')^2$

This corresponds to $\phi(x) = (x_1^2, x_2^2, \sqrt{2}x_1x_2)^T$ — you can verify:

$$\phi(x)^T\phi(x') = x_1^2{x_1'}^2 + x_2^2{x_2'}^2 + 2x_1x_2x_1'x_2' = (x^Tx')^2$$

**RBF (Gaussian)**: $K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$

The RBF kernel corresponds to an *infinite-dimensional* feature map. It can be shown via Taylor expansion of $\exp$:

$$e^{x^Tx'/\sigma^2} = \sum_{k=0}^{\infty} \frac{(x^Tx')^k}{k!\sigma^{2k}}$$

Each term $(x^Tx')^k$ corresponds to a polynomial kernel of degree $k$.

### Kernel Composition Rules

If $K_1$ and $K_2$ are valid kernels, then:
- $K_1 + K_2$ is a valid kernel
- $cK_1$ for $c > 0$ is a valid kernel
- $K_1 \cdot K_2$ is a valid kernel
- $f(x)K_1(x,x')f(x')$ is a valid kernel for any function $f$
- $\exp(K_1)$ is a valid kernel

---

## Code

### NumPy From-Scratch

```python
import numpy as np

def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """X1: (n1, d), X2: (n2, d) → (n1, n2)"""
    return X1 @ X2.T


def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int = 3, c: float = 1.0) -> np.ndarray:
    """X1: (n1, d), X2: (n2, d) → (n1, n2)"""
    return (X1 @ X2.T + c) ** degree


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    RBF (Gaussian) kernel.
    X1: (n1, d), X2: (n2, d) → (n1, n2)

    Uses the expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2a^Tb
    """
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
    dist_sq = sq1 + sq2.T - 2 * X1 @ X2.T         # (n1, n2)
    return np.exp(-dist_sq / (2 * sigma ** 2))


def kernel_ridge_regression(
    X_train: np.ndarray,   # (n, d)
    y_train: np.ndarray,   # (n,)
    X_test: np.ndarray,    # (m, d)
    kernel_fn,
    lam: float = 1.0,
) -> np.ndarray:
    """
    Kernel Ridge Regression.
    Returns: (m,) predictions on test data
    """
    K_train = kernel_fn(X_train, X_train)         # (n, n)
    n = K_train.shape[0]

    # α = (K + λI)^{-1} y
    alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)  # (n,)

    # Predict: k_*^T α
    K_test = kernel_fn(X_test, X_train)            # (m, n)
    return K_test @ alpha                           # (m,)


def is_psd(K: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if kernel matrix is positive semi-definite."""
    eigenvalues = np.linalg.eigvalsh(K)  # (n,) — real eigenvalues for symmetric matrix
    return bool(np.all(eigenvalues >= -tol))


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)                      # (100, 2)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])   # (100,)

    # Kernel ridge with RBF
    from functools import partial
    rbf = partial(rbf_kernel, sigma=1.0)

    y_pred = kernel_ridge_regression(X, y, X, rbf, lam=0.1)
    print(f"Training MSE: {np.mean((y - y_pred) ** 2):.6f}")

    # Verify kernel matrix is PSD
    K = rbf(X, X)
    print(f"Kernel matrix is PSD: {is_psd(K)}")
    print(f"Kernel matrix shape: {K.shape}")
```

### PyTorch Equivalent

```python
import torch

def rbf_kernel_torch(X1: torch.Tensor, X2: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """X1: (n1, d), X2: (n2, d) → (n1, n2)"""
    dist_sq = torch.cdist(X1, X2, p=2) ** 2  # (n1, n2)
    return torch.exp(-dist_sq / (2 * sigma ** 2))

# Kernel Ridge Regression in PyTorch
X = torch.randn(100, 2)
y = torch.sin(X[:, 0]) + 0.5 * torch.cos(X[:, 1])

K = rbf_kernel_torch(X, X, sigma=1.0)       # (100, 100)
lam = 0.1
alpha = torch.linalg.solve(K + lam * torch.eye(100), y)  # (100,)
y_pred = K @ alpha                            # (100,)
```

---

## Resources

- Scholkopf & Smola: *Learning with Kernels* — the definitive reference
- CS229 Notes: Kernels — [cs229.stanford.edu/main_notes.pdf](https://cs229.stanford.edu/main_notes.pdf)
- Hofmann, Scholkopf, Smola: "Kernel Methods in Machine Learning" (survey paper)
